"""Retrieval provenance and query shape checks; no live DB/model dependencies."""

import base64
import copy
import json
import struct
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from wikilens import research
from wikilens.database import vector_search_sql

VECTOR = struct.pack("<1024f", *([0.01] * 1024))


def article(lang, page, topic):
    return {
        "lang": lang,
        "page_id": page,
        "rev_id": page + 100,
        "wikidata_id": topic,
        "title": f"{lang} {page}",
        "url": f"https://{lang}.wikipedia.org/wiki/Example",
        "text": "A literal source passage. " * 100,
        "truncated": False,
    }


def run_retrieval():
    calls = []
    candidates = {
        lang: [dict(article(lang, i, f"Q{i}"), distance=i / 10) for i in (1, 2, 3)]
        for lang in research.LANGUAGES
    }
    # Add an aligned edition not returned by the vector ranking.
    hydrated = [
        article(lang, i, f"Q{i}") for lang in research.LANGUAGES for i in (1, 2, 3)
    ] + [article("en", 10, "Q1")]
    cursor = MagicMock()

    def execute(sql, params):
        calls.append((sql, params))
        if sql.startswith("EXPLAIN "):
            cursor.fetchall.return_value = [{"key": "article_semantics"}]
        elif sql == vector_search_sql("fi"):
            cursor.fetchall.return_value = copy.deepcopy(candidates[params[1]])
        else:
            cursor.fetchall.return_value = copy.deepcopy(hydrated)

    cursor.execute.side_effect = execute
    db = MagicMock()
    db.__enter__.return_value = db
    db.cursor.return_value.__enter__.return_value = cursor
    with patch.object(research, "connect", return_value=db):
        result = research.retrieve_multilingual("Question?", {"direct": VECTOR})
    return result, calls


def test_language_quotas_and_exact_actual_query_plans():
    result, calls = run_retrieval()
    assert Counter(r["lang"] for r in result["candidates"]) == {
        lang: 3 for lang in research.LANGUAGES
    }
    for lang in research.LANGUAGES:
        params = (VECTOR, lang, 3)
        assert (vector_search_sql(lang), params) in calls
        assert ("EXPLAIN " + vector_search_sql(lang), params) in calls
        assert (
            result["explain_vector_queries"][lang]["direct"][0]["key"]
            == "article_semantics"
        )


def test_alignment_keeps_counterpart_origin_and_exact_revision_provenance():
    result, _ = run_retrieval()
    assert result["aligned_topics"] == ["Q1", "Q2", "Q3"]
    assert all(
        a["origin"] == "vector_candidate"
        for a in result["articles"]
        if a["page_id"] in (1, 2, 3)
    )
    counterpart = next(a for a in result["articles"] if a["page_id"] == 10)
    assert counterpart["origin"] == "wikidata_counterpart"
    assert (
        counterpart["revision_url"] == "https://en.wikipedia.org/w/index.php?oldid=110"
    )
    dossier = research.edition_dossier(result, "Q1")
    assert all(e["status"] == "present" for e in dossier["editions"])
    for edition in dossier["editions"]:
        for source in edition["sources"]:
            original = next(
                a
                for a in result["articles"]
                if (a["lang"], a["page_id"]) == (source["lang"], source["page_id"])
            )
            assert (
                source["text"]
                == original["text"][source["start_char"] : source["end_char"]]
            )
            assert source["selection"] == "article_introduction"


def test_dossier_does_not_claim_absence_for_unexpanded_topic():
    result, _ = run_retrieval()
    with pytest.raises(ValueError, match="fully loaded"):
        research.edition_dossier(result, "Q999")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_vectors_rejected_before_database_access(value):
    invalid = struct.pack("<1024f", value, *([0.0] * 1023))
    with patch.object(research, "connect") as connect:
        with pytest.raises(ValueError):
            research.retrieve_multilingual("Question?", {"direct": invalid})
        connect.assert_not_called()


def test_cached_vectors_never_silently_invoke_a_model(tmp_path):
    cache = tmp_path / "vectors.json"
    cache.write_text(json.dumps({"Question?": base64.b64encode(VECTOR).decode()}))
    with (
        patch.object(research, "CACHE", cache),
        patch("wikilens.database.model_request") as model,
    ):
        assert research.cached_vector("Question?") == VECTOR
        with pytest.raises(ValueError, match="listed example questions"):
            research.cached_vector("A new question")
        model.assert_not_called()
    cache.write_text(json.dumps({"Question?": base64.b64encode(b"short").decode()}))
    with (
        patch.object(research, "CACHE", cache),
        pytest.raises(ValueError, match="1024"),
    ):
        research.cached_vector("Question?")
