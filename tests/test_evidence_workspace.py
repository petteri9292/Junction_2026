"""Measured-retrieval replay contract and literal dataset provenance checks."""

import json
from pathlib import Path

import pytest

from wikilens.web import create_app

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = json.loads((ROOT / "wikilens/static/evidence-captures.json").read_text())


def test_workspace_and_recorded_captures_work_without_database_or_models():
    client = create_app().test_client()
    response = client.get("/evidence")
    assert response.status_code == 200
    assert b"Original source text" in response.data
    response = client.get("/static/evidence-captures.json")
    assert response.status_code == 200
    assert response.json["kind"] == "measured_retrieval_capture"
    assert response.json["answer_generated"] is False
    assert len(response.json["examples"]) == 3


def test_saved_examples_preserve_measured_trace_and_all_edition_states():
    for example in CAPTURES["examples"]:
        assert example["retrieval_seconds"] > 0
        assert len(example["candidates"]) == 9
        assert set(example["explain_vector_queries"]) == {"fi", "en", "sv"}
        for dossier in example["dossiers"]:
            assert {e["lang"] for e in dossier["editions"]} == {"fi", "en", "sv"}
            for edition in dossier["editions"]:
                assert (edition["status"] == "present") == bool(edition["sources"])
                for source in edition["sources"]:
                    assert source["selection"] == "article_introduction"
                    assert source["origin"] in {
                        "vector_candidate",
                        "wikidata_counterpart",
                    }
                    assert (
                        source["revision_url"]
                        == f"https://{source['lang']}.wikipedia.org/w/index.php?oldid={source['rev_id']}"
                    )


def test_replay_excerpts_and_revisions_match_original_local_parquet():
    import pyarrow.parquet as pq

    directory = ROOT / "data/s-2026-10"
    if not (directory / "articles-fi.parquet").exists():
        pytest.skip("Optional original dataset not installed")
    articles = {}
    for lang in ("fi", "en", "sv"):
        table = pq.read_table(
            directory / f"articles-{lang}.parquet",
            columns=["page_id", "rev_id", "wikidata_id", "title", "text"],
        )
        articles[lang] = {r["page_id"]: r for r in table.to_pylist()}
    for example in CAPTURES["examples"]:
        for dossier in example["dossiers"]:
            for edition in dossier["editions"]:
                for source in edition["sources"]:
                    original = articles[source["lang"]][source["page_id"]]
                    assert (
                        original["text"][source["start_char"] : source["end_char"]]
                        == source["text"]
                    )
                    assert all(
                        original[k] == source[k]
                        for k in ("rev_id", "wikidata_id", "title")
                    )
