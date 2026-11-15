"""Synthetic capture/DB tests; these are not actual inference or human reviews."""

import copy
import json
from unittest.mock import MagicMock, patch

import pytest

from test_capture import capture as synthetic_capture
from wikilens.capture import build_fixture
from wikilens import provenance
from wikilens.web import create_app


def fixture():
    record, review = synthetic_capture()
    record["stages"]["retrieval"]["articles"][0]["origin"] = "wikidata_counterpart"
    record["stages"]["retrieval"]["explain_vector_queries"] = {
        "en": {"direct": [{"key": "article_semantics"}]}
    }
    return build_fixture(record, review)


def cursor():
    value = MagicMock()
    value.fetchone.return_value = None
    return value


def test_import_revalidates_raw_sources_and_display_before_opening_db():
    original = fixture()
    assert provenance.validate_capture(original) == original
    corrupted = copy.deepcopy(original)
    corrupted["examples"][0]["answer"][0]["text"] = "Invented displayed claim."
    with patch.object(provenance, "connect") as connect:
        with pytest.raises(ValueError, match="differs"):
            provenance.import_capture(corrupted)
        connect.assert_not_called()
    corrupted = copy.deepcopy(original)
    corrupted["raw_record"]["stages"]["ranking"]["selected"][0]["text"] = (
        "Invented quotation."
    )
    with pytest.raises(ValueError):
        provenance.validate_capture(corrupted)


def test_normalized_rows_preserve_exact_quote_offsets_origin_and_review():
    value = fixture()
    cur = cursor()
    result = provenance._insert_capture(cur, value)
    assert result["imported"] is True
    calls = cur.execute.call_args_list
    evidence = next(
        c.args[1] for c in calls if "INSERT INTO provenance_evidence" in c.args[0]
    )
    source = value["raw_record"]["stages"]["quotations"]["sources"][0]
    assert evidence[9] == "wikidata_counterpart"
    assert evidence[10:16] == (
        source["start_char"],
        source["end_char"],
        source["text"],
        source["quote"],
        source["quote_start_char"],
        source["quote_end_char"],
    )
    run = next(c.args[1] for c in calls if "INSERT INTO provenance_run" in c.args[0])
    assert (
        json.loads(run[7])["retrieval"]["explain_vector_queries"]["en"]["direct"][0][
            "key"
        ]
        == "article_semantics"
    )
    review = next(
        c.args[1] for c in calls if "INSERT INTO provenance_review " in c.args[0]
    )
    assert review[1:3] == ("Test reviewer", "2026-11-14T02:00:00+00:00")
    assert all("UPDATE " not in c.args[0] and "DELETE " not in c.args[0] for c in calls)


def test_identical_import_is_idempotent_and_review_conflict_never_overwrites():
    value = fixture()
    report_hash = value["examples"][0]["raw_record_sha256"]
    cur = cursor()
    cur.fetchone.return_value = {
        "report_hash": report_hash,
        "capture_hash": provenance._hash(value),
    }
    result = provenance._insert_capture(cur, value)
    assert result["imported"] is False
    assert cur.execute.call_count == 1
    cur.fetchone.return_value["capture_hash"] = "0" * 64
    with pytest.raises(ValueError, match="different immutable review"):
        provenance._insert_capture(cur, value)


def test_import_commits_all_rows_once_or_rolls_everything_back():
    value = fixture()
    db = MagicMock()
    db.cursor.return_value.__enter__.return_value = cursor()
    with (
        patch.object(provenance, "initialize_provenance"),
        patch.object(provenance, "connect", return_value=db),
    ):
        assert provenance.import_capture(value)["imported"]
    db.begin.assert_called_once()
    db.commit.assert_called_once()
    db.rollback.assert_not_called()
    db.close.assert_called_once()
    db = MagicMock()
    with (
        patch.object(provenance, "initialize_provenance"),
        patch.object(provenance, "connect", return_value=db),
        patch.object(
            provenance, "_insert_capture", side_effect=RuntimeError("test failure")
        ),
    ):
        with pytest.raises(RuntimeError):
            provenance.import_capture(value)
    db.rollback.assert_called_once()
    db.commit.assert_not_called()
    db.close.assert_called_once()


@pytest.mark.parametrize("value", ["bad", "A" * 64, "../private", None])
def test_inspection_rejects_invalid_hash_before_database(value):
    with patch.object(provenance, "connect") as connect:
        with pytest.raises(ValueError, match="SHA-256"):
            provenance.inspect_run(value)
        connect.assert_not_called()


def test_read_only_api_reports_missing_or_reconstructed_run():
    client = create_app().test_client()
    report_hash = "a" * 64
    with patch.object(provenance, "inspect_run", return_value=None):
        assert client.get("/api/runs/" + report_hash).status_code == 404
    reconstructed = {
        "report_hash": report_hash,
        "claims": [{"text": "Reviewed test sentence."}],
    }
    with patch.object(provenance, "inspect_run", return_value=reconstructed):
        response = client.get("/api/runs/" + report_hash)
        assert response.status_code == 200 and response.json == reconstructed
    assert client.post("/api/runs/" + report_hash).status_code == 405
