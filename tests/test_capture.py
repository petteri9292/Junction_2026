import copy
import hashlib
import json

import pytest

from wikilens.capture import build_fixture, main
from wikilens.grounding import GroundingError, chunk_articles, validate_quotations

TEXT = "Construction of the fortress began in 1748 under the Swedish Crown."


def capture():
    article = {
        "lang": "en",
        "page_id": 1,
        "rev_id": 2,
        "wikidata_id": "Q1",
        "title": "Fortress",
        "text": TEXT,
    }
    passages = chunk_articles([article])
    selections = [{"source_id": passages[0]["id"], "quote": TEXT}]
    sources = validate_quotations(selections, passages)
    answer = {
        "insufficient_evidence": False,
        "sentences": [
            {"text": "Construction began in 1748.", "sources": [passages[0]["id"]]}
        ],
    }
    record = {
        "status": "draft",
        "question": "When did construction begin?",
        "created_at": "2026-11-14T01:00:00+00:00",
        "dataset_release": "s-2026-10",
        "seconds": 10.0,
        "runtime": {
            "models": {"bge": "BAAI/bge-m3", "qwen": "Qwen/Qwen3-1.7B"},
            "revisions": {"bge": "bge-revision", "qwen": "qwen-revision"},
            "records": [
                {"stage": s, "status": "completed", "seconds": 1.0}
                for s in ("qwen", "bge")
            ],
        },
        "stages": {
            "hyde": {
                "status": "completed",
                "text": "Hypothetical retrieval context.",
                "raw_output": "Hypothetical retrieval context.",
                "role": "retrieval_only_never_evidence",
                "seconds": 1.0,
            },
            "retrieval": {
                "status": "completed",
                "articles": [article],
                "embedding_and_retrieval_seconds": 2.0,
            },
            "ranking": {"status": "completed", "selected": passages, "seconds": 2.0},
            "quotations": {
                "status": "completed",
                "raw_output": json.dumps(selections),
                "sources": sources,
                "seconds": 1.0,
            },
            "synthesis": {
                "status": "draft",
                "raw_output": json.dumps(answer),
                "sentences": answer["sentences"],
                "seconds": 1.0,
            },
        },
    }
    review = {
        "reviewer": "Test reviewer",
        "reviewed_at": "2026-11-14T02:00:00+00:00",
        "claims": [{"index": 0, "supported": True, "source_ids": [passages[0]["id"]]}],
    }
    return record, review


def test_consistent_capture_and_explicit_review_create_standalone_traceable_fixture():
    record, review = capture()
    before = copy.deepcopy(record)
    output = build_fixture(record, review)
    example = output["examples"][0]
    assert output["schema_version"] == 2
    assert example["status"] == "reviewed_pipeline_capture"
    assert example["answer"][0]["text"] == "Construction began in 1748."
    assert example["evidence"][0]["passage"] == TEXT
    assert (
        example["evidence"][0]["revision_url"]
        == "https://en.wikipedia.org/w/index.php?oldid=2"
    )
    assert example["verification"]["claim_support_reviewed"] is True
    assert example["hyde_role"] == "retrieval_only_never_evidence"
    assert record == before
    canonical = json.dumps(
        record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    assert (
        example["raw_record_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    )
    assert output["raw_record"] == record


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(status="placeholder"),
        lambda r: r.update(origin="hand_authored"),
        lambda r: r["runtime"].update(mocked=True),
        lambda r: r["runtime"].update(revisions={}),
        lambda r: r["stages"]["quotations"].update(status="rejected"),
        lambda r: r["stages"]["quotations"]["sources"][0].update(
            quote="Invented quote."
        ),
        lambda r: r["stages"]["retrieval"].update(articles=[]),
        lambda r: r["stages"]["ranking"]["selected"][0].update(
            text="Tampered source text."
        ),
        lambda r: r["stages"]["synthesis"]["sentences"][0].update(
            text="Tampered answer."
        ),
        lambda r: r["stages"]["synthesis"].update(
            raw_output='{"insufficient_evidence":true,"sentences":[]}'
        ),
        lambda r: r["stages"]["hyde"].update(role="evidence"),
        lambda r: r.update(seconds=float("nan")),
    ],
)
def test_incomplete_tampered_or_hand_authored_runs_are_rejected(change):
    record, review = capture()
    change(record)
    with pytest.raises(GroundingError):
        build_fixture(record, review)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["claims"][0].update(supported=False),
        lambda r: r["claims"][0].update(supported=1),
        lambda r: r["claims"][0].update(source_ids=["unknown"]),
        lambda r: r["claims"][0].update(source_ids=[]),
        lambda r: r["claims"][0].update(index=True),
        lambda r: r.update(claims=[]),
        lambda r: r.update(reviewer=" "),
        lambda r: r.update(reviewed_at="2026-11-14T02:00:00"),
        lambda r: r.update(reviewed_at="2026-11-13T02:00:00+00:00"),
    ],
)
def test_review_must_explicitly_cover_every_sentence_and_its_evidence(change):
    record, review = capture()
    change(review)
    with pytest.raises(GroundingError):
        build_fixture(record, review)


def test_unknown_generated_citation_rejected_even_if_reviewer_approves_it():
    record, review = capture()
    answer = {
        "insufficient_evidence": False,
        "sentences": [{"text": "An answer.", "sources": ["invented"]}],
    }
    record["stages"]["synthesis"].update(
        raw_output=json.dumps(answer), sentences=answer["sentences"]
    )
    review["claims"][0]["source_ids"] = ["invented"]
    with pytest.raises(GroundingError):
        build_fixture(record, review)


def test_cli_refuses_to_overwrite_existing_fixture(tmp_path):
    record, review = capture()
    report_path, review_path, output_path = [
        tmp_path / f for f in ("report.json", "review.json", "output.json")
    ]
    report_path.write_text(json.dumps(record))
    review_path.write_text(json.dumps(review))
    main([str(report_path), str(review_path), str(output_path)])
    first = output_path.read_text()
    with pytest.raises(FileExistsError):
        main([str(report_path), str(review_path), str(output_path)])
    assert output_path.read_text() == first
