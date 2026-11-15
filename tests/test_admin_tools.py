"""Real request settings, document identity and failure-stage diagnostics."""

import pytest
from pydantic import ValidationError

from wikilens.admin_tools import PaperSettings, RetrievalInspector
from wikilens.paper_live import ModalPaperBridge


@pytest.mark.parametrize(
    "settings",
    [
        {"retrieval_top_k": 101},
        {"retrieval_top_k": True},
        {"max_nodes": 0},
        {"article_tokens": 99999},
        {"score_cutoff": float("nan")},
        {"score_cutoff": float("inf")},
        {"relative_gap": -1},
        {"unknown": 42},
    ],
)
def test_bad_settings_rejected_before_remote_queue(settings):
    with pytest.raises(ValidationError):
        PaperSettings.model_validate(settings)


def test_cutoff_can_be_disabled_without_removing_other_limits():
    result = PaperSettings.model_validate(
        {"score_cutoff": None, "relative_gap": 1.5}
    ).model_dump()
    assert result["score_cutoff"] is None and result["relative_gap"] == 1.5
    assert result["retrieval_top_k"] == 60 and result["max_nodes"] == 25


def article(page_id, rank):
    return {
        "lang": "fi",
        "page_id": page_id,
        "rev_id": page_id + 100,
        "title": "Article " + str(page_id),
        "revision_url": "https://fi.wikipedia.org/w/index.php?oldid="
        + str(page_id + 100),
        "retrieval_rank": rank,
        "distance": 0.2,
        "text": "Literal full article.",
    }


def test_retrieved_rejected_and_selected_documents_remain_distinct():
    a, b = article(1, 1), article(2, 2)
    inspector = RetrievalInspector("Question", "Context", PaperSettings().model_dump())
    inspector.capture("hyde", {"text": "Hypothetical only.", "detected_language": "en"})
    inspector.capture("retrieval", {"articles": [a, b]})
    ranked = [
        {
            **a,
            "id": "chunk-a",
            "text": "Literal",
            "start_char": 0,
            "end_char": 7,
            "rerank_score": 1.2,
        },
        {
            **b,
            "id": "chunk-b",
            "text": "Literal",
            "start_char": 0,
            "end_char": 7,
            "rerank_score": -4,
        },
    ]
    inspector.capture(
        "reranking",
        {
            "ranked": ranked,
            "score_filter": {
                "kept_ids": ["chunk-a"],
                "min_score": -2,
                "counts": {"kept": 1, "input": 2},
            },
        },
    )
    result = inspector.capture("selection", {"nodes": [{**ranked[0], "id": "S1"}]})
    assert result["documents"][0]["selected_count"] == 1
    assert result["documents"][1]["selected_count"] == 0
    assert result["documents"][1]["passing_count"] == 0
    assert inspector.documents["D1"]["passages"][0]["source_id"] == "S1"
    assert inspector.documents["D2"]["text"] == b["text"]
    assert result["hyde"]["role"] == "retrieval_only_never_evidence"
    assert "text" not in result["documents"][0]  # full text loads separately


def test_no_surviving_passages_still_has_all_retrieved_documents():
    a = article(1, 1)
    inspector = RetrievalInspector(
        "Question", "", PaperSettings(score_cutoff=20).model_dump()
    )
    inspector.capture("retrieval", {"articles": [a]})
    inspector.capture(
        "reranking",
        {
            "ranked": [
                {**a, "id": "a", "start_char": 0, "end_char": 5, "rerank_score": 1}
            ],
            "score_filter": {"kept_ids": [], "counts": {"kept": 0, "input": 1}},
        },
    )
    assert inspector.snapshot["stage"] == "reranking"
    assert inspector.snapshot["documents"][0]["passing_count"] == 0
    assert inspector.documents["D1"]["text"] == a["text"]


def test_document_api_checks_identifiers_and_keeps_requests_isolated():
    bridge = ModalPaperBridge.__new__(ModalPaperBridge)
    bridge.job = {}
    bridge.session_id = "session"
    bridge.results = {"session:" + "a" * 32 + ":document:D1": {"title": "A"}}
    assert bridge.diagnostics("a" * 32, "D1")["title"] == "A"
    assert bridge.diagnostics("b" * 32, "D1").get("error")
    with pytest.raises(ValueError):
        bridge.diagnostics("a" * 32, "../D1")
    with pytest.raises(ValueError):
        bridge.diagnostics("../request")


def test_requested_live_defaults_and_6000_token_budget():
    settings = PaperSettings().model_dump()
    assert settings["score_cutoff"] == -4
    assert settings["retrieval_top_k"] == 60
    assert settings["article_tokens"] == 6000
    assert PaperSettings(article_tokens=6000).article_tokens == 6000
    with pytest.raises(ValidationError):
        PaperSettings(article_tokens=6001)
