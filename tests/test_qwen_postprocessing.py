import json
import pytest
from wikilens.grounding import GroundingError, chunk_articles
from wikilens.qwen_postprocessing import validate_compression, qwen_postprocess
from wikilens.node_pipeline import citation_context

TEXT = "The war began in November 1939. It ended with a peace treaty in March 1940."
ARTICLE = dict(lang="en", page_id=1, rev_id=2, title="War", text=TEXT)
PASSAGE = chunk_articles([ARTICLE])[0]


@pytest.mark.parametrize("score", [True, -0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_rejects_invalid_scores(score):
    with pytest.raises(GroundingError):
        validate_compression({"score": score, "excerpts": []}, PASSAGE)


@pytest.mark.parametrize(
    "excerpts",
    [
        ["A fabricated peace agreement."],
        ["The war began", "The war began"],
        ["x" * 701],
        [""],
    ],
)
def test_rejects_untraceable_or_invalid_compression(excerpts):
    with pytest.raises(GroundingError):
        validate_compression({"score": 0.8, "excerpts": excerpts}, PASSAGE)


class Runtime:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, messages, **kw):
        self.calls.append((messages, kw))
        return next(self.outputs)

    def embed(self, *a, **kw):
        raise AssertionError("BGE must not rerank or compress nodes")


def test_qwen_scores_sort_separate_excerpts_keep_literal_citations():
    other = {
        **ARTICLE,
        "page_id": 3,
        "text": "A broad background passage with less useful information.",
    }
    out = [
        json.dumps(
            {
                "score": 0.9,
                "excerpts": [
                    "The war began in November 1939.",
                    "It ended with a peace treaty in March 1940.",
                ],
            }
        ),
        json.dumps({"score": 0.3, "excerpts": [other["text"]]}),
    ]
    runtime = Runtime(out)
    result = qwen_postprocess(runtime, "When and how?", [ARTICLE, other])
    assert result["status"] == "completed" and len(result["nodes"]) == 2
    assert result["nodes"][0]["id"] == "S1"
    for node in result["nodes"]:
        assert ARTICLE["text"][node["start_char"] : node["end_char"]] == node["text"]
        assert node["revision_url"].endswith("oldid=2")
        assert (
            citation_context(result["nodes"], [ARTICLE, other], node["id"])["passage"]
            == node["text"]
        )
    assert len(runtime.calls) == 2
    assert all(k["max_time"] == 15 for _, k in runtime.calls)


def test_bad_output_rejected_without_becoming_zero_relevance_or_answer_context():
    r = qwen_postprocess(Runtime(["not JSON"]), "Query", [ARTICLE])
    assert r["status"] == "rejected" and r["nodes"] == []
    assert r["judgments"][0]["status"] == "rejected"
    assert "score" not in r["judgments"][0]


def test_zero_score_is_explicit_insufficiency_and_no_input_has_no_model_call():
    r = qwen_postprocess(Runtime(['{"score":0,"excerpts":[]}']), "Query", [ARTICLE])
    assert r["status"] == "insufficient_evidence"
    runtime = Runtime([])
    assert qwen_postprocess(runtime, "Query", [])["status"] == "insufficient_evidence"
    assert runtime.calls == []


def test_call_cap_fails_before_model_without_leading_truncation():
    runtime = Runtime([])
    with pytest.raises(ValueError, match="selected passages exceed max_calls"):
        qwen_postprocess(
            runtime, "Query", [{**ARTICLE, "text": TEXT * 100}], max_calls=1
        )
    assert runtime.calls == []


def test_expired_budget_keeps_judgments_but_does_not_select_partial_context(
    monkeypatch,
):
    import wikilens.qwen_postprocessing as module

    ticks = iter([0, 0, 601, 602])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    other = {**ARTICLE, "page_id": 3}
    r = qwen_postprocess(
        Runtime([json.dumps({"score": 0.9, "excerpts": [TEXT]})]),
        "Query",
        [ARTICLE, other],
    )
    assert r["status"] == "budget_exhausted" and len(r["judgments"]) == 1
    assert r["nodes"] == [] and r["ranked"]


def test_only_ten_passages_scored_and_retrieved_articles_get_first_pass():
    articles = [
        {
            **ARTICLE,
            "page_id": i + 1,
            "title": f"Article {i + 1}",
            "retrieval_rank": i + 1,
            "text": TEXT * 100,
        }
        for i in range(12)
    ]
    runtime = Runtime(['{"score":0,"excerpts":[]}'] * 10)
    result = qwen_postprocess(runtime, "Query", articles)
    assert len(runtime.calls) == 10 and result["candidate_count"] == 10
    assert result["available_passage_count"] > 10
    assert [json.loads(m[1]["content"])["title"] for m, k in runtime.calls] == [
        f"Article {i + 1}" for i in range(10)
    ]
    assert all(p.endswith("-0-1600") for p in result["candidate_ids"])


def test_additional_windows_only_after_first_pass_over_retrieved_articles():
    articles = [
        {
            **ARTICLE,
            "page_id": i + 1,
            "title": f"Article {i + 1}",
            "retrieval_rank": i + 1,
            "text": TEXT * 100,
        }
        for i in range(2)
    ]
    runtime = Runtime(['{"score":0,"excerpts":[]}'] * 4)
    result = qwen_postprocess(runtime, "Query", articles, max_passages=4)
    assert [json.loads(m[1]["content"])["title"] for m, k in runtime.calls] == [
        "Article 1",
        "Article 2",
        "Article 1",
        "Article 2",
    ]
    assert result["candidate_count"] == 4
