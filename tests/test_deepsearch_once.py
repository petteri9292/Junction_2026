"""Bounded, optional deep search: mocked inference and retrieval only."""

import json
from types import SimpleNamespace
import pytest
from wikilens.deepsearch_once import review_and_search_once, merge_articles


def node(page=1):
    text = f"Literal source text for article {page}."
    return dict(
        id=f"S{page}",
        lang="fi",
        page_id=page,
        rev_id=100 + page,
        title=f"Article {page}",
        text=text,
        start_char=0,
        end_char=len(text),
        retrieval_rank=page,
        revision_url=f"https://fi.wikipedia.org/w/index.php?oldid={100 + page}",
    )


class Runtime:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []
        self.profiles = []
        self.tokenizer = object()
        self.lookup = lambda *a, **k: []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.decision

    def stats(self):
        return {"revisions": {}}


def decision(good=False, query="Puuttuva asia"):
    return json.dumps(
        dict(
            good_enough=good,
            reason="A bounded evidence assessment.",
            follow_up_query=query,
        )
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        (decision(True), "sufficient"),
        (decision(False, None), "no_useful_query"),
        ("invalid json", "fallback"),
        (decision(False, "Original question"), "no_useful_query"),
        (decision(False, "First\nSecond"), "fallback"),
    ],
)
def test_sufficient_invalid_or_unhelpful_review_makes_no_query(raw, expected):
    runtime = Runtime(raw)
    calls = []
    articles, report = review_and_search_once(
        runtime, "Original question", "Focus", [node()], lambda q: calls.append(q)
    )
    assert articles == [] and calls == [] and report["extra_queries"] == 0
    assert report["status"] == expected and len(runtime.calls) == 1
    inputs = json.loads(runtime.calls[0][0][1]["content"])
    assert (
        inputs["question"] == "Original question"
        and inputs["passages"][0]["text"] == node()["text"]
    )
    assert runtime.calls[0][1]["output_schema"]["properties"]["follow_up_query"]


def test_reviewer_can_request_only_one_search_and_failures_fall_back():
    calls = []

    def search(q):
        calls.append(q)
        raise RuntimeError("Retrieval unavailable")

    articles, report = review_and_search_once(
        Runtime(decision()), "Original question", "Focus", [node()], search
    )
    assert (
        articles == []
        and len(calls) == 1
        and report["extra_queries"] == 1
        and report["status"] == "fallback"
    )


def test_merging_revisions_does_not_duplicate_or_rewrite_source_text():
    original = node()
    more = node(2)
    merged = merge_articles([original], [original, more])
    assert len(merged) == 2 and [n["text"] for n in merged] == [
        original["text"],
        more["text"],
    ]
    assert original == node()


@pytest.mark.parametrize(
    "enabled,raw,extra_calls,final_count",
    [
        (False, decision(), 0, 1),
        (True, decision(True), 0, 1),
        (True, decision(), 1, 2),
        (True, "bad json", 0, 1),
    ],
)
def test_live_engine_keeps_baseline_path_and_only_expands_once(
    monkeypatch, tmp_path, enabled, raw, extra_calls, final_count
):
    pytest.importorskip("gradio")
    import wikilens.gradio_demo as live
    import wikilens.hyde_search as hyde
    import wikilens.reranker_comparison as ranking
    import wikilens.focus_evidence as focus
    import wikilens.reranker_policy as policy
    import wikilens.citation_pipelines as citations

    (tmp_path / "data").mkdir()
    monkeypatch.setattr(live, "ROOT", tmp_path)
    runtime = Runtime(raw)
    engine = live.ProductEngine.__new__(live.ProductEngine)
    engine.models = runtime
    engine.available_source_languages = {"fi"}
    engine.corpus_info = {"dataset": "test"}
    engine.embedding_revision = "test"
    monkeypatch.setattr(
        hyde,
        "generate_focused_hyde",
        lambda *a, **k: dict(
            status="completed",
            text="Finnish hypothetical text",
            answer_language="Finnish",
            detected_language="fi",
            seconds=0,
        ),
    )
    monkeypatch.setattr(
        hyde, "retrieve_hyde", lambda *a, **k: dict(articles=[node()], seconds=0)
    )
    searches = []

    def extra(*a, **kw):
        searches.append(a[1])
        return dict(articles=[node(), node(2)], seconds=0)

    monkeypatch.setattr(hyde, "retrieve_direct", extra)
    monkeypatch.setattr(ranking, "prepare_chunks", lambda articles, **kw: articles)
    engine.rank = lambda q, f, chunks, **kw: dict(
        nodes=chunks, ranked=chunks, seconds=0, score_filter={}, ranking_hint_role=None
    )
    monkeypatch.setattr(
        focus, "filter_focused_nodes", lambda r, q, f, n: (n, {"seconds": 0})
    )
    monkeypatch.setattr(policy, "pack_generation_nodes", lambda n, *a: (n, {}))
    captured = []

    def generate(capture, *a, **kw):
        source = capture["variants"]["cross_encoder"]["run"]
        captured.append(source)
        return {
            "status": "draft_unverified",
            "sentences": [{"text": "Mock answer"}],
            "runtime": {"revisions": {}},
            "stages": {},
        }

    monkeypatch.setattr(citations, "run_pipeline", generate)
    result = engine.run(
        "Original question",
        "Auto",
        "Finnish",
        "Original focus",
        60,
        500,
        6000,
        lambda *a, **k: None,
        deep_search=enabled,
    )
    assert len(searches) == extra_calls and len(runtime.calls) == int(enabled)
    assert (
        captured[0]["question"] == "Original question"
        and captured[0]["search_focus"] == "Original focus"
    )
    assert len(captured[0]["stages"]["postprocessing"]["nodes"]) == final_count
    assert len(captured) == 1 and result["deep_search"]["extra_queries"] == extra_calls
