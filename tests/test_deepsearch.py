"""Controller safety and iterative evidence flow, without GPU/DB/network."""

from unittest.mock import Mock, patch

import pytest

from wikilens.deepsearch import LiveDeepSearch, citation_errors, run_deepsearch


def node(page=1):
    return {
        "id": "S1",
        "lang": "en",
        "page_id": page,
        "rev_id": 10,
        "start_char": 0,
        "end_char": 14,
        "text": "Exact evidence",
        "title": "Fixture",
    }


def answer(nodes):
    return {
        "status": "draft_unverified",
        "sentences": [
            {
                "text": "A supported claim.",
                "citations": [
                    {"source_id": n["id"], "quote": n["text"]} for n in nodes
                ],
            }
        ],
    }


def review(ok=True, queries=None):
    return {
        "score": 4 if ok else 2,
        "answers_question": ok,
        "supported_by_evidence": True,
        "critique": "Complete" if ok else "Find the cause",
        "unsupported_claims": [],
        "gaps": [] if ok else ["Cause"],
        "follow_up_queries": queries or [],
    }


def run(**overrides):
    options = {
        "search": lambda q: [node()],
        "select": lambda q, pool: pool,
        "draft": lambda q, nodes, previous: answer(nodes),
        "evaluate": lambda *args: review(),
        "progress": None,
    }
    options.update(overrides)
    return run_deepsearch("Original question", **options)


def test_reflection_drives_search_and_preserves_original_question():
    search = Mock(side_effect=[[node()], [node(), node(2)]])
    draft = Mock(side_effect=lambda q, nodes, prev: answer(nodes))
    evaluator = Mock(side_effect=[review(False, ["Cause query"]), review()])
    result = run(search=search, draft=draft, evaluate=evaluator)
    assert result["status"] == "model_satisfied"
    assert result["queries"] == ["Original question", "Cause query"]
    assert (
        len(result["evidence_pool"]) == 2
    )  # repeated S1 is deduplicated by provenance
    assert all(call.args[0] == "Original question" for call in draft.call_args_list)
    assert draft.call_args_list[1].args[2]["gaps"] == ["Cause"]


@pytest.mark.parametrize(
    "settings,status",
    [({"max_rounds": 1}, "round_limit"), ({"max_queries": 1}, "query_limit")],
)
def test_limits(settings, status):
    result = run(settings=settings, evaluate=lambda *a: review(False, ["New query"]))
    assert result["status"] == status
    assert len(result["queries"]) == 1


def test_repeated_queries_stop_and_empty_evidence_abstains():
    assert (
        run(evaluate=lambda *a: review(False, ["  ORIGINAL   question "]))["status"]
        == "no_new_queries"
    )
    draft = Mock()
    assert run(search=lambda q: [], draft=draft)["status"] == "insufficient_evidence"
    draft.assert_not_called()


def test_critic_cannot_override_bad_citations():
    bad = answer([node()])
    bad["sentences"][0]["citations"][0]["quote"] = "Invented quotation"
    result = run(draft=lambda *a: bad)
    assert result["status"] == "no_new_queries"
    assert result["rounds"][0]["citation_errors"]
    assert citation_errors({"status": "failed", "sentences": []}, [node()])


def test_failures_keep_previous_answer_and_checkpoint():
    saved = []
    result = run(
        evaluate=Mock(side_effect=ValueError("Malformed JSON")),
        checkpoint=lambda r: saved.append(r["status"]),
    )
    assert result["status"] == "failed"
    assert result["answer"]["sentences"]
    assert saved[-1] == "failed"


def test_deadline_prevents_next_stage():
    clock = [0.0]

    def slow_search(q):
        clock[0] = 10.0
        return [node()]

    draft = Mock()
    with patch("wikilens.deepsearch.time.monotonic", side_effect=lambda: clock[0]):
        result = run(search=slow_search, draft=draft, settings={"max_seconds": 1.0})
    assert result["status"] == "time_limit"
    draft.assert_not_called()


def test_live_draft_uses_existing_pipeline_and_revision_data():
    engine = Mock()
    adapter = LiveDeepSearch(engine)
    with patch(
        "wikilens.citation_pipelines.run_pipeline", return_value=answer([node()])
    ) as pipeline:
        adapter.draft("Original question", [node()], review(False))
    import json

    source = pipeline.call_args.args[0]["variants"]["cross_encoder"]["run"]
    data = json.loads(source["stages"]["generation"]["messages"][1]["content"])
    assert data["revision_review"]["gaps"] == ["Cause"]
    assert data["question"] == "Original question"
    assert data["passages"][0]["text"] == "Exact evidence"


def test_live_adapter_through_real_citation_pipeline_with_mocked_model():
    engine = Mock()
    engine.models.generate.side_effect = [
        "Exact evidence",
        '{"items":[{"claim_id":"P1","span_ids":["S1_1"]}]}',
    ]
    engine.models.stats.return_value = {"mocked": True}

    def lookup(runtime, question, units, nodes, **kwargs):
        for unit in units:
            unit["reference_candidates"] = [{"source_id": "S1", "score": 1.0}]
        return {"seconds": 0.0}

    engine.models.lookup.side_effect = lookup
    source = {**node(), "revision_url": "https://en.wikipedia.org/w/index.php?oldid=10"}
    adapter = LiveDeepSearch(engine)
    result = adapter.draft("Original question", [source], review(False))
    assert result["status"] == "draft_unverified"
    assert result["sentences"][0]["citations"][0]["quote"] == source["text"]
    assert not citation_errors(result, [source])
    messages = engine.models.generate.call_args_list[0].args[0]
    assert "fallible editorial feedback" in messages[0]["content"]
