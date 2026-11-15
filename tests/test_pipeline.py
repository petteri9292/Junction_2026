"""Pipeline integration using synthetic source text and mocked model/database calls."""

import json
from unittest.mock import patch

import pytest

from wikilens.grounding import validate_quotations
from wikilens.pipeline import (
    generate_hyde,
    rank_evidence,
    retrieve,
    run_pipeline,
    select_quotes,
    synthesize,
)

TEXT = "Construction of the fortress began in 1748 under the Swedish Crown."


def article(lang="fi", page_id=1):
    return {
        "lang": lang,
        "page_id": page_id,
        "rev_id": 10,
        "title": "Fortress",
        "url": f"https://{lang}.wikipedia.org/wiki/Fortress",
        "wikidata_id": "Q1",
        "text": TEXT,
    }


class Runtime:
    def __init__(self):
        self.generations = []
        self.embeddings = []
        self.closed = False
        self.bad_quotes = False
        self.bad_answer = False

    def generate(self, messages, **kwargs):
        self.generations.append((messages, kwargs))
        prompt = messages[0]["content"]
        if "encyclopedia-style" in prompt:
            return "HYPOTHETICAL retrieval-only context."
        data = json.loads(messages[1]["content"])
        if "Select evidence" in prompt:
            source_id = data["passages"][0]["source_id"]
            return json.dumps(
                [
                    {
                        "source_id": "invented" if self.bad_quotes else source_id,
                        "quote": TEXT,
                    }
                ]
            )
        source_id = data["evidence"][0]["source_id"]
        return json.dumps(
            {
                "insufficient_evidence": False,
                "sentences": [
                    {
                        "text": "Construction began in 1748.",
                        "sources": ["invented" if self.bad_answer else source_id],
                    }
                ],
            }
        )

    def embed(self, texts, **kwargs):
        self.embeddings.append((texts, kwargs))
        return [[1.0] + [0.0] * 1023 for _ in texts]

    def close(self):
        self.closed = True

    def stats(self):
        return {"mocked": True, "closed": self.closed}


def test_full_pipeline_capture_uses_balanced_retrieval_and_excludes_hyde_evidence():
    runtime = Runtime()
    rows = [article(lang, index) for index, lang in enumerate(("fi", "en", "sv"), 1)]
    with patch(
        "wikilens.research.retrieve_multilingual", return_value={"articles": rows}
    ) as sql:
        record = run_pipeline("When was the fortress built?", runtime)
    assert record["status"] == "draft"
    assert runtime.closed
    assert sql.call_args.kwargs == {"per_language": 3, "topic_limit": 3}
    packed = sql.call_args.args[1]
    assert set(packed) == {"direct", "hyde"}
    assert all(isinstance(v, bytes) and len(v) == 4096 for v in packed.values())
    assert len(runtime.generations) == 3
    assert all(
        "HYPOTHETICAL" not in messages[1]["content"]
        for messages, _ in runtime.generations[1:]
    )
    assert all(config["max_time"] <= 60 for _, config in runtime.generations)
    synthesis = record["stages"]["synthesis"]
    assert synthesis["claim_matrix"][0]["editions"]["fi"] == "cited"
    assert synthesis["claim_matrix"][0]["editions"]["en"] == "not_cited"
    assert synthesis["claim_support_reviewed"] is False
    json.dumps(record)


def test_failed_quote_validation_preserves_raw_and_skips_synthesis():
    runtime = Runtime()
    runtime.bad_quotes = True
    with patch(
        "wikilens.research.retrieve_multilingual",
        return_value={"articles": [article()]},
    ):
        record = run_pipeline("When?", runtime)
    assert record["status"] == "rejected"
    assert "invented" in record["stages"]["quotations"]["raw_output"]
    assert "synthesis" not in record["stages"]
    assert len(runtime.generations) == 2 and runtime.closed


def test_synthesis_unknown_id_is_rejected_and_raw_preserved():
    runtime = Runtime()
    runtime.bad_answer = True
    passage = {"id": "source1", "text": TEXT, "lang": "sv"}
    verified = validate_quotations([{"source_id": "source1", "quote": TEXT}], [passage])
    result = synthesize(runtime, "When?", verified)
    assert result["status"] == "rejected"
    assert result["sentences"] == []
    assert "invented" in result["raw_output"]


def test_no_candidates_abstains_without_quote_or_synthesis_generation():
    runtime = Runtime()
    assert rank_evidence(runtime, "When?", [])["status"] == "insufficient_evidence"
    assert select_quotes(runtime, "When?", [])["sources"] == []
    assert synthesize(runtime, "When?", [])["status"] == "insufficient_evidence"
    assert runtime.generations == [] and runtime.embeddings == []
    with patch(
        "wikilens.research.retrieve_multilingual", return_value={"articles": []}
    ):
        record = run_pipeline("When?", runtime)
    assert record["status"] == "insufficient_evidence"
    assert len(runtime.generations) == 1 and runtime.closed


def test_full_article_chunking_language_seeding_and_article_cap():
    runtime = Runtime()
    rows = [
        article(lang, i) | {"text": TEXT * 40}
        for i, lang in enumerate(("fi", "en", "sv"), 1)
    ]
    result = rank_evidence(runtime, "When?", rows)
    assert len(result["selected"]) == 6
    assert {p["lang"] for p in result["selected"][:3]} == {"fi", "en", "sv"}
    assert any(p["start_char"] > 1400 for p in result["ranked"])
    for row in rows:
        selected = [p for p in result["selected"] if p["lang"] == row["lang"]]
        assert len(selected) <= 2
        for p in selected:
            assert p["text"] == row["text"][p["start_char"] : p["end_char"]]
    assert runtime.embeddings[0][1] == {"batch_size": 1, "max_tokens": 1024}


def test_candidate_bound_stops_before_embedding_and_synthesis_requires_provenance():
    runtime = Runtime()
    with pytest.raises(ValueError, match="no passages were truncated"):
        rank_evidence(runtime, "When?", [article() | {"text": "x" * 1_651_000}])
    assert runtime.embeddings == []
    with pytest.raises(ValueError, match="provenance-checked"):
        synthesize(runtime, "When?", [{"id": "S1", "quote": TEXT}])


def test_failed_runtime_preserves_hyde_stage_and_always_closes():
    runtime = Runtime()
    with patch.object(
        runtime, "generate", side_effect=RuntimeError("CUDA out of memory")
    ):
        record = run_pipeline("When?", runtime)
    assert record["status"] == "failed"
    assert record["stages"]["hyde"]["raw_output"] is None
    assert "CUDA out of memory" in record["stages"]["hyde"]["error"]
    assert runtime.closed


@pytest.mark.parametrize("question", ["", "  ", None, "x" * 2001])
def test_invalid_questions_rejected_before_model_or_sql(question):
    runtime = Runtime()
    with pytest.raises(ValueError, match="1–2000"):
        generate_hyde(runtime, question)
    with pytest.raises(ValueError, match="1–2000"):
        retrieve(question, "hypothetical", runtime)
    assert runtime.generations == [] and runtime.embeddings == []


def test_invalid_orchestration_question_also_closes_runtime():
    runtime = Runtime()
    with pytest.raises(ValueError):
        run_pipeline("", runtime)
    assert runtime.closed


def test_interactive_stage_rerun_invalidates_downstream_answers():
    from wikilens.pipeline import begin_stage

    record = {
        "question": "When?",
        "status": "draft",
        "stages": {
            name: {"status": "completed"}
            for name in ("hyde", "retrieval", "ranking", "quotations", "synthesis")
        },
    }
    begin_stage(record, "ranking", "When?")
    assert set(record["stages"]) == {"hyde", "retrieval"}
    assert record["status"] == "incomplete"
    with pytest.raises(ValueError, match="preceding stage"):
        begin_stage(record, "synthesis", "When?")
    with pytest.raises(ValueError, match="Question changed"):
        begin_stage(record, "hyde", "A different question?")
