"""Review ordering and failure isolation; no fitting, models or downloads."""

import json

from wikilens.citation_pipelines import run_pipeline
from wikilens.reranker_policy import pack_generation_nodes
from wikilens.text_review import REVIEW_PROMPT_FI

NODE = {
    "id": "S1",
    "title": "Talvisota (videopeli)",
    "lang": "fi",
    "text": "Peli julkaistiin vuonna 1987.",
    "start_char": 0,
    "revision_url": "https://fi.wikipedia.org/w/index.php?oldid=1",
}
SETTINGS = {
    "context_tokens": 12288,
    "article_tokens": 4000,
    "article_seconds": 300,
    "review_tokens": 4000,
    "review_seconds": 300,
    "review_before_citations": True,
    "posthoc_prompt": "finnish_no_length_target",
    "generation_method": "post_hoc",
    "target_words": None,
    "citation_unit": "paragraph",
    "reference_top_k": 3,
    "reference_pair_tokens": 1024,
    "attribution_batch_size": 2,
    "attribution_mode": "source_spans",
    "attribution_tokens": 1200,
    "attribution_seconds": 90,
}


def capture():
    return {
        "variants": {
            "cross_encoder": {
                "run": {
                    "question": "talvisota",
                    "stages": {
                        "postprocessing": {"nodes": [NODE]},
                        "generation": {
                            "messages": [
                                {"role": "system", "content": "Prepared input"},
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        {
                                            "question": "talvisota",
                                            "search_focus": "videopeli",
                                            "passages": [
                                                {
                                                    "source_id": "S1",
                                                    "title": NODE["title"],
                                                    "lang": "fi",
                                                    "text": NODE["text"],
                                                }
                                            ],
                                        }
                                    ),
                                },
                            ]
                        },
                    },
                }
            }
        }
    }


class Runtime:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def stats(self):
        return {}

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return "Peli julkaistiin vuonna 2025."
        if len(self.calls) == 2:
            assert messages[0]["content"] == REVIEW_PROMPT_FI
            assert messages[2]["content"] == "Peli julkaistiin vuonna 2025."
            assert NODE["text"] in messages[1]["content"]
            if self.fail:
                raise RuntimeError("Review unavailable")
            return "Peli julkaistiin vuonna 1987."
        return '{"items":[{"claim_id":"P1","span_ids":["S1_1"]}]}'


def test_citations_apply_only_to_revised_draft_and_both_outputs_survive():
    runtime = Runtime()

    def lookup(runtime, question, units, nodes, **kwargs):
        assert units[0]["text"] == "Peli julkaistiin vuonna 1987."
        units[0]["reference_candidates"] = [{"source_id": "S1", "score": 1}]
        return {"seconds": 0}

    result = run_pipeline(
        capture(),
        "post_hoc",
        SETTINGS,
        runtime=runtime,
        reference_lookup=lookup,
        progress=None,
    )
    assert result["status"] == "draft_unverified"
    assert result["initial_draft"] == "Peli julkaistiin vuonna 2025."
    assert result["reviewed_draft"] == "Peli julkaistiin vuonna 1987."
    assert result["sentences"][0]["citations"][0]["quote"] == NODE["text"]
    assert result["review_status"] == "self_reviewed_unverified"
    assert not result["claim_support_reviewed"]
    assert len(runtime.calls) == 3


def test_review_failure_does_not_present_original_as_reviewed():
    runtime = Runtime(fail=True)

    def lookup(*args, **kwargs):
        raise AssertionError("Failed review must not proceed to citation lookup")

    result = run_pipeline(
        capture(),
        "post_hoc",
        SETTINGS,
        runtime=runtime,
        reference_lookup=lookup,
        progress=None,
    )
    assert result["status"] == "failed" and result["review_status"] == "failed"
    assert result["initial_draft"] == "Peli julkaistiin vuonna 2025."
    assert not result["sentences"] and "reviewed_draft" not in result


def test_context_packing_reserves_draft_and_review_without_truncation():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages, ensure_ascii=False)

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text)))}

    nodes = [NODE, {**NODE, "id": "S2", "text": "X" * 3500}]
    plain, _ = pack_generation_nodes(
        nodes,
        "talvisota",
        "videopeli",
        {**SETTINGS, "review_before_citations": False},
        Tokenizer(),
    )
    reviewed, diagnostics = pack_generation_nodes(
        nodes, "talvisota", "videopeli", SETTINGS, Tokenizer()
    )
    assert len(reviewed) < len(plain)
    assert reviewed[0]["text"] == NODE["text"]
    assert (
        diagnostics["review_context_tokens_with_reserved_draft_and_output"]
        <= SETTINGS["context_tokens"]
    )
    assert diagnostics["counts"]["rejected_review_context_budget"] > 0
