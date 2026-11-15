"""Small CPU-only policy/context checks; no models, network or CUDA."""

import copy
import json
import pytest

from wikilens.reranker_policy import filter_ranked, pack_generation_nodes


class CharacterTokenizer:
    """A deliberately simple exact tokenizer for verifying packing invariants."""

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return json.dumps(messages, ensure_ascii=False) + "<assistant>"

    def __call__(self, rendered, *, truncation, add_special_tokens):
        assert truncation is False
        assert add_special_tokens is False
        return {"input_ids": list(range(len(rendered)))}


def node(name, text="passage", score=0):
    return {
        "id": name,
        "title": "Article",
        "lang": "fi",
        "text": text,
        "start_char": 10,
        "end_char": 10 + len(text),
        "rerank_score": score,
        "metadata": {"original": True},
    }


def settings(context=8192, **extra):
    return {
        "context_tokens": context,
        "article_tokens": 1000,
        "target_words": 500,
        "answer_language": "English",
        "citation_unit": "paragraph",
        **extra,
    }


def test_filter_is_inclusive_order_stable_and_has_no_implicit_cutoff():
    rows = [
        node("a", score=-3),
        node("b", score=-1),
        node("c", score=-2),
        node("d", score=-9),
    ]
    original = copy.deepcopy(rows)
    kept, diag = filter_ranked(rows)
    assert [n["id"] for n in kept] == ["a", "b", "c", "d"]
    assert diag["effective_min_score"] is None
    kept, diag = filter_ranked(rows, min_score=-3, relative_gap=1)
    assert [n["id"] for n in kept] == [
        "b",
        "c",
    ]  # b is the max even if input is not sorted.
    assert diag["counts"]["rejected_absolute"] == 1
    assert diag["counts"]["rejected_relative"] == 2
    assert diag["counts"]["rejected_both"] == 1
    assert diag["counts"]["rejected"] == 2
    kept[0]["metadata"]["original"] = False
    assert rows == original
    assert filter_ranked([], relative_gap=1)[1]["best_score"] is None


@pytest.mark.parametrize(
    "kwargs", [{"min_score": float("nan")}, {"relative_gap": -1}, {"min_score": True}]
)
def test_filter_rejects_invalid_policies(kwargs):
    with pytest.raises(ValueError):
        filter_ranked([node("a")], **kwargs)


@pytest.mark.parametrize("context", [8192, 12288])
def test_packer_uses_real_prompt_budget_and_retains_whole_later_nodes(context):
    rows = [
        node("too-big", "X" * context),
        node("fits", "Finnish source, untouched. " * 5),
        node("last", "More exact source. " * 5),
    ]
    original = copy.deepcopy(rows)
    kept, diag = pack_generation_nodes(
        rows, "question", "history", settings(context), CharacterTokenizer()
    )
    assert [n["id"] for n in kept] == ["fits", "last"]
    assert diag["counts"]["rejected_context_budget"] == 1
    assert diag["total_tokens_with_output_budget"] <= context
    assert diag["input_tokens"] > sum(
        len(n["text"]) for n in kept
    )  # System/template/metadata counted.
    assert kept[0]["text"] == original[1]["text"]
    kept[0]["metadata"]["original"] = False
    assert rows == original


def test_packer_distinguishes_max_nodes_and_impossible_base_prompt():
    kept, diag = pack_generation_nodes(
        [node("a"), node("b")],
        "question",
        "history",
        settings(max_nodes=1),
        CharacterTokenizer(),
    )
    assert [n["id"] for n in kept] == ["a"]
    assert diag["rejected"] == [{"id": "b", "input_index": 1, "reason": "max_nodes"}]
    with pytest.raises(ValueError, match="even with no source passages"):
        pack_generation_nodes(
            [],
            "question",
            "history",
            settings(context=8192, article_tokens=8000),
            CharacterTokenizer(),
        )
