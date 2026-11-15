"""Compatibility and retrieval-input checks, with no model download or fitting."""

from types import SimpleNamespace

from wikilens import hyde_search
from wikilens.torch_runtime import (
    QWEN4_MODEL,
    QWEN_MODEL,
    TorchRuntime,
    upgrade_runtime,
)


def test_notebook_default_and_explicit_generator_metadata():
    assert TorchRuntime().stats()["models"]["qwen"] == QWEN_MODEL
    runtime = TorchRuntime(
        qwen_model=QWEN4_MODEL, qwen_revision="four-commit", context_tokens=12288
    )
    assert runtime.stats()["models"]["qwen"] == QWEN4_MODEL
    assert runtime.stats()["requested_qwen_revision"] == "four-commit"


def test_notebook_upgrade_preserves_selected_generator():
    old = SimpleNamespace(
        qwen_model=QWEN4_MODEL,
        qwen_revision="four-commit",
        generate=lambda messages: None,
        close=lambda: None,
        stats=lambda: {"context_tokens": 12288, "revisions": {"qwen": "four-commit"}},
    )
    upgraded = upgrade_runtime(old)
    assert upgraded.qwen_model == QWEN4_MODEL
    assert upgraded.qwen_revision == "four-commit"


def test_direct_query_embeds_original_question_and_context_without_generation(
    monkeypatch,
):
    seen = []

    def retrieve(runtime, question, record, **options):
        seen.append(record)
        return {
            "articles": [{"origin": "hyde_vector_candidate"}],
            "embedding_input": record["text"],
        }

    monkeypatch.setattr(hyde_search, "retrieve_hyde", retrieve)
    result = hyde_search.retrieve_direct(
        object(), "talvisota", "videopelit", language="fi"
    )
    assert result["embedding_input"] == "talvisota\nSearch focus: videopelit"
    assert result["method"] == "direct_query"
    assert result["articles"][0]["origin"] == "direct_vector_candidate"
    assert seen[0]["question"] == "talvisota"
    assert (
        hyde_search.retrieve_direct(object(), "talvisota")["embedding_input"]
        == "talvisota"
    )
