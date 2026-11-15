"""CPU-only lifecycle and memory-budget checks; no model fitting or downloads."""

import json
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from wikilens.torch_runtime import BGE_REVISION, TorchRuntime, _TransformersBackend


class Model:
    pass


class FakeBackend:
    def __init__(self):
        self.events = []
        self.models = []
        self.requested_revisions = []
        self.fail = False

    def load(self, stage, revision):
        assert all(model() is None for model in self.models), (
            "previous model is still alive"
        )
        self.events.append(("load", stage))
        self.requested_revisions.append((stage, revision))
        model = Model()
        self.models.append(weakref.ref(model))
        return model, object(), BGE_REVISION if stage == "bge" else "qwen-commit"

    def generate(self, *args):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return "answer"

    def embed(self, model, tokenizer, texts, *args):
        return [[1.0] + [0.0] * 1023 for _ in texts]

    def reset_peak(self):
        pass

    def cleanup(self):
        self.events.append(("cleanup", None))

    def memory(self):
        return {"peak_allocated_bytes": 100}


MESSAGES = [{"role": "user", "content": "What happened?"}]


def test_stage_switch_releases_previous_model_before_next_load():
    backend = FakeBackend()
    with TorchRuntime(_backend=backend) as runtime:
        assert runtime.generate(MESSAGES) == "answer"
        assert runtime.generate(MESSAGES) == "answer"
        vectors = runtime.embed(["question", "source"])
        assert len(vectors) == 2 and len(vectors[0]) == 1024
        assert runtime.generate(MESSAGES) == "answer"
        assert [e for e in backend.events if e[0] == "load"] == [
            ("load", "qwen"),
            ("load", "bge"),
            ("load", "qwen"),
        ]
        assert runtime.stats()["revisions"] == {
            "qwen": "qwen-commit",
            "bge": BGE_REVISION,
        }
        assert backend.requested_revisions[0] == ("qwen", None)
        assert backend.requested_revisions[-1] == ("qwen", "qwen-commit")
        assert len(runtime.stats()["records"]) == 4
        json.dumps(runtime.stats())
    assert runtime.active_stage is None
    assert all(model() is None for model in backend.models)


def test_oom_unloads_state_no_auto_retry_and_next_call_can_recover():
    backend = FakeBackend()
    backend.fail = True
    runtime = TorchRuntime(_backend=backend)
    with pytest.raises(RuntimeError, match="runtime unloaded.*CUDA out of memory"):
        runtime.generate(MESSAGES)
    assert runtime.active_stage is None
    assert all(model() is None for model in backend.models)
    assert len([e for e in backend.events if e[0] == "load"]) == 1
    assert runtime.records[-1]["status"] == "failed"
    backend.fail = False
    assert runtime.generate(MESSAGES) == "answer"
    runtime.close()


def test_empty_embedding_and_invalid_limits_do_not_load_model():
    backend = FakeBackend()
    runtime = TorchRuntime(_backend=backend)
    assert runtime.embed([]) == []
    for kwargs in ({"batch_size": 0}, {"max_tokens": 0}, {"batch_size": 9}):
        with pytest.raises(ValueError):
            runtime.embed(["evidence"], **kwargs)
    for kwargs in (
        {"max_new_tokens": 4096},
        {"max_time": 601},
        {"max_time": float("nan")},
    ):
        with pytest.raises(ValueError):
            runtime.generate(MESSAGES, **kwargs)
    with pytest.raises(ValueError):
        runtime.generate([{"role": "user", "content": 5}])
    assert backend.events == []


class BudgetTokenizer:
    def __init__(self, width):
        self.width = width
        self.chat_options = None
        self.token_options = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_options = kwargs
        return "rendered prompt"

    def __call__(self, *args, **kwargs):
        self.token_options = kwargs
        return {"input_ids": SimpleNamespace(shape=(1, self.width))}


def test_generation_budget_includes_requested_output_and_disables_thinking():
    backend = object.__new__(_TransformersBackend)
    backend.torch = object()
    tokenizer = BudgetTokenizer(200)
    with pytest.raises(ValueError, match="plus output.*exceeds"):
        backend.generate(object(), tokenizer, MESSAGES, 100, 60, 256)
    assert tokenizer.chat_options["enable_thinking"] is False
    assert tokenizer.token_options["truncation"] is False
    assert tokenizer.token_options["add_special_tokens"] is False


@pytest.mark.parametrize(
    "width,max_tokens,max_batch_tokens,match",
    [
        (513, 512, 2048, "split it explicitly"),
        (600, 1024, 1024, "reduce batch_size"),
    ],
)
def test_embedding_rejects_oversize_evidence_or_padded_batch_before_gpu_transfer(
    width, max_tokens, max_batch_tokens, match
):
    backend = object.__new__(_TransformersBackend)
    backend.torch = object()
    tokenizer = BudgetTokenizer(width)
    with pytest.raises(ValueError, match=match):
        backend.embed(
            object(), tokenizer, ["one", "two"], 2, max_tokens, max_batch_tokens
        )
    assert tokenizer.token_options == {
        "padding": True,
        "truncation": False,
        "return_tensors": "pt",
    }


def test_module_import_needs_no_torch_or_transformers():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; from wikilens.torch_runtime import TorchRuntime; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_output_above_700_allowed_within_context_and_overflow_rejected_before_load():
    backend = FakeBackend()
    runtime = TorchRuntime(context_tokens=4096, _backend=backend)
    assert runtime.generate(MESSAGES, max_new_tokens=1500) == "answer"
    runtime.close()
    backend.events.clear()
    for length in (4096, 4097, 0, -1, True, 1.5):
        with pytest.raises(ValueError):
            runtime.generate(MESSAGES, max_new_tokens=length)
    assert backend.events == []


def test_upgrade_replaces_old_schema_runtime_with_context_budget_runtime():
    from wikilens.torch_runtime import upgrade_runtime

    class OldRuntime:
        qwen_revision = "requested-commit"
        closed = False

        def generate(self, messages, *, max_new_tokens=600, output_schema=None):
            raise AssertionError("Old generation must not be called")

        def stats(self):
            return {
                "context_tokens": 6000,
                "max_batch_tokens": 512,
                "revisions": {"qwen": "resolved-commit"},
                "records": [{"stage": "qwen", "status": "completed"}],
            }

        def close(self):
            self.closed = True

    old = OldRuntime()
    updated = upgrade_runtime(old)
    assert old.closed
    assert updated.output_budget_policy == "context"
    assert updated.context_tokens == 6000 and updated.device == "cuda"
    assert updated.qwen_revision == "requested-commit"
    assert updated.revisions == {"qwen": "resolved-commit"}
    assert updated.records == old.stats()["records"]
    assert upgrade_runtime(updated) is updated
