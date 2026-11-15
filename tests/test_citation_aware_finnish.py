"""Citation attachment, bibliography boundaries and FP8 loading without GPU work."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wikilens.citation_aware import parse_citation_aware
from wikilens.citation_pipelines import run_pipeline
from wikilens.finnish_prompts import INSUFFICIENT_EVIDENCE_FI
from wikilens.reranker_policy import pack_generation_nodes
from wikilens.torch_runtime import (
    QWEN4_FP8_MODEL,
    QWEN4_FP8_REVISION,
    _TransformersBackend,
)


def test_real_capture_payload_uses_finnish_prompt_and_one_generation_call():
    saved = (
        Path(__file__).resolve().parents[1]
        / "data/demo-captures/demo-fi-20261114T204842Z/video-games.json"
    )
    if not saved.exists():
        pytest.skip("Ignored real demo capture is not present in this checkout")
    baseline = json.loads(saved.read_text())
    source = baseline["input_capture"]
    settings = {**baseline["settings"], "generation_method": "citation_aware"}
    raw = (
        "Talvisota julkaistiin MSX:lle vuonna 1987 [S1]. "
        "Peli oli tarkoitus julkaista myös Commodore 64:lle [S1].\n\n"
        "Lähteet\n- [S1] Talvisota (videopeli) (fi)."
    )
    calls = []

    class Runtime:
        def generate(self, messages, **options):
            calls.append(messages)
            return raw

        def stats(self):
            return {}

    def forbidden_lookup(*args, **kwargs):
        raise AssertionError("Citation-aware must not perform post-hoc attribution")

    result = run_pipeline(
        {"variants": {"cross_encoder": {"run": source}}},
        "citation_aware",
        settings,
        progress=None,
        runtime=Runtime(),
        reference_lookup=forbidden_lookup,
    )
    assert result["status"] == "draft_unverified"
    assert result["raw_article"] == raw
    assert len(calls) == 1
    assert calls[0][0]["content"].startswith("Olet WikiLens")
    request = json.loads(calls[0][1]["content"])
    assert request["focused_question"] == "talvisota\nPyydetty asiayhteys: videopelit"
    assert request["passages"][0]["text"] == baseline["nodes"][0]["text"]
    assert (
        request["passages"][0]["revision_url"] == baseline["nodes"][0]["revision_url"]
    )
    assert [s["text"] for s in result["sentences"]] == [
        "Talvisota julkaistiin MSX:lle vuonna 1987.",
        "Peli oli tarkoitus julkaista myös Commodore 64:lle.",
    ]
    assert result["source_list_review"]["matches_first_use_order"]
    assert result["mechanical_metrics"]["known_source_ids"] == 2
    assert result["mechanical_metrics"]["quotes_located"] == 0
    assert all(s["citations"][0]["status"] == "id_only" for s in result["sentences"])

    class RecordingTokenizer:
        def apply_chat_template(self, messages, **unused):
            self.messages = messages
            return json.dumps(messages, ensure_ascii=False)

        def __call__(self, rendered, **unused):
            return {"input_ids": list(range(len(rendered)))}

    tokenizer = RecordingTokenizer()
    packed, diagnostics = pack_generation_nodes(
        baseline["nodes"],
        source["question"],
        source["search_focus"],
        {**settings, "context_tokens": 50000},
        tokenizer,
    )
    assert packed == baseline["nodes"]
    assert tokenizer.messages == calls[0]
    assert diagnostics["total_tokens_with_output_budget"] <= 50000


def test_marker_attachment_preserves_both_positions_and_excludes_bibliography():
    units, bibliography = parse_citation_aware(
        "Ensimmäinen väite [S1][S2]. Toinen väite. [S3] Kolmas väite [S999].\n\n"
        "## Lähteet\n- [S1] Otsikko (fi).\n- [S2] Toinen otsikko (fi)."
    )
    assert [[c["source_id"] for c in u["citations"]] for u in units] == [
        ["S1", "S2"],
        ["S3"],
        ["S999"],
    ]
    assert [u["text"] for u in units] == [
        "Ensimmäinen väite.",
        "Toinen väite.",
        "Kolmas väite.",
    ]
    assert bibliography.startswith("- [S1]")
    units, bibliography = parse_citation_aware(INSUFFICIENT_EVIDENCE_FI)
    assert not units[0]["requires_citation"]
    assert bibliography == ""


@pytest.mark.parametrize("capability", [(7, 5), (8, 9)])
def test_fp8_rejects_t4_and_loads_directly_on_supported_cuda(monkeypatch, capability):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    calls = []
    param = SimpleNamespace(dtype="fp8", device=SimpleNamespace(type="cuda"))
    model = SimpleNamespace(
        parameters=lambda: iter([param]),
        config=SimpleNamespace(_commit_hash="resolved"),
    )
    model.eval = lambda: model
    # Deliberately no model.to(): prequantized weights must load on-device.
    factory = SimpleNamespace(
        from_pretrained=lambda model_id, **options: (
            calls.append((model_id, options)) or model
        )
    )
    backend = object.__new__(_TransformersBackend)
    backend.qwen_model = QWEN4_FP8_MODEL
    backend.device = "cuda:0"
    backend.torch = SimpleNamespace(
        float8_e4m3fn="fp8",
        cuda=SimpleNamespace(
            is_available=lambda: True, get_device_capability=lambda device: capability
        ),
    )
    backend.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: object()
    )
    backend.AutoModelForCausalLM = factory
    if capability < (8, 9):
        with pytest.raises(RuntimeError, match="T4 is unsupported"):
            backend.load("qwen", QWEN4_FP8_REVISION)
        assert not calls
    else:
        loaded, _, revision = backend.load("qwen", QWEN4_FP8_REVISION)
        assert loaded is model and revision == "resolved"
        assert calls == [
            (
                QWEN4_FP8_MODEL,
                {
                    "revision": QWEN4_FP8_REVISION,
                    "dtype": "auto",
                    "device_map": {"": "cuda:0"},
                    "use_safetensors": True,
                },
            )
        ]
