"""Frozen-message and sampling controls; no vLLM installation or GPU needed."""

from types import SimpleNamespace

import pytest

from wikilens.vllm_runtime import VLLMRuntime


class Tokenizer:
    def apply_chat_template(self, messages, **options):
        assert options["enable_thinking"] is False
        return "rendered"

    def __call__(self, text, **options):
        assert options["add_special_tokens"] is False
        return {"input_ids": [1, 2, 3]}


class Session:
    def __init__(self, prompt_tokens=3):
        self.calls = []
        self.prompt_tokens = prompt_tokens

    def post(self, url, **options):
        self.calls.append((url, options))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 2},
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "Actual response"}}
                ],
            },
        )


def test_preserves_messages_and_greedy_settings_with_structured_schema():
    session = Session()
    runtime = VLLMRuntime(_session=session, _tokenizer=Tokenizer())
    messages = [
        {"role": "system", "content": "Suomenkielinen ohje."},
        {"role": "user", "content": "talvisota"},
    ]
    schema = {"type": "object", "properties": {"id": {"enum": ["S1"]}}}
    assert (
        runtime.generate(
            messages, max_new_tokens=100, max_time=30, output_schema=schema
        )
        == "Actual response"
    )
    url, options = session.calls[0]
    payload = options["json"]
    assert url == "http://127.0.0.1:8000/v1/chat/completions"
    assert payload["messages"] == messages
    assert payload["temperature"] == 0 and payload["repetition_penalty"] == 1.05
    assert payload["max_tokens"] == 100 and payload["seed"] == 42
    assert payload["structured_outputs"] == {"json": schema}
    assert options["timeout"] == (10, 30)
    assert runtime.profiles[0]["output_tokens"] == 2


def test_template_mismatch_disables_further_requests_without_retry():
    session = Session(prompt_tokens=4)
    runtime = VLLMRuntime(_session=session, _tokenizer=Tokenizer())
    with pytest.raises(RuntimeError, match="token count differs"):
        runtime.generate([{"role": "user", "content": "talvisota"}])
    with pytest.raises(RuntimeError, match="no automatic retry"):
        runtime.generate([{"role": "user", "content": "talvisota"}])
    assert len(session.calls) == 1


def test_context_overflow_rejected_before_request():
    session = Session()
    runtime = VLLMRuntime(_session=session, _tokenizer=Tokenizer())
    with pytest.raises(ValueError, match="context"):
        runtime.generate(
            [{"role": "user", "content": "talvisota"}], max_new_tokens=12288
        )
    assert not session.calls
