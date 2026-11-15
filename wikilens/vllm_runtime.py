"""Private loopback vLLM adapter for a frozen-evidence inference comparison."""

import hashlib
import time
from importlib.metadata import version

from .reranker_comparison import CROSS_MODEL, CROSS_REVISION
from .torch_runtime import QWEN4_FP8_MODEL, QWEN4_FP8_REVISION


class VLLMRuntime:
    def __init__(
        self, base_url="http://127.0.0.1:8000", *, _session=None, _tokenizer=None
    ):
        if _session is None:
            import requests

            _session = requests.Session()
        if _tokenizer is None:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(
                QWEN4_FP8_MODEL, revision=QWEN4_FP8_REVISION, local_files_only=True
            )
        self.session, self.tokenizer = _session, _tokenizer
        self.base_url = base_url
        self.profiles, self.lookup_profiles = [], []
        self.cross = self.cross_tokenizer = None
        self.dead = False

    def generate(
        self, messages, *, max_new_tokens=600, max_time=60, output_schema=None
    ):
        if self.dead:
            raise RuntimeError("Previous vLLM request failed; no automatic retry")
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_tokens = len(
            self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        )
        if input_tokens + max_new_tokens > 12288:
            raise ValueError(
                "Prompt plus requested output exceeds frozen 12288-token context"
            )
        payload = {
            "model": QWEN4_FP8_MODEL,
            "messages": messages,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.05,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": max_new_tokens,
            "seed": 42,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if output_schema is not None:
            payload["structured_outputs"] = {"json": output_schema}
        profile = {
            "input_tokens": input_tokens,
            "requested_output_tokens": max_new_tokens,
            "schema_constrained": output_schema is not None,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
        started = time.perf_counter()
        try:
            response = self.session.post(
                self.base_url + "/v1/chat/completions",
                json=payload,
                timeout=(10, max_time),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"vLLM HTTP {response.status_code}: {response.text[:1200]}"
                )
            value = response.json()
            profile["response"] = value
            usage = value["usage"]
            if usage["prompt_tokens"] != input_tokens:
                raise RuntimeError(
                    "vLLM token count differs from the pinned rendered chat template"
                )
            choice = value["choices"][0]
            result = choice["message"].get("content") or ""
            profile.update(
                output_tokens=usage["completion_tokens"],
                finish=choice["finish_reason"],
                status="returned",
            )
            return result.strip()
        except Exception as error:
            self.dead = True
            profile.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            profile["generation_seconds"] = time.perf_counter() - started
            profile["effective_output_tokens_per_second"] = profile.get(
                "output_tokens", 0
            ) / max(profile["generation_seconds"], 1e-9)
            self.profiles.append(profile)

    def load_references(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.cross_tokenizer = AutoTokenizer.from_pretrained(
            CROSS_MODEL, revision=CROSS_REVISION, local_files_only=True
        )
        self.cross = (
            AutoModelForSequenceClassification.from_pretrained(
                CROSS_MODEL,
                revision=CROSS_REVISION,
                local_files_only=True,
                use_safetensors=True,
                dtype=torch.float16,
            )
            .to("cuda")
            .eval()
        )

    def lookup(self, *args, **kwargs):
        from .resident_inference import ResidentModels

        return ResidentModels.lookup(self, *args, **kwargs)

    def stats(self):
        return {
            "device": "cuda",
            "backend": "vllm",
            "gpu": self.torch.cuda.get_device_name(0),
            "resident": True,
            "models": {"qwen": QWEN4_FP8_MODEL},
            "revisions": {"qwen": QWEN4_FP8_REVISION, "cross_encoder": CROSS_REVISION},
            "versions": {p: version(p) for p in ["vllm", "torch", "transformers"]},
            "records": list(self.profiles),
            "timing_note": "Loopback request wall time includes tokenization, prefill, decode and response serialization; not decode-only throughput.",
        }

    def close(self):
        pass

    def shutdown(self):
        self.cross = self.cross_tokenizer = None
        self.session.close()
