"""Bounded pretrained inference with one owned GPU model at a time.

Importing this module does not import torch. Use this runtime exclusively after
restarting a notebook kernel; it cannot release models owned by other variables.
"""

import gc
import time

BGE_MODEL = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
QWEN_MODEL = "Qwen/Qwen3-1.7B"
QWEN_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
QWEN4_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
QWEN4_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
QWEN4_FP8_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
QWEN4_FP8_REVISION = "8591804019c8b22094c3b5b4454e0edc05dffc98"


class TorchRuntime:
    """Own and alternate Qwen/BGE; outputs leave the GPU before returning.

    ``generate`` returns text and ``embed`` returns a list of normalized vectors.
    ``records`` preserves stage, resolved revision, runtime and allocator peaks.
    No retry, training, remote job creation, or background work is performed.
    """

    output_budget_policy = "context"
    max_generation_seconds = 600

    def __init__(
        self,
        *,
        device="cuda",
        qwen_revision=None,
        context_tokens=4096,
        max_batch_tokens=1024,
        qwen_model=QWEN_MODEL,
        _backend=None,
    ):
        if not 256 <= context_tokens <= 131072:
            raise ValueError("context_tokens must be between 256 and 131072")
        if not 1 <= max_batch_tokens <= 8192:
            raise ValueError("max_batch_tokens must be between 1 and 8192")
        self.device = device
        self.qwen_revision = qwen_revision
        self.qwen_model = qwen_model
        self.context_tokens = context_tokens
        self.max_batch_tokens = max_batch_tokens
        self._backend = _backend
        self._model = self._tokenizer = self._stage = None
        self.records = []
        self.revisions = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def active_stage(self):
        return self._stage

    def _load(self, stage):
        if self._stage == stage:
            return
        self.close()
        if self._backend is None:
            self._backend = _TransformersBackend(
                self.device, qwen_model=self.qwen_model
            )
        # Keep the first resolved generator commit across Qwen → BGE → Qwen
        # reloads, including when the original request was a moving branch/tag.
        requested_qwen = self.revisions.get("qwen") or self.qwen_revision
        model, tokenizer, revision = self._backend.load(stage, requested_qwen)
        self._model, self._tokenizer, self._stage = model, tokenizer, stage
        self.revisions[stage] = revision

    def stats(self):
        """JSON-serializable configuration, resolved revisions, and stage records."""
        return {
            "device": self.device,
            "active_stage": self.active_stage,
            "output_budget_policy": self.output_budget_policy,
            "context_tokens": self.context_tokens,
            "requested_qwen_revision": self.qwen_revision,
            "max_batch_tokens": self.max_batch_tokens,
            "models": {"bge": BGE_MODEL, "qwen": self.qwen_model},
            "revisions": dict(self.revisions),
            "records": [dict(record) for record in self.records],
        }

    def close(self):
        """Release owned model refs, then collect unused allocator blocks."""
        self._model = self._tokenizer = self._stage = None
        gc.collect()
        if self._backend is not None:
            self._backend.cleanup()

    def _run(self, stage, operation):
        started = time.perf_counter()
        error_message = None
        result = None
        try:
            self._load(stage)
            self._backend.reset_peak()
            result = operation()
        except Exception as error:
            # Leave the exception scope before cleanup: its traceback may retain
            # activations, even after the model reference has been removed.
            error_message = f"{type(error).__name__}: {error}"
        if error_message is not None:
            self.close()
            self.records.append(
                {
                    "stage": stage,
                    "status": "failed",
                    "error": error_message,
                    "seconds": time.perf_counter() - started,
                }
            )
            raise RuntimeError(
                f"{stage} inference failed; runtime unloaded. {error_message}"
            ) from None
        self.records.append(
            {
                "stage": stage,
                "status": "completed",
                "revision": self.revisions.get(stage),
                "requested_device": self.device,
                "seconds": time.perf_counter() - started,
                **self._backend.memory(),
            }
        )
        return result

    def generate(
        self, messages, *, max_new_tokens=600, max_time=60, output_schema=None
    ):
        """Greedy Qwen generation, thinking disabled; reject oversized input."""
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if max_new_tokens >= self.context_tokens:
            raise ValueError(
                "Requested output must leave room for the input in context_tokens"
            )
        if (
            not isinstance(max_time, (int, float))
            or not 0 < max_time <= self.max_generation_seconds
        ):
            raise ValueError("max_time must be greater than 0 and at most 600 seconds")
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or m.get("role") not in {"system", "user", "assistant"}
                or not isinstance(m.get("content"), str)
                for m in messages
            )
        ):
            raise ValueError("Supply a nonempty list of role/content messages")
        if output_schema is not None and not isinstance(output_schema, dict):
            raise ValueError("output_schema must be a JSON Schema dictionary")
        options = {"output_schema": output_schema} if output_schema is not None else {}
        return self._run(
            "qwen",
            lambda: self._backend.generate(
                self._model,
                self._tokenizer,
                messages,
                max_new_tokens,
                max_time,
                self.context_tokens,
                **options,
            ),
        )

    def embed(self, texts, *, batch_size=2, max_tokens=512):
        """Normalized BGE-M3 CLS vectors; no silent truncation or GPU outputs.

        ``max_tokens`` includes special tokens. Split oversized evidence into
        shorter passages with preserved offsets before calling this function.
        """
        if type(batch_size) is not int or not 1 <= batch_size <= 8:
            raise ValueError("batch_size must be an integer between 1 and 8")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens must be an integer between 1 and 8192")
        if not isinstance(texts, (list, tuple)) or any(
            not isinstance(t, str) or not t.strip() for t in texts
        ):
            raise ValueError("texts must contain nonempty strings")
        if not texts:
            return []
        return self._run(
            "bge",
            lambda: self._backend.embed(
                self._model,
                self._tokenizer,
                list(texts),
                batch_size,
                max_tokens,
                self.max_batch_tokens,
            ),
        )


def upgrade_runtime(runtime):
    """Upgrade an older notebook-owned runtime once, preserving CPU diagnostics.

    Existing model refs are explicitly released before constructing the updated
    CUDA runtime. This avoids requiring retrieval/postprocessing to be repeated.
    """
    import inspect

    parameters = inspect.signature(runtime.generate).parameters
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        return runtime
    if (
        "output_schema" in parameters
        and getattr(runtime, "output_budget_policy", None) == "context"
        and getattr(runtime, "max_generation_seconds", 0) == 600
    ):
        return runtime
    previous = runtime.stats()
    runtime.close()
    updated = TorchRuntime(
        device="cuda",
        qwen_model=getattr(runtime, "qwen_model", QWEN_MODEL),
        qwen_revision=getattr(runtime, "qwen_revision", None),
        context_tokens=previous.get("context_tokens", 4096),
        max_batch_tokens=previous.get("max_batch_tokens", 1024),
    )
    updated.records.extend(previous.get("records", []))
    updated.revisions.update(previous.get("revisions", {}))
    return updated


class _TransformersBackend:
    def __init__(self, device, *, qwen_model=QWEN_MODEL):
        self.qwen_model = qwen_model
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use the existing T4 environment")
        self.torch = torch
        self.AutoModel = AutoModel
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.device = device
        self._enforcer_tokenizer_data = None
        self.model_device = None

    def load(self, stage, qwen_revision):
        is_bge = stage == "bge"
        model_id = BGE_MODEL if is_bge else self.qwen_model
        revision = BGE_REVISION if is_bge else qwen_revision
        options = {"revision": revision} if revision else {}
        is_fp8 = not is_bge and model_id.endswith("-FP8")
        if is_fp8:
            from importlib.util import find_spec

            if not self.device.startswith("cuda") or not self.torch.cuda.is_available():
                raise RuntimeError(
                    "The FP8 generator requires a CUDA GPU; no dequantized fallback is enabled."
                )
            capability = self.torch.cuda.get_device_capability(self.device)
            if capability < (8, 9):
                raise RuntimeError(
                    "The locked Transformers FP8 loader requires compute capability >= 8.9 "
                    "(for example L40S or H100); T4 is unsupported."
                )
            if find_spec("accelerate") is None:
                raise RuntimeError(
                    "FP8 loading requires the fp8 dependency group: uv sync --locked --group fp8"
                )
            # Load pre-quantized weights directly on CUDA. Do not cast FP8 weights
            # via .to(dtype) or silently dequantize to the previous FP16 model.
            options.update(dtype="auto", device_map={"": self.device})
        else:
            options["dtype"] = (
                self.torch.float16
                if self.device.startswith("cuda")
                else self.torch.float32
            )
        tokenizer = self.AutoTokenizer.from_pretrained(
            model_id, **({"revision": revision} if revision else {})
        )
        factory = self.AutoModel if is_bge else self.AutoModelForCausalLM
        model = factory.from_pretrained(model_id, use_safetensors=not is_bge, **options)
        if not is_fp8:
            model = model.to(self.device)
        model = model.eval()
        if is_fp8 and not any(
            p.dtype == self.torch.float8_e4m3fn for p in model.parameters()
        ):
            raise RuntimeError(
                "The requested FP8 checkpoint did not retain FP8 weights; refusing a silent fallback."
            )
        actual_device = next(model.parameters()).device
        if self.device.startswith("cuda") and actual_device.type != "cuda":
            raise RuntimeError(f"Expected GPU weights, found {actual_device}")
        self.model_device = str(actual_device)
        self._enforcer_tokenizer_data = None
        return model, tokenizer, getattr(model.config, "_commit_hash", revision)

    def reset_peak(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self):
        if not self.device.startswith("cuda"):
            return {}
        return {
            "model_device": self.model_device,
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device),
        }

    def cleanup(self):
        self._enforcer_tokenizer_data = None
        if self.device.startswith("cuda"):
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()

    def generate(
        self,
        model,
        tokenizer,
        messages,
        max_new_tokens,
        max_time,
        context_tokens,
        output_schema=None,
    ):
        torch = self.torch
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = output = None
        try:
            # The rendered chat template already supplies its special tokens.
            tokens = tokenizer(
                prompt, return_tensors="pt", truncation=False, add_special_tokens=False
            )
            length = tokens["input_ids"].shape[1]
            if length + max_new_tokens > context_tokens:
                raise ValueError(
                    f"Prompt ({length}) plus output ({max_new_tokens}) exceeds "
                    f"the {context_tokens}-token budget; reduce evidence explicitly"
                )
            tokens = tokens.to(self.device)
            generation_options = {}
            if output_schema is not None:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_token_enforcer_tokenizer_data,
                    build_transformers_prefix_allowed_tokens_fn,
                )

                if self._enforcer_tokenizer_data is None:
                    self._enforcer_tokenizer_data = build_token_enforcer_tokenizer_data(
                        tokenizer
                    )
                generation_options["prefix_allowed_tokens_fn"] = (
                    build_transformers_prefix_allowed_tokens_fn(
                        self._enforcer_tokenizer_data,
                        JsonSchemaParser(output_schema),
                    )
                )
            with torch.inference_mode():
                output = model.generate(
                    **tokens,
                    max_new_tokens=max_new_tokens,
                    max_time=max_time,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                    **generation_options,
                )
            return tokenizer.decode(
                output[0, length:].cpu(), skip_special_tokens=True
            ).strip()
        finally:
            tokens = output = model = tokenizer = None

    def embed(self, model, tokenizer, texts, batch_size, max_tokens, max_batch_tokens):
        torch = self.torch
        results = []
        tokens = hidden = vectors = None
        try:
            for offset in range(0, len(texts), batch_size):
                batch = texts[offset : offset + batch_size]
                tokens = tokenizer(
                    batch, padding=True, truncation=False, return_tensors="pt"
                )
                width = tokens["input_ids"].shape[1]
                if width > max_tokens:
                    raise ValueError(
                        f"Passage has {width} tokens, exceeding max_tokens={max_tokens}; "
                        "split it explicitly without losing source offsets"
                    )
                if width * len(batch) > max_batch_tokens:
                    raise ValueError(
                        "Padded batch exceeds max_batch_tokens; reduce batch_size"
                    )
                tokens = tokens.to(self.device)
                with torch.inference_mode():
                    hidden = model(**tokens).last_hidden_state
                    vectors = torch.nn.functional.normalize(
                        hidden[:, 0].float(), p=2, dim=1
                    )
                    if vectors.shape[1] != 1024 or not torch.isfinite(vectors).all():
                        raise ValueError(
                            "Expected finite 1024-dimensional BGE-M3 vectors"
                        )
                    results.extend(vectors.cpu().tolist())
                # Release each batch's activations before allocating the next.
                tokens = hidden = vectors = None
            return results
        finally:
            tokens = hidden = vectors = model = tokenizer = None
