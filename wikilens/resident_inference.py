"""Resident FP16 inference with explicit load, prefill and decode measurements.

Uses the existing Transformers/LM Format Enforcer stack, no training. A first
forward synchronization is added for timing; decode measurements include the
remaining generation bookkeeping and constrained-decoding CPU work.
"""

import gc
import time
from .torch_runtime import _TransformersBackend, QWEN_MODEL
from .reranker_comparison import CROSS_MODEL, CROSS_REVISION


class ResidentModels:
    def __init__(self, revision, context_tokens=8192, *, qwen_model=QWEN_MODEL):
        self.qwen_model = qwen_model
        import_started = time.perf_counter()
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        import_seconds = time.perf_counter() - import_started
        self.torch = torch
        self.context_tokens = context_tokens
        self.profiles = []
        self.lookup_profiles = []
        self.loads = {"library_import_seconds": import_seconds}
        self.backend = _TransformersBackend("cuda", qwen_model=qwen_model)
        t = time.perf_counter()
        self.model, self.tokenizer, self.revision = self.backend.load("qwen", revision)
        torch.cuda.synchronize()
        self.loads["qwen_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
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
        torch.cuda.synchronize()
        self.loads["reranker_seconds"] = time.perf_counter() - t
        self.loads["resident_allocated_bytes"] = torch.cuda.memory_allocated()
        self._enforcer_data = None

    def close(self):
        # The shared owner calls shutdown after all measured examples.
        pass

    def shutdown(self):
        self.model = self.tokenizer = self.cross = self.cross_tokenizer = (
            self._enforcer_data
        ) = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def stats(self):
        return {
            "device": "cuda",
            "gpu": self.torch.cuda.get_device_name(0),
            "resident": True,
            "models": {"qwen": self.qwen_model},
            "revisions": {"qwen": self.revision, "cross_encoder": CROSS_REVISION},
            "loads": dict(self.loads),
            "records": list(self.profiles),
        }

    def generate(
        self, messages, *, max_new_tokens=600, max_time=60, output_schema=None
    ):
        torch = self.torch
        wall = time.perf_counter()
        profile = {"schema_constrained": output_schema is not None}
        t = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=False, add_special_tokens=False
        )
        length = tokens["input_ids"].shape[1]
        if length + max_new_tokens > self.context_tokens:
            raise ValueError("Prompt plus output exceeds configured context")
        profile.update(
            input_tokens=length,
            requested_output_tokens=max_new_tokens,
            tokenization_seconds=time.perf_counter() - t,
        )
        t = time.perf_counter()
        tokens = tokens.to("cuda")
        torch.cuda.synchronize()
        profile["input_transfer_seconds"] = time.perf_counter() - t
        options = {}
        t = time.perf_counter()
        if output_schema is not None:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_token_enforcer_tokenizer_data,
                build_transformers_prefix_allowed_tokens_fn,
            )

            if self._enforcer_data is None:
                self._enforcer_data = build_token_enforcer_tokenizer_data(
                    self.tokenizer
                )
            options["prefix_allowed_tokens_fn"] = (
                build_transformers_prefix_allowed_tokens_fn(
                    self._enforcer_data, JsonSchemaParser(output_schema)
                )
            )
        profile["schema_setup_seconds"] = time.perf_counter() - t
        first = {}
        events = {}
        started = time.perf_counter()

        def pre(module, args):
            if "start" not in events:
                events["start"] = torch.cuda.Event(enable_timing=True)
                events["end"] = torch.cuda.Event(enable_timing=True)
                events["start"].record()

        def post(module, args, result):
            if not first:
                events["end"].record()
                torch.cuda.synchronize()
                first["wall"] = time.perf_counter()
                first["gpu_seconds"] = (
                    events["start"].elapsed_time(events["end"]) / 1000
                )

        h1 = self.model.register_forward_pre_hook(pre)
        h2 = self.model.register_forward_hook(post)
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                output = self.model.generate(
                    **tokens,
                    max_new_tokens=max_new_tokens,
                    max_time=max_time,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **options,
                )
            torch.cuda.synchronize()
            end = time.perf_counter()
            generated = output.shape[1] - length
            profile.update(
                generation_seconds=end - started,
                prefill_first_forward_seconds=first["wall"] - started,
                prefill_forward_gpu_seconds=first["gpu_seconds"],
                decode_and_generation_overhead_seconds=end - first["wall"],
                output_tokens=generated,
                decode_tokens_per_second=(generated - 1)
                / max(1e-9, end - first["wall"]),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                finish="eos"
                if output[0, -1].item() == self.tokenizer.eos_token_id
                else "budget_or_time",
            )
            t = time.perf_counter()
            text = self.tokenizer.decode(
                output[0, length:].cpu(), skip_special_tokens=True
            ).strip()
            profile["output_decode_seconds"] = time.perf_counter() - t
            return text
        finally:
            h1.remove()
            h2.remove()
            profile["wall_seconds"] = time.perf_counter() - wall
            self.profiles.append(profile)

    def lookup(
        self,
        runtime,
        question,
        sentences,
        nodes,
        *,
        top_k=3,
        batch_size=8,
        max_pair_tokens=1024,
        **unused,
    ):
        torch = self.torch
        start = time.perf_counter()
        queries = [s for s in sentences if s["requires_citation"]]
        pairs = [(s, n) for s in queries for n in nodes]
        scores = []
        batches = []
        for i in range(0, len(pairs), batch_size):
            t = time.perf_counter()
            batch = pairs[i : i + batch_size]
            inputs = self.cross_tokenizer(
                [
                    [
                        f"Topic: {question}\nClaim: {s['text']}",
                        n["title"] + "\n" + n["text"],
                    ]
                    for s, n in batch
                ],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if inputs["input_ids"].shape[1] > max_pair_tokens:
                raise ValueError("Reference pair exceeds explicit token budget")
            tokenization = time.perf_counter() - t
            t = time.perf_counter()
            inputs = inputs.to("cuda")
            torch.cuda.synchronize()
            transfer = time.perf_counter() - t
            t = time.perf_counter()
            with torch.inference_mode():
                out = self.cross(**inputs)
            torch.cuda.synchronize()
            forward = time.perf_counter() - t
            scores.extend(out.logits.view(-1).float().cpu().tolist())
            batches.append(
                {
                    "pairs": len(batch),
                    "padded_tokens": inputs["input_ids"].shape[1],
                    "tokenization_seconds": tokenization,
                    "transfer_seconds": transfer,
                    "forward_seconds": forward,
                }
            )
        matches = {s["claim_id"]: [] for s in queries}
        for (s, n), score in zip(pairs, scores):
            matches[s["claim_id"]].append({"source_id": n["id"], "score": float(score)})
        for s in queries:
            matches[s["claim_id"]].sort(key=lambda x: -x["score"])
            s["reference_candidates"] = matches[s["claim_id"]][:top_k]
        result = {
            "seconds": time.perf_counter() - start,
            "ranked": matches,
            "diagnostics": {
                "model": CROSS_MODEL,
                "revision": CROSS_REVISION,
                "resident": True,
                "device": "cuda:0",
                "batches": batches,
            },
        }
        self.lookup_profiles.append(result)
        return result
