"""Measure long-prefill memory with the real three-model demo engine.

Run only after full-L import/other inference have stopped. No training, cloud
launch, runtime patching, or automatic retry. A single smaller 8192-budget probe
is attempted only after an actual CUDA OOM in the primary12288-budget probe.
Synthetic input tests capacity, never answer quality or factual support.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def proc_memory():
    import psutil

    process = psutil.Process()
    info = process.memory_info()
    result = {"rss_bytes": info.rss, "vms_bytes": info.vms}
    for line in Path("/proc/self/status").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in ("RssAnon", "RssFile", "RssShmem", "VmPeak", "VmHWM"):
            result[key + "_bytes"] = int(line.split()[1]) * 1024
    cgroup = Path("/sys/fs/cgroup")
    for filename in ("memory.current", "memory.peak", "memory.max"):
        p = cgroup / filename
        if p.exists():
            value = p.read_text().strip()
            result["cgroup_" + filename.replace(".", "_") + "_bytes"] = (
                int(value) if value.isdigit() else value
            )
    events = cgroup / "memory.events"
    if events.exists():
        result["cgroup_memory_events"] = {
            line.split()[0]: int(line.split()[1])
            for line in events.read_text().splitlines()
        }
    result["mariadb_rss_bytes"] = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            if p.info["name"] in ("mariadbd", "mysqld"):
                result["mariadb_rss_bytes"] += p.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return result


class MemorySampler:
    """Sample process anonymous/RSS and whole-cgroup memory every50ms."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.maximum = {}
        self.samples = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                current = proc_memory()
                for key in (
                    "rss_bytes",
                    "RssAnon_bytes",
                    "cgroup_memory_current_bytes",
                    "mariadb_rss_bytes",
                ):
                    value = current.get(key)
                    if isinstance(value, int):
                        self.maximum[key] = max(value, self.maximum.get(key, 0))
                self.samples += 1
            except (OSError, ValueError):
                pass
            self.stop_event.wait(0.05)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        return {
            "interval_seconds": 0.05,
            "sample_count": self.samples,
            "sampled_peak_bytes": self.maximum.copy(),
            "note": "Sampled peaks may miss short spikes; cgroup memory.peak is lifetime-wide, not probe-specific.",
        }


def make_input(tokenizer, target):
    """Build an actual chat-template input just below the requested token count."""
    system = "This is a synthetic context-length smoke test, not a Wikipedia answer task. Continue the neutral text. Do not treat the input as factual evidence."
    prefix = (
        "Repeated neutral material follows solely to exercise the context budget.\n"
    )
    unit = " Context budget probe: source passages remain unchanged while memory and token counts are measured. "

    def render(repetitions, padding=0):
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": prefix + unit * repetitions + " padding" * padding,
            },
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = tokenizer(text, truncation=False, add_special_tokens=False)
        return messages, text, len(encoded["input_ids"])

    unit_tokens = len(tokenizer(unit, add_special_tokens=False)["input_ids"])
    high = max(1, target // max(1, unit_tokens) + 2)
    while render(high)[2] <= target:
        high *= 2
    low = 0
    while low + 1 < high:
        mid = (low + high) // 2
        if render(mid)[2] <= target:
            low = mid
        else:
            high = mid
    repetitions = low
    remaining = max(0, target - render(repetitions)[2])
    low, high = 0, remaining * 2 + 8
    while low + 1 < high:
        mid = (low + high) // 2
        if render(repetitions, mid)[2] <= target:
            low = mid
        else:
            high = mid
    messages, text, count = render(repetitions, low)
    if not target - 32 <= count <= target:
        raise ValueError(
            f"Synthetic input size was {count}, expected within32 tokens below {target}"
        )
    return (
        messages,
        text,
        {
            "target_input_tokens": target,
            "input_tokens": count,
            "unit_text": unit,
            "repeat_count": repetitions,
            "tail_padding_words": low,
            "system": system,
            "user_prefix": prefix,
            "rendered_sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    )


def probe(engine, context_limit, target, output_tokens):
    torch = engine.models.torch
    messages, rendered, metadata = make_input(engine.models.tokenizer, target)
    encoded = engine.models.tokenizer(
        rendered, return_tensors="pt", truncation=False, add_special_tokens=False
    )
    input_tokens = int(encoded["input_ids"].shape[1])
    if input_tokens + output_tokens > context_limit:
        raise ValueError("Actual prompt plus output exceeds the probe context limit")
    if context_limit > engine.models.model.config.max_position_embeddings:
        raise ValueError("Probe context exceeds actual model capacity")
    before = proc_memory()
    sampler = MemorySampler()
    sampler.start()
    result = {
        "status": "started",
        "probe_context_limit": context_limit,
        "output_tokens_requested": output_tokens,
        "input": metadata,
        "total_token_budget": input_tokens + output_tokens,
        "cpu_before": before,
        "gpu_before": {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        },
        "synthetic_probe_not_quality_evaluation": True,
    }
    inputs = outputs = None
    failure = None
    tick = time.perf_counter()
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        inputs = encoded.to("cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        generation_start = time.perf_counter()
        with torch.inference_mode():
            outputs = engine.models.model.generate(
                **inputs,
                max_new_tokens=output_tokens,
                min_new_tokens=output_tokens,
                max_time=180,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=engine.models.tokenizer.eos_token_id,
            )
        end.record()
        torch.cuda.synchronize()
        generated = outputs[0, input_tokens:].cpu().tolist()
        result.update(
            status="passed"
            if len(generated) == output_tokens
            else "incomplete_generation",
            generation_wall_seconds=time.perf_counter() - generation_start,
            generation_cuda_elapsed_seconds=start.elapsed_time(end) / 1000,
            actual_output_tokens=len(generated),
            output_token_ids=generated,
            output_text=engine.models.tokenizer.decode(
                generated, skip_special_tokens=True
            ),
        )
    except Exception as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
            isinstance(error, RuntimeError)
            and "cuda out of memory" in str(error).lower()
        )
        failure = {
            "status": "cuda_oom" if is_oom else "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        result.update(
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            probe_wall_seconds=time.perf_counter() - tick,
            cpu_sampling=sampler.finish(),
            cpu_after=proc_memory(),
        )
        inputs = outputs = encoded = None
        gc.collect()
        torch.cuda.empty_cache()
    if failure:
        result.update(failure)
    result["gpu_after_cleanup"] = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-input-tokens", type=int, default=11700)
    parser.add_argument("--output-tokens", type=int, default=128)
    args = parser.parse_args()
    if (
        not 1 <= args.output_tokens < 12288
        or not 1 <= args.target_input_tokens <= 12288 - args.output_tokens
    ):
        raise ValueError("Primary input/output request must fit the12288-token context")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output or ROOT / f"reports/demo-context-smoke-{stamp}.json"
    report = {
        "created_at": stamp,
        "status": "initializing",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_context_limit": 12288,
        "probes": [],
        "cpu_before_model_load": proc_memory(),
        "note": "Capacity smoke only. Synthetic repeated input; min/max output both128 by default to exercise decode. No model/UI code changed. Run only after corpus import/other inference stop.",
    }
    atomic_json(path, report)
    engine = None
    sampler = MemorySampler()
    sampler.start()
    setup_start = time.perf_counter()
    try:
        from wikilens.gradio_demo import ProductEngine
        import transformers

        engine = ProductEngine()
        torch = engine.models.torch
        report["setup"] = {
            "wall_seconds": time.perf_counter() - setup_start,
            "cpu_sampling": sampler.finish(),
            "cpu_after": proc_memory(),
            "model_loads": engine.models.loads,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "attention_implementation": getattr(
                engine.models.model.config, "_attn_implementation", None
            ),
            "resident_context_tokens": engine.models.context_tokens,
            "actual_model_capacity": engine.models.model.config.max_position_embeddings,
            "devices": {
                "qwen": str(next(engine.models.model.parameters()).device),
                "reranker": str(next(engine.models.cross.parameters()).device),
                "embedding": str(next(engine.embedding_model.parameters()).device),
            },
            "dtypes": {
                "qwen": str(next(engine.models.model.parameters()).dtype),
                "reranker": str(next(engine.models.cross.parameters()).dtype),
                "embedding": str(next(engine.embedding_model.parameters()).dtype),
            },
        }
        if engine.models.context_tokens < 12288:
            raise ValueError(
                "Resident engine is not configured for the requested12288 context"
            )
        if any(not d.startswith("cuda") for d in report["setup"]["devices"].values()):
            raise ValueError("All three models must actually be resident on CUDA")
        report["status"] = "probing"
        atomic_json(path, report)
        primary = probe(engine, 12288, args.target_input_tokens, args.output_tokens)
        report["probes"].append(primary)
        atomic_json(path, report)
        if primary["status"] == "cuda_oom":
            print(
                "Primary CUDA OOM; running the single8192-budget fallback probe.",
                flush=True,
            )
            fallback = probe(
                engine, 8192, min(7600, 8192 - args.output_tokens), args.output_tokens
            )
            report["probes"].append(fallback)
            report["status"] = "primary_oom_fallback_" + fallback["status"]
        else:
            report["status"] = primary["status"]
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
    finally:
        sampler.finish()
        if engine is not None:
            engine.close()
            report["gpu_allocated_after_shutdown_bytes"] = (
                engine.models.torch.cuda.memory_allocated()
            )
        report["cpu_after_shutdown"] = proc_memory()
        atomic_json(path, report)
    print("REPORT", path, flush=True)
    print("STATUS", report["status"], flush=True)
    for p in report["probes"]:
        print(
            "PROBE",
            p["probe_context_limit"],
            p["status"],
            "input",
            p["input"]["input_tokens"],
            "output",
            p.get("actual_output_tokens"),
            "peakCUDA",
            p["peak_cuda_allocated_bytes"],
            "wall",
            round(p["probe_wall_seconds"], 3),
            flush=True,
        )
    if report["status"] != "passed":
        raise SystemExit(
            "Primary context smoke did not pass; inspect the preserved report before changing defaults."
        )


if __name__ == "__main__":
    main()
