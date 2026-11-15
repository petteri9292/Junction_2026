"""Replay exact saved calls, then regenerate the reviewed/cited fixed-evidence flow."""

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    import requests

    from wikilens.citation_pipelines import call_messages, run_pipeline
    from wikilens.torch_runtime import QWEN4_FP8_MODEL, QWEN4_FP8_REVISION
    from wikilens.vllm_runtime import VLLMRuntime

    source = {
        name: json.loads((args.baseline / (name + ".json")).read_text())
        for name in ["no-context", "video-games"]
    }
    for baseline in source.values():
        assert baseline["runtime"]["models"]["qwen"] == QWEN4_FP8_MODEL
        assert baseline["runtime"]["revisions"]["qwen"] == QWEN4_FP8_REVISION
    args.output.mkdir(parents=True, exist_ok=True)
    gpu_inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    save(args.output / "hardware.json", {"nvidia_smi": gpu_inventory})
    if "H100" not in gpu_inventory or len(gpu_inventory.splitlines()) != 1:
        raise RuntimeError(
            "Benchmark requires exactly one actual H100: " + gpu_inventory
        )
    provenance = {
        "baseline": str(args.baseline),
        "model": QWEN4_FP8_MODEL,
        "revision": QWEN4_FP8_REVISION,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *Path("wikilens").glob("*.py"),
                Path(__file__),
                Path("inference/vllm/uv.lock"),
            ]
        },
        "baseline_sha256": {
            n: hashlib.sha256((args.baseline / (n + ".json")).read_bytes()).hexdigest()
            for n in source
        },
        "git_head": os.environ.get("WIKILENS_SOURCE_COMMIT", "unknown; see hashes"),
        "note": "Hardware AND engine change together. Exact frozen calls isolate input variation; one observation each, not a general benchmark. Citation grammar backend changes from LM Format Enforcer to vLLM structured outputs.",
    }
    save(args.output / "provenance.json", provenance)
    command = [
        str(Path(sys.executable).parent / "vllm"),
        "serve",
        QWEN4_FP8_MODEL,
        "--revision",
        QWEN4_FP8_REVISION,
        "--tokenizer-revision",
        QWEN4_FP8_REVISION,
        "--dtype",
        "auto",
        "--max-model-len",
        "12288",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.7",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "12288",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--seed",
        "42",
        "--chat-template-content-format",
        "string",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    save(
        args.output / "server-config.json",
        {"command": command, "timeout_seconds": 900, "attempts": 1},
    )
    started = time.perf_counter()
    runtime = None
    with (args.output / "server.log").open("a") as log:
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while time.perf_counter() - started < 600:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server exited {server.returncode}; inspect server.log"
                    )
                try:
                    response = requests.get("http://127.0.0.1:8000/health", timeout=2)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError("vLLM server startup exceeded 600s; no retry")
            runtime = VLLMRuntime()
            report = {
                "model_setup_seconds": time.perf_counter() - started,
                "provenance": provenance,
                "frozen_calls": [],
                "pipeline_runs": [],
            }
            save(args.output / "report.json", report)
            for case, baseline in source.items():
                profiles = baseline["runtime"]["records"][-len(baseline["calls"]) :]
                for index, stage in [(0, "draft"), (1, "review")]:
                    save(
                        args.output / "progress.json",
                        {"case": case, "stage": "frozen " + stage},
                    )
                    old = baseline["calls"][index]
                    before = len(runtime.profiles)
                    record = call_messages(
                        runtime,
                        copy.deepcopy(old["messages"]),
                        max_new_tokens=old["max_new_tokens"],
                        max_time=old["max_time"],
                        schema=old.get("schema"),
                    )
                    assert record["messages"] == old["messages"]
                    save(
                        args.output / "frozen-calls" / f"{case}-{stage}.json",
                        {
                            "call": record,
                            "profile": runtime.profiles[-1]
                            if len(runtime.profiles) > before
                            else None,
                        },
                    )
                    if record["status"] != "returned":
                        raise RuntimeError(record.get("error", "Frozen call failed"))
                    row = {
                        "case": case,
                        "stage": stage,
                        "baseline_seconds": old["seconds"],
                        "vllm_seconds": record["seconds"],
                        "baseline_output_tokens": profiles[index]["output_tokens"],
                        "vllm_output_tokens": runtime.profiles[-1]["output_tokens"],
                        "output_equal": record["raw_output"] == old["raw_output"],
                        "prompt_equal": True,
                        "baseline_prompt_tokens": profiles[index]["input_tokens"],
                        "vllm_prompt_tokens": runtime.profiles[-1]["input_tokens"],
                    }
                    report["frozen_calls"].append(row)
                    save(args.output / "report.json", report)
                    print(json.dumps(row), flush=True)
            t = time.perf_counter()
            runtime.load_references()
            report["reference_model_setup_seconds"] = time.perf_counter() - t
            pipeline_report = {
                "question": "talvisota",
                "contexts": ["", "videopeli"],
                "answer_language": "Finnish",
                "source_language": "Finnish",
                "model_setup_seconds": report["model_setup_seconds"],
                "runtime": runtime.stats(),
                "fresh_search": False,
                "review_before_citations": True,
                "arm": "fixed-passages-h100-vllm",
                "runs": [],
            }
            for case, baseline in source.items():
                save(
                    args.output / "progress.json",
                    {"case": case, "stage": "fixed-evidence pipeline"},
                )
                source_capture = copy.deepcopy(baseline["input_capture"])
                source_capture["runtime"] = runtime.stats()
                profile_start = len(runtime.profiles)
                before = time.perf_counter()
                result = run_pipeline(
                    {"variants": {"cross_encoder": {"run": source_capture}}},
                    baseline["method"],
                    copy.deepcopy(baseline["settings"]),
                    runtime=runtime,
                    reference_lookup=runtime.lookup,
                    progress=lambda message, case=case, **unused: print(
                        case, message, flush=True
                    ),
                )
                result["runtime"]["records"] = runtime.profiles[profile_start:]
                result["input_capture"] = source_capture
                result["interactive_timings"] = {
                    **result["stages"],
                    "total_seconds": time.perf_counter() - before,
                }
                result["comparison_note"] = (
                    "Saved upstream retrieval/filtering reused, not rerun or timed; generation/review/reference/citation flow rerun."
                )
                assert result["nodes"] == baseline["nodes"]
                assert (
                    result["calls"][0]["messages"] == baseline["calls"][0]["messages"]
                )
                save(args.output / "pipeline" / (case + ".json"), result)
                row = {
                    "case": case,
                    "status": result["status"],
                    "seconds": result["interactive_timings"]["total_seconds"],
                    "timings": result["interactive_timings"],
                    "metrics": result["mechanical_metrics"],
                }
                pipeline_report["runs"].append(row)
                report["pipeline_runs"].append(row)
                save(args.output / "pipeline/report.json", pipeline_report)
                save(args.output / "report.json", report)
                print(json.dumps(row), flush=True)
            report["total_worker_seconds"] = time.perf_counter() - started
            save(args.output / "report.json", report)
            save(args.output / "progress.json", {"stage": "complete"})
        finally:
            if runtime is not None:
                runtime.shutdown()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == "__main__":
    main()
