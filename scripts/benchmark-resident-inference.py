"""Paired saved English/Finnish inputs, resident FP16 Qwen and BGE reranker."""

import copy, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wikilens.resident_inference import ResidentModels
from wikilens.citation_pipelines import run_pipeline
from wikilens.inference_report import publish


def main():
    keep_resident = "--serve" in sys.argv
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    baseline_path = ROOT / "reports/paragraph-citations-20261114T123702Z.json"
    baseline = json.loads(baseline_path.read_text())
    english = json.loads(
        (ROOT / "reports/reranker-comparison-20261114T104507Z.json").read_text()
    )
    revision = english["variants"]["cross_encoder"]["run"]["runtime"]["revisions"][
        "qwen"
    ]
    from huggingface_hub import snapshot_download
    from wikilens.reranker_comparison import CROSS_MODEL, CROSS_REVISION

    downloads = {}
    for name, rev in [("Qwen/Qwen3-1.7B", revision), (CROSS_MODEL, CROSS_REVISION)]:
        t = time.perf_counter()
        snapshot_download(
            name,
            revision=rev,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        downloads[name] = time.perf_counter() - t
    report = {
        "created_at": stamp,
        "baseline_report": str(baseline_path.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "configuration": "Same FP16 pinned models, greedy editor prompt, thinking off, source-span paragraph citations; persistent weights. Same frozen inputs; retrieval excluded.",
        "timing_definitions": {
            "loads": "CPU weight/tokenizer loading and CUDA transfer, synchronized; excludes prior downloads",
            "prefill_first_forward_seconds": "Generation entry through completion of first model forward, with one CUDA synchronization. Includes generation setup; not exact time-to-first-token.",
            "prefill_forward_gpu_seconds": "CUDA events around the first forward",
            "decode_and_generation_overhead_seconds": "After first forward through generate completion; includes token selection, CPU grammar constraints and generation bookkeeping",
            "decode_tokens_per_second": "(output tokens minus 1) / decode-and-generation-overhead seconds; not a pure GPU kernel rate",
        },
        "downloads_seconds": downloads,
        "runs": [],
        "examples": {},
        "code_hashes": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in [
                "wikilens/resident_inference.py",
                "wikilens/citation_pipelines.py",
            ]
        },
    }
    path = ROOT / f"reports/resident-inference-{stamp}.json"
    models = ResidentModels(revision)
    report["loads"] = models.loads
    print("LOADS", models.loads, flush=True)
    try:
        models.generate(
            [{"role": "user", "content": "Reply with OK."}],
            max_new_tokens=8,
            max_time=30,
        )
        report["warmup_profiles"] = copy.deepcopy(models.profiles)
        for repetition in range(1, 3):
            for name in ["English", "Finnish"]:
                previous = baseline["examples"][name]
                capture = english if name == "English" else previous["capture"]
                settings = copy.deepcopy(previous["result"]["settings"])
                source = capture["variants"]["cross_encoder"]["run"]
                nodes = source["stages"]["postprocessing"]["nodes"]
                if nodes != previous["result"]["nodes"]:
                    raise ValueError("Baseline passages differ")
                count = len(models.profiles)
                result = run_pipeline(
                    capture,
                    "post_hoc",
                    settings,
                    runtime=models,
                    reference_lookup=models.lookup,
                )
                profiles = copy.deepcopy(models.profiles[count:])
                result["runtime"]["records"] = profiles
                cs = [c for s in result["sentences"] for c in s["citations"]]
                lookup = {n["id"]: n for n in nodes}
                result["mechanical_metrics"]["strict_exact_quotes"] = sum(
                    c["quote"]
                    == lookup[c["source_id"]]["text"][
                        c["selected_start_char"] : c["selected_end_char"]
                    ]
                    for c in cs
                )
                item = {
                    "language": name,
                    "repetition": repetition,
                    "result": result,
                    "profiles": profiles,
                    "baseline_posthoc_seconds": previous["result"]["seconds"],
                    "identical_draft_to_t4": result["initial_draft"]
                    == previous["result"]["initial_draft"],
                    "input_sha256": hashlib.sha256(
                        json.dumps(
                            {
                                "nodes": nodes,
                                "question": source["question"],
                                "settings": settings,
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                        ).encode()
                    ).hexdigest(),
                }
                report["runs"].append(item)
                report["examples"][f"{name} · pass {repetition}"] = {
                    "result": result,
                    "preparation_seconds": 0,
                }
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(
                    "RESULT",
                    name,
                    repetition,
                    result["status"],
                    round(result["seconds"], 3),
                    result["mechanical_metrics"],
                    "same draft",
                    item["identical_draft_to_t4"],
                    flush=True,
                )
        report["resident_model_devices"] = {
            "qwen": str(next(models.model.parameters()).device),
            "reranker": str(next(models.cross.parameters()).device),
        }
        report["resident_allocated_bytes_after_runs"] = (
            models.torch.cuda.memory_allocated()
        )
    finally:
        if not keep_resident:
            models.shutdown()
        report["models_retained_for_interactive_reuse"] = keep_resident
        report["gpu_allocated_after_runs_bytes"] = models.torch.cuda.memory_allocated()
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        (ROOT / "reports/resident-inference-latest.json").write_text(path.read_text())
        publish(report, path.with_suffix(".html"))
    print("REPORT", path, flush=True)
    if keep_resident:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def send_json(self, value, status=200):
                body = json.dumps(value, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send_json(
                    {
                        "ready": True,
                        "gpu": models.torch.cuda.get_device_name(0),
                        "resident_allocated_bytes": models.torch.cuda.memory_allocated(),
                        "report": str(path),
                    }
                )

            def do_POST(self):
                try:
                    data = json.loads(
                        self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    )
                    name = data["language"]
                    if name not in ("English", "Finnish"):
                        raise ValueError("Choose English or Finnish")
                    previous = baseline["examples"][name]
                    capture = english if name == "English" else previous["capture"]
                    count = len(models.profiles)
                    result = run_pipeline(
                        capture,
                        "post_hoc",
                        previous["result"]["settings"],
                        runtime=models,
                        reference_lookup=models.lookup,
                    )
                    result["runtime"]["records"] = copy.deepcopy(
                        models.profiles[count:]
                    )
                    self.send_json(result)
                except Exception as error:
                    self.send_json({"error": str(error)}, 400)

            def log_message(self, *args):
                pass

        print(
            "RESIDENT_READY: loopback 127.0.0.1:8766; retained until job timeout",
            flush=True,
        )
        try:
            HTTPServer(("127.0.0.1", 8766), Handler).serve_forever()
        finally:
            models.shutdown()


if __name__ == "__main__":
    main()
