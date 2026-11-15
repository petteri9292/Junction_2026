"""Capture two real product runs for offline demo playback (inference only)."""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def serve_prompts(engine, args, report):
    """Reuse the same model/DB for private queued requests until the hard deadline."""
    import queue

    import modal

    inbox = modal.Queue.from_name("wikilens-fp8-prompts", create_if_missing=True)
    session = {
        "status": "ready",
        "started_at_unix": time.time(),
        "closes_at_unix": args.keep_open_until,
        "accepts_requests_until_unix": args.keep_open_until - 450,
        "queue": "wikilens-fp8-prompts",
        "partition": args.output.name,
        "requests": [],
    }
    save(args.output / "session.json", session)
    while time.time() < args.keep_open_until:
        try:
            request = inbox.get(
                timeout=min(5, max(0.01, args.keep_open_until - time.time())),
                partition=args.output.name,
            )
        except queue.Empty:
            continue
        if request.get("action") == "stop":
            session["status"] = "stopped_on_request"
            break
        request_id = request["request_id"]
        if not request_id.isalnum():
            raise ValueError("Invalid request ID")
        output = args.output / "followups" / f"{request_id}.json"
        if output.exists():
            raise RuntimeError("Duplicate request; refusing another inference attempt")
        before = time.perf_counter()
        if time.time() > session["accepts_requests_until_unix"]:
            row = {"request_id": request_id, "status": "rejected_session_ending"}
            save(output, row)
        else:
            session["status"] = "answering"
            save(args.output / "session.json", session)
            try:

                def progress(value, desc="", request_id=request_id):
                    save(
                        args.output / "progress.json",
                        {"case": request_id, "progress": value, "stage": desc},
                    )

                result = engine.run(
                    request["question"],
                    "Finnish",
                    "Finnish",
                    request.get("context", ""),
                    60,
                    500,
                    4000,
                    progress,
                    min_score=-2,
                    max_nodes=25,
                )
                save(output, result)
                row = {
                    "request_id": request_id,
                    "question": request["question"],
                    "context": request.get("context", ""),
                    "status": result["status"],
                    "seconds": time.perf_counter() - before,
                    "timings": result["interactive_timings"],
                }
            except Exception as error:  # noqa: BLE001 -- retain failure without retrying
                row = {
                    "request_id": request_id,
                    "status": "failed",
                    "seconds": time.perf_counter() - before,
                    "error": f"{type(error).__name__}: {error}",
                }
                save(output, row)
        session["requests"].append(row)
        session["status"] = "ready"
        save(args.output / "session.json", session)
        print(json.dumps(row), flush=True)
    if session["status"] == "ready":
        session["status"] = "deadline_reached"
    save(args.output / "session.json", session)
    report["session"] = session
    save(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", default="videopelit")
    parser.add_argument("--keep-open-until", type=float, default=0)
    args = parser.parse_args()
    from wikilens.gradio_demo import LIVE_DEFAULTS, ProductEngine

    started = time.perf_counter()
    import random

    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    engine = ProductEngine()
    report = {
        "question": "talvisota",
        "contexts": ["", args.context],
        "min_score": -2,
        "max_nodes": 25,
        "retrieval_top_k": 60,
        "target_words": LIVE_DEFAULTS["target_words"],
        "article_tokens": 4000,
        "answer_language": "Finnish",
        "source_language": "Finnish",
        "fresh_search": True,
        "model_setup_seconds": time.perf_counter() - started,
        "runtime": engine.models.stats(),
        "review_before_citations": LIVE_DEFAULTS.get("review_before_citations", False),
        "method": LIVE_DEFAULTS["generation_method"],
        "posthoc_prompt": LIVE_DEFAULTS.get("posthoc_prompt"),
        "seed": 42,
        "git_head": os.environ.get(
            "WIKILENS_SOURCE_COMMIT", "unknown; see source hashes"
        ),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*Path("wikilens").glob("*.py"), Path(__file__), Path("uv.lock")]
        },
        "quantization_config": getattr(
            engine.models.model.config, "quantization_config", None
        ),
        "fp8_parameter_count": sum(
            p.numel()
            for p in engine.models.model.parameters()
            if p.dtype == torch.float8_e4m3fn
        ),
        "runs": [],
    }
    if hasattr(report["quantization_config"], "to_dict"):
        report["quantization_config"] = report["quantization_config"].to_dict()
    save(args.output / "report.json", report)
    try:
        for name, focus in [("no-context", ""), ("video-games", args.context)]:

            def progress(value, desc="", name=name):
                print(f"{name}: {value:.0%} {desc}", flush=True)
                save(
                    args.output / "progress.json",
                    {"case": name, "progress": value, "stage": desc},
                )

            before = time.perf_counter()
            try:
                result = engine.run(
                    "talvisota",
                    "Finnish",
                    "Finnish",
                    focus,
                    60,
                    500,
                    4000,
                    progress,
                    min_score=-2,
                    max_nodes=25,
                )
                save(args.output / f"{name}.json", result)
                row = {
                    "case": name,
                    "status": result["status"],
                    "seconds": time.perf_counter() - before,
                    "metrics": result["mechanical_metrics"],
                    "timings": result["interactive_timings"],
                }
            except Exception as error:  # noqa: BLE001 -- persist failed cases, without retrying
                row = {
                    "case": name,
                    "status": "failed",
                    "seconds": time.perf_counter() - before,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                save(args.output / f"{name}-failure.json", row)
                print(row["traceback"], flush=True)
            report["runs"].append(row)
            report["total_seconds"] = time.perf_counter() - started
            save(args.output / "report.json", report)
            print(json.dumps(row), flush=True)
        if args.keep_open_until > time.time():
            serve_prompts(engine, args, report)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
