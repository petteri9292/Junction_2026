"""One inference-only comparison: fixed evidence, then HyDE/direct full runs."""

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from wikilens.torch_runtime import QWEN4_MODEL, QWEN4_REVISION


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--qwen-model", default=QWEN4_MODEL)
    parser.add_argument("--qwen-revision", default=QWEN4_REVISION)
    args = parser.parse_args()
    from wikilens.citation_pipelines import run_pipeline
    from wikilens.gradio_demo import ProductEngine

    captures = {
        name: json.loads((args.baseline / f"{name}.json").read_text())
        for name in ("no-context", "video-games")
    }
    provenance = {
        "model": args.qwen_model,
        "revision": args.qwen_revision,
        "seed": 42,
        "git_head": os.environ.get(
            "WIKILENS_SOURCE_COMMIT", "unknown; see source hashes"
        ),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*Path("wikilens").glob("*.py"), Path(__file__), Path("uv.lock")]
        },
        "baseline_sha256": {
            name: hashlib.sha256(
                (args.baseline / f"{name}.json").read_bytes()
            ).hexdigest()
            for name in captures
        },
        "note": "Uncommitted source identified by hashes. No training. Greedy decoding unchanged. "
        "Old T4 versus new L40S timings confound hardware and model differences.",
    }
    save(args.output / "provenance.json", provenance)
    import random

    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    started = time.perf_counter()
    engine = ProductEngine(qwen_model=args.qwen_model, qwen_revision=args.qwen_revision)
    setup = time.perf_counter() - started
    summary = {"provenance": provenance, "model_setup_seconds": setup, "arms": {}}
    try:
        for arm in ("fixed-passages", "full-hyde", "full-direct"):
            out = args.output / arm
            report = {
                "question": "talvisota",
                "answer_language": "Finnish",
                "source_language": "Finnish",
                "contexts": ["", "videopelit"],
                "min_score": -2,
                "max_nodes": 25,
                "retrieval_top_k": 60,
                "target_words": 500,
                "article_tokens": 4000,
                "model_setup_seconds": setup,
                "runtime": engine.models.stats(),
                "arm": arm,
                "fresh_search": arm != "fixed-passages",
                "runs": [],
            }
            save(out / "report.json", report)
            for name, baseline in captures.items():

                def progress(value=None, desc="", arm=arm, name=name, **unused):
                    stage = desc or str(value)
                    print(arm, name, stage, flush=True)
                    save(
                        args.output / "progress.json",
                        {"arm": arm, "case": name, "stage": stage},
                    )

                before = time.perf_counter()
                try:
                    if arm == "fixed-passages":
                        source = copy.deepcopy(baseline["input_capture"])
                        source["runtime"] = engine.models.stats()
                        profile_start = len(engine.models.profiles)
                        result = run_pipeline(
                            {"variants": {"cross_encoder": {"run": source}}},
                            baseline["method"],
                            copy.deepcopy(baseline["settings"]),
                            progress=progress,
                            runtime=engine.models,
                            reference_lookup=engine.models.lookup,
                        )
                        result["runtime"]["records"] = engine.models.profiles[
                            profile_start:
                        ]
                        result["input_capture"] = source
                        result["interactive_timings"] = {
                            **result["stages"],
                            "total_seconds": result["seconds"],
                        }
                        result["comparison_note"] = (
                            "Upstream stages reused from 1.7B capture, not rerun or timed."
                        )
                        if (
                            result["calls"][0]["messages"]
                            != baseline["calls"][0]["messages"]
                        ):
                            raise RuntimeError(
                                "Fixed-evidence generation prompt changed"
                            )
                        if result["nodes"] != baseline["nodes"]:
                            raise RuntimeError("Fixed-evidence passages changed")
                    else:
                        result = engine.run(
                            "talvisota",
                            "Finnish",
                            "Finnish",
                            baseline["input_capture"]["search_focus"],
                            60,
                            500,
                            4000,
                            progress,
                            min_score=-2,
                            max_nodes=25,
                            retrieval_mode="hyde" if arm == "full-hyde" else "direct",
                        )
                    result["comparison_arm"] = arm
                    save(out / f"{name}.json", result)
                    row = {
                        "case": name,
                        "status": result["status"],
                        "seconds": time.perf_counter() - before,
                        "metrics": result["mechanical_metrics"],
                        "timings": result["interactive_timings"],
                    }
                except Exception as error:  # noqa: BLE001 -- preserve failed cases without retrying
                    row = {
                        "case": name,
                        "status": "failed",
                        "seconds": time.perf_counter() - before,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    save(out / f"{name}-failure.json", row)
                report["runs"].append(row)
                save(out / "report.json", report)
                print(json.dumps(row), flush=True)
            summary["arms"][arm] = report
            save(args.output / "report.json", summary)
    finally:
        engine.close()
        summary["total_seconds"] = time.perf_counter() - started
        save(args.output / "report.json", summary)


if __name__ == "__main__":
    main()
