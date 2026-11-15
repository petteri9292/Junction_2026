"""Execute both real notebooks over one frozen retrieval using fresh GPU kernels.

One measured pass per variant; model loads are included. No training/new job.
Uses installed jupyter_client; does not require nbclient or nbformat.
"""

import argparse
import hashlib
import html
import json
import queue
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def execute(notebook, output_path):
    manager = KernelManager(kernel_name="junction-t4")
    manager.start_kernel(cwd=str(ROOT))
    client = manager.client()
    client.start_channels()
    durations = {}
    started = time.perf_counter()
    try:
        client.wait_for_ready(timeout=60)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            cell["outputs"] = []
            source = "".join(cell["source"])
            print("  Cell:", cell["id"], flush=True)
            cell_started = time.perf_counter()
            msg_id = client.execute(source, stop_on_error=True, allow_stdin=False)
            failed = False
            while True:
                if time.perf_counter() - cell_started > 1000:
                    raise TimeoutError(
                        "Notebook cell exceeded 1000 seconds: " + cell["id"]
                    )
                try:
                    message = client.get_iopub_msg(timeout=1)
                except queue.Empty:
                    if not manager.is_alive():
                        raise RuntimeError("Notebook kernel exited")
                    continue
                if message.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                kind, content = message["header"]["msg_type"], message["content"]
                if kind == "execute_input":
                    cell["execution_count"] = content["execution_count"]
                elif kind == "stream":
                    if (
                        cell["outputs"]
                        and cell["outputs"][-1].get("output_type") == "stream"
                        and cell["outputs"][-1]["name"] == content["name"]
                    ):
                        cell["outputs"][-1]["text"] += content["text"]
                    else:
                        cell["outputs"].append(
                            {
                                "output_type": "stream",
                                "name": content["name"],
                                "text": content["text"],
                            }
                        )
                    if cell["id"] == "postprocessing" and re.match(
                        r"^\d+/\d+:|^\d+ heading chunks|^BGE postprocessing:",
                        content["text"],
                    ):
                        print("   ", content["text"].strip(), flush=True)
                elif kind in ("display_data", "execute_result"):
                    output = {
                        "output_type": kind,
                        "data": content["data"],
                        "metadata": content.get("metadata", {}),
                    }
                    if kind == "execute_result":
                        output["execution_count"] = content["execution_count"]
                    cell["outputs"].append(output)
                elif kind == "error":
                    failed = True
                    cell["outputs"].append(
                        {
                            "output_type": "error",
                            **{k: content[k] for k in ("ename", "evalue", "traceback")},
                        }
                    )
                    print("  ERROR:", content["ename"], content["evalue"], flush=True)
                elif kind == "status" and content["execution_state"] == "idle":
                    break
            durations[cell["id"]] = time.perf_counter() - cell_started
            output_path.write_text(
                json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
            )
            print(f"  Done {cell['id']}: {durations[cell['id']]:.2f}s", flush=True)
            if failed:
                raise RuntimeError("Cell failed: " + cell["id"])
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    return durations, time.perf_counter() - started


def prose_html(text, nodes):
    lookup = {n["id"]: n for n in nodes}
    blocks = []
    for paragraph in (text or "").split("\n\n"):
        safe = html.escape(paragraph.strip())
        safe = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", safe)

        def link(match):
            node = lookup.get(match[1])
            return (
                (
                    '<a href="'
                    + html.escape(node["revision_url"], quote=True)
                    + '" target="_blank" rel="noopener">'
                    + match[0]
                    + "</a>"
                )
                if node
                else match[0] + " (unknown source)"
            )

        safe = re.sub(r"\[(S\d+)\]", link, safe)
        tag = "h3" if safe.startswith("#") and "\n" not in safe else "p"
        blocks.append(
            f"<{tag}>" + (safe.lstrip("# ") if tag == "h3" else safe) + f"</{tag}>"
        )
    return "\n".join(blocks)


def publish(report, destination):
    rows, cards = [], []
    filtered = (
        report.get("variants", {})
        .get("heading_bge", {})
        .get("run", {})
        .get("stages", {})
        .get("postprocessing", {})
        .get("config", {})
        .get("skip_reference_sections", False)
    )
    coverage = (
        "BGE scores all non-reference heading chunks."
        if filtered
        else "BGE scores all heading chunks, including references (initial run)."
    )
    observations = " ".join(report.get("quality_observations", {}).values())
    for key, variant in report["variants"].items():
        post = variant["run"]["stages"].get("postprocessing", {})
        answer = variant["run"]["stages"].get("generation", {})
        raw = (
            answer.get("draft_text")
            or answer.get("raw_output")
            or answer.get("error", "No output")
        )
        metrics = variant["metrics"]
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(value)) + "</td>"
                for value in [
                    key,
                    post.get("candidate_count"),
                    len(post.get("nodes", [])),
                    f"{metrics['postprocessing_seconds']:.2f}",
                    f"{metrics['generation_seconds']:.2f}",
                    f"{metrics['combined_seconds']:.2f}",
                    metrics["output_words"],
                    answer.get("status"),
                ]
            )
            + "</tr>"
        )
        passages = []
        for node in post.get("nodes", []):
            passages.append(
                "<details><summary>["
                + html.escape(node["id"])
                + "] "
                + html.escape(node["title"])
                + " · "
                + html.escape(node["lang"])
                + "</summary><p>"
                + html.escape(" > ".join(node.get("heading_path", [])))
                + "</p><blockquote>"
                + html.escape(node["text"])
                + '</blockquote><a href="'
                + html.escape(node["revision_url"], quote=True)
                + '">Original revision</a></details>'
            )
        cards.append(
            "<section><h2>"
            + html.escape(key)
            + '</h2><p class="label">AI draft · citation support unverified</p>'
            + prose_html(raw, post.get("nodes", []))
            + "<hr><h3>Selected evidence</h3>"
            + "".join(passages)
            + "</section>"
        )
    destination.write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>WikiLens · postprocessing comparison</title><style>
body{margin:0;background:#f5f4f0;color:#202b30;font:16px/1.75 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:40px}h1,h2,h3{line-height:1.3}h1{font-size:36px}table{width:100%;border-collapse:collapse;margin:28px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #d5d9d9}.cards{display:grid;grid-template-columns:1fr 1fr;gap:30px}section{background:white;padding:32px;border:1px solid #dfe4e3;border-radius:12px}a{color:#286664}p{white-space:pre-wrap}.label{font-size:13px;color:#855800}details{margin:16px 0}summary{cursor:pointer}blockquote{white-space:pre-wrap;font-size:14px;margin:14px 0;padding:14px;background:#f5f6f4}@media(max-width:900px){.cards{grid-template-columns:1fr}main{padding:20px}}
</style><main><h1>WikiLens · same articles, two postprocessors</h1><p>"""
        + html.escape(report["question"])
        + " · "
        + html.escape(report.get("search_focus") or "no focus")
        + """ focus · one real measured pass per variant on Tesla T4.</p><p>Cell timings include imports, model loading and display; shared cached HyDE/retrieval is excluded. Qwen sees ten passage windows; """
        + html.escape(coverage)
        + """ This compares two pipeline recipes, not an isolated reranker benchmark.</p><p class="label">"""
        + html.escape(observations)
        + """</p><table><thead><tr><th>Variant</th><th>Candidates</th><th>Selected</th><th>Postprocess s</th><th>Generation s</th><th>Combined s</th><th>Words</th><th>Status</th></tr></thead><tbody>"""
        + "".join(rows)
        + '</tbody></table><div class="cards">'
        + "".join(cards)
        + "</div></main></html>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-report", default="reports/nodes-to-answer-20261114T094841336858Z.json"
    )
    parser.add_argument(
        "--variant", choices=["all", "qwen_compression", "heading_bge"], default="all"
    )
    parser.add_argument(
        "--baseline-report",
        help="Keep an already measured other variant; source hash must match",
    )
    args = parser.parse_args()
    source = ROOT / args.input_report
    capture = json.loads(source.read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "data/comparisons" / stamp
    directory.mkdir(parents=True)
    settings = capture["settings"]
    variants = [
        ("qwen_compression", "02_postprocessing_generation.ipynb"),
        ("heading_bge", "03_heading_bge_generation.ipynb"),
    ]
    report = {
        "created_at": stamp,
        "question": capture["question"],
        "search_focus": capture.get("search_focus"),
        "source_report": str(source.relative_to(ROOT)),
        "source_sha256": digest,
        "settings": settings,
        "variants": {},
        "order": [v[0] for v in variants],
        "timing_basis": "Elapsed postprocessing/generation cells, including imports, model loads and display; one pass each in fresh kernels; no warmup; shared cached retrieval excluded.",
        "limitation": "Candidate coverage and scoring change together. Single question, no repeat-based latency statistics or quality verdict.",
        "code_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "wikilens/heading_postprocessing.py",
                ROOT / "wikilens/qwen_postprocessing.py",
                ROOT / "wikilens/torch_runtime.py",
                ROOT / "wikilens/node_pipeline.py",
            ]
        },
        "gpu_before": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    report_path = ROOT / "reports" / f"postprocessing-comparison-{stamp}.json"
    if args.baseline_report:
        previous = json.loads((ROOT / args.baseline_report).read_text())
        if previous["source_sha256"] != digest:
            raise ValueError("Baseline report retrieval differs")
        report["variants"] = {
            k: v for k, v in previous["variants"].items() if k != args.variant
        }
        report["reused_baseline_report"] = args.baseline_report
    for key, filename in variants:
        if args.variant != "all" and key != args.variant:
            continue
        print("\nRUN:", key, flush=True)
        notebook = json.loads((ROOT / "notebooks" / filename).read_text())
        variant_file = directory / (key + "-run.json")
        executed = directory / (key + "-executed.ipynb")
        for cell in notebook["cells"]:
            cid = cell.get("id")
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            if cid == "settings":
                cell["source"] = [
                    "SETTINGS = "
                    + repr(settings)
                    + '\nREADY_FOR_GENERATION = True\nBGE_SETTINGS = {"chunk_tokens": 384, "overlap_tokens": 48, "max_tokens": 512, "skip_reference_sections": True}\n'
                ]
            if cid == "comparison":
                cell["source"] = [
                    'print("Comparison report will be published after this execution; open the final HTML report.")\n'
                ]
            if cid == "setup":
                cell["source"].append(
                    "\nshared_capture = json.loads(Path("
                    + repr(str(source))
                    + ').read_text())\nnode_run["comparison_input_sha256"] = '
                    + repr(digest)
                    + "\n"
                )
            if cid == "hyde":
                cell["source"] = [
                    'start_node_stage("hyde")\nnode_run["stages"]["hyde"] = shared_capture["stages"]["hyde"]\nprint("Reusing frozen HyDE; excluded from comparison timing")\n'
                ]
            if cid == "load":
                cell["source"] = [
                    'start_node_stage("input")\nnode_run["stages"]["input"] = shared_capture["stages"]["input"]\nprint("Frozen retrieval:", len(node_run["stages"]["input"]["articles"]), "articles")\n'
                ]
        notebook["cells"].append(
            {
                "cell_type": "code",
                "id": "export-comparison-run",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "Path("
                    + repr(str(variant_file))
                    + ').write_text(json.dumps(node_run, ensure_ascii=False, indent=2))\nprint("Comparison run saved")\n'
                ],
            }
        )
        try:
            cell_times, elapsed = execute(notebook, executed)
            run = json.loads(variant_file.read_text())
            post, generation = (
                run["stages"]["postprocessing"],
                run["stages"]["generation"],
            )
            text = generation.get("draft_text") or generation.get("raw_output") or ""
            metrics = {
                "postprocessing_seconds": cell_times["postprocessing"],
                "generation_seconds": cell_times["generation"],
                "internal_postprocessing_seconds": post["seconds"],
                "internal_generation_seconds": generation.get("seconds", 0),
                "combined_seconds": cell_times["postprocessing"]
                + cell_times["generation"],
                "output_words": len(text.split()),
                "notebook_wall_seconds": elapsed,
                "cell_seconds": cell_times,
            }
            report["variants"][key] = {
                "run": run,
                "metrics": metrics,
                "executed_notebook": str(executed.relative_to(ROOT)),
            }
            print("RESULT:", key, json.dumps(metrics), flush=True)
        except Exception as error:
            report.setdefault("failures", {})[key] = f"{type(error).__name__}: {error}"
            print("FAILED:", key, report["failures"][key], flush=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["variants"]:
        html_path = report_path.with_suffix(".html")
        publish(report, html_path)
        (ROOT / "reports/postprocessing-comparison-latest.json").write_text(
            report_path.read_text()
        )
        print("REPORT:", report_path, flush=True)
        print("HTML:", html_path, flush=True)
    if len(report["variants"]) != 2:
        raise SystemExit(
            "One or more real runs failed; inspect preserved notebooks/reports."
        )


if __name__ == "__main__":
    main()
