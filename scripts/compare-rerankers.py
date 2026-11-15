"""One larger shared retrieval, identical heading chunks, three real ranker runs."""

import argparse
import copy
import hashlib
import html
import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wikilens.reranker_comparison import MODELS, prepare_chunks


def executor():
    spec = importlib.util.spec_from_file_location(
        "notebook_runner", ROOT / "scripts/compare-postprocessing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_models():
    from huggingface_hub import snapshot_download

    records = {}
    for method in ("model2vec", "cross_encoder"):
        name, revision = MODELS[method]
        started = time.perf_counter()
        print("Preparing pinned model:", name, revision, flush=True)
        snapshot_download(
            name,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "sentencepiece.bpe.model",
            ],
            ignore_patterns=["onnx/*", "openvino/*", "*.bin"],
        )
        records[method] = {
            "model": name,
            "revision": revision,
            "download_setup_seconds": time.perf_counter() - started,
        }
    path = ROOT / "data/reranker-model-setup.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")
    return records


def expand_retrieval(source, count, destination):
    from wikilens.database import connect, _vector_search, vector_search_sql
    from wikilens.evidence import pack_vector, revision_url

    capture = json.loads(source.read_text())
    frozen = capture["stages"]["input"]
    started = time.perf_counter()
    packed = pack_vector(frozen["embedding"])
    with connect() as db, db.cursor() as cursor:
        rows = _vector_search(cursor, packed, limit=count)
        cursor.execute("EXPLAIN " + vector_search_sql(None), (packed, count))
        plan = cursor.fetchall()
        keys = [(r["lang"], r["page_id"]) for r in rows]
        cursor.execute(
            "SELECT lang,page_id,rev_id,wikidata_id,title,url,text FROM article WHERE (lang,page_id) IN ("
            + ",".join(["(%s,%s)"] * len(keys))
            + ")",
            tuple(v for key in keys for v in key),
        )
        full = {(r["lang"], r["page_id"]): r for r in cursor.fetchall()}
        articles = [
            {
                **full[(row["lang"], row["page_id"])],
                "distance": row["distance"],
                "retrieval_rank": rank,
                "origin": "frozen_hyde_expanded_vector_candidate",
                "revision_url": revision_url(
                    row["lang"], full[(row["lang"], row["page_id"])]["rev_id"]
                ),
            }
            for rank, row in enumerate(rows, 1)
        ]
    capture["created_at"] = datetime.now(timezone.utc).isoformat()
    capture["settings"].update(
        search_top_k=count,
        input_mode="saved",
        saved_retrieval_file=str(destination.relative_to(ROOT)),
    )
    capture["stages"]["input"] = {
        **frozen,
        "articles": articles,
        "candidates": rows,
        "query_plan": plan,
        "limit": count,
        "seconds": time.perf_counter() - started,
        "method": "same_frozen_hyde_embedding_expanded_retrieval",
    }
    for stage in ("postprocessing", "generation"):
        capture["stages"].pop(stage, None)
    capture["expanded_from"] = str(source.relative_to(ROOT))
    capture["status"] = "input_prepared"
    destination.write_text(json.dumps(capture, ensure_ascii=False, indent=2) + "\n")
    return capture


def publish(report, path, renderer):
    rows, cards = [], []
    for method, item in report["variants"].items():
        run = item["run"]
        post = run["stages"]["postprocessing"]
        answer = run["stages"]["generation"]
        m = item["metrics"]
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(v)) + "</td>"
                for v in [
                    method,
                    post["diagnostics"]["device"],
                    post["candidate_count"],
                    len(post["nodes"]),
                    f"{m['postprocessing_seconds']:.2f}",
                    f"{m['generation_seconds']:.2f}",
                    f"{m['combined_seconds']:.2f}",
                    m["output_words"],
                    m["inline_citation_markers"],
                ]
            )
            + "</tr>"
        )
        text = (
            answer.get("draft_text")
            or answer.get("raw_output")
            or answer.get("error", "No answer")
        )
        evidence = "".join(
            "<details><summary>["
            + html.escape(n["id"])
            + "] "
            + html.escape(n["title"])
            + " · "
            + html.escape(n["lang"])
            + "</summary><p>"
            + html.escape(" > ".join(n["heading_path"]))
            + "</p><blockquote>"
            + html.escape(n["text"])
            + '</blockquote><a href="'
            + html.escape(n["revision_url"], quote=True)
            + '">Original revision</a></details>'
            for n in post["nodes"]
        )
        cards.append(
            "<section><h2>"
            + html.escape(method)
            + '</h2><p class="label">'
            + html.escape(post["diagnostics"]["model"])
            + "<br>AI draft · factual support unverified</p>"
            + renderer.prose_html(text, post["nodes"])
            + "<hr><h3>Selected evidence</h3>"
            + evidence
            + "</section>"
        )
    path.write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>WikiLens · three rankers, thirty articles</title><style>
body{margin:0;background:#f5f4f0;color:#20302f;font:16px/1.7 system-ui,sans-serif}main{max-width:1900px;padding:36px;margin:auto}h1,h2,h3{line-height:1.3}h1{font-size:36px}table{width:100%;border-collapse:collapse;margin:24px 0}td,th{padding:12px;text-align:left;border-bottom:1px solid #ccd5d2}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}section{background:#fff;padding:26px;border-radius:12px;border:1px solid #dfe5e2}.label{color:#855800;font-size:13px}p,blockquote{white-space:pre-wrap}blockquote{font-size:14px;margin:12px 0;padding:12px;background:#f4f6f3}details{margin:14px 0}summary{cursor:pointer}a{color:#236e64}@media(max-width:1100px){.cards{grid-template-columns:1fr}}
</style><main><h1>WikiLens · same chunks, three ranking methods</h1><p>"""
        + html.escape(report["question"])
        + " · "
        + str(report["article_count"])
        + " articles · "
        + str(report["chunk_count"])
        + """ identical content chunks.</p><p>Same heading splitter, context budget and Qwen prompt. Jarbas Model2Vec is a static CPU embedder; BGE-M3 embeds on GPU; BGE reranker-v2-m3 scores query–passage pairs on GPU. Model downloads and shared retrieval/chunk preparation are excluded below and recorded separately. Cell times include imports and model loading. One measured run each; no quality verdict.</p><table><thead><tr><th>Method</th><th>Device</th><th>Candidates</th><th>Selected</th><th>Ranking s</th><th>Generation s</th><th>Combined s</th><th>Words</th><th>Citation markers</th></tr></thead><tbody>"""
        + "".join(rows)
        + '</tbody></table><div class="cards">'
        + "".join(cards)
        + "</div></main></html>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=int, default=30)
    parser.add_argument(
        "--source", default="reports/nodes-to-answer-20261114T094841336858Z.json"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--models-ready", action="store_true")
    args = parser.parse_args()
    if not args.models_ready:
        preparation = prepare_models()
    else:
        preparation = json.loads((ROOT / "data/reranker-model-setup.json").read_text())
    if args.prepare_only:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "data/reranker-comparisons" / stamp
    directory.mkdir(parents=True)
    source_path = ROOT / f"reports/reranker-input-{args.articles}-20261114.json"
    capture = expand_retrieval(ROOT / args.source, args.articles, source_path)
    chunk_settings = {
        "chunk_tokens": 384,
        "overlap_tokens": 48,
        "max_tokens": 512,
        "skip_reference_sections": True,
    }
    started = time.perf_counter()
    chunks = prepare_chunks(capture["stages"]["input"]["articles"], **chunk_settings)
    chunk_seconds = time.perf_counter() - started
    chunk_file = directory / "shared-chunks.json"
    chunk_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n")
    renderer = executor()
    report = {
        "created_at": stamp,
        "question": capture["question"],
        "search_focus": capture.get("search_focus"),
        "article_count": len(capture["stages"]["input"]["articles"]),
        "chunk_count": len(chunks),
        "source_report": str(source_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "chunk_file": str(chunk_file.relative_to(ROOT)),
        "chunk_sha256": hashlib.sha256(chunk_file.read_bytes()).hexdigest(),
        "shared_retrieval_seconds": capture["stages"]["input"]["seconds"],
        "shared_chunk_seconds": chunk_seconds,
        "model_setup": preparation,
        "settings": capture["settings"],
        "chunk_settings": chunk_settings,
        "timing_basis": "Fresh kernels per method. Entire ranking/generation cells include imports/model loading; downloads and identical shared retrieval/chunk preparation are separate.",
        "variants": {},
        "order": list(MODELS),
        "code_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "wikilens/reranker_comparison.py",
                "wikilens/heading_postprocessing.py",
                "wikilens/node_pipeline.py",
                "wikilens/torch_runtime.py",
            ]
        },
    }
    report_path = ROOT / f"reports/reranker-comparison-{stamp}.json"
    for method in MODELS:
        print("\nRUN:", method, flush=True)
        n = json.loads(
            (ROOT / "notebooks/04_dedicated_reranker_comparison.ipynb").read_text()
        )
        result_path = directory / f"{method}-run.json"
        for cell in n["cells"]:
            cid = cell.get("id")
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            if cid == "settings":
                cell["source"] = [
                    "SETTINGS = "
                    + repr(capture["settings"])
                    + "\nREADY_FOR_GENERATION = True\nBGE_SETTINGS = "
                    + repr(chunk_settings)
                    + "\nRERANK_SETTINGS = "
                    + repr({"method": method, "cross_batch_size": 8})
                    + "\n"
                ]
            if cid == "setup":
                cell["source"].append(
                    "\nshared_capture = json.loads(Path("
                    + repr(str(source_path))
                    + ").read_text())\nSHARED_CHUNKS_FILE = "
                    + repr(str(chunk_file))
                    + '\nnode_runtime.qwen_revision = shared_capture.get("runtime", {}).get("revisions", {}).get("qwen")\n'
                )
            if cid == "hyde":
                cell["source"] = [
                    'start_node_stage("hyde")\nnode_run["stages"]["hyde"] = {**shared_capture["stages"]["hyde"], "reused": True, "seconds": 0.0}\nprint("Shared cached HyDE; not measured again")\n'
                ]
            if cid == "load":
                cell["source"] = [
                    'start_node_stage("input")\nnode_run["stages"]["input"] = {**shared_capture["stages"]["input"], "reused": True, "seconds": 0.0}\nprint("Identical shared retrieval:",len(node_run["stages"]["input"]["articles"]),"articles")\n'
                ]
            if cid == "comparison":
                cell["source"] = [
                    'print("Full comparison report will be published after all variants finish.")\n'
                ]
        n["cells"].append(
            {
                "cell_type": "code",
                "id": "export-reranker-run",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "Path("
                    + repr(str(result_path))
                    + ").write_text(json.dumps(node_run,ensure_ascii=False,indent=2))\n"
                ],
            }
        )
        try:
            cells, wall = renderer.execute(n, directory / f"{method}-executed.ipynb")
            run = json.loads(result_path.read_text())
            answer = run["stages"]["generation"]
            text = answer.get("draft_text") or answer.get("raw_output") or ""
            metrics = {
                "postprocessing_seconds": cells["postprocessing"],
                "generation_seconds": cells["generation"],
                "combined_seconds": cells["postprocessing"] + cells["generation"],
                "output_words": len(text.split()),
                "inline_citation_markers": len(re.findall(r"\[S\d+\]", text)),
                "notebook_wall_seconds": wall,
                "cell_seconds": cells,
            }
            report["variants"][method] = {
                "run": run,
                "metrics": metrics,
                "executed_notebook": str(
                    (directory / f"{method}-executed.ipynb").relative_to(ROOT)
                ),
            }
            print("RESULT:", method, json.dumps(metrics), flush=True)
            (ROOT / f"reports/reranker-comparison-{stamp}-{method}.md").write_text(
                "# "
                + method
                + "\n\nUnverified AI draft from actual inference.\n\n"
                + text
                + "\n"
            )
        except Exception as error:
            report.setdefault("failures", {})[method] = (
                f"{type(error).__name__}: {error}"
            )
            print("FAILED:", method, report["failures"][method], flush=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["variants"]:
        publish(report, report_path.with_suffix(".html"), renderer)
        (ROOT / "reports/reranker-comparison-latest.json").write_text(
            report_path.read_text()
        )
    print("REPORT:", report_path, flush=True)
    if len(report["variants"]) != len(MODELS):
        raise SystemExit("A variant failed; its notebook and report are preserved.")


if __name__ == "__main__":
    main()
