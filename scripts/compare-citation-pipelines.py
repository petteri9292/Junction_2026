"""Two real citation pipelines over the same frozen cross-encoder evidence."""

import copy
import argparse
import hashlib
import html
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = ["citation_aware", "post_hoc"]


def renderer():
    spec = importlib.util.spec_from_file_location(
        "runner", ROOT / "scripts/compare-postprocessing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(report, path):
    rows, cards = [], []
    for name in report.get("order", report["variants"]):
        if name not in report["variants"]:
            continue
        result = report["variants"][name]
        m = result["mechanical_metrics"]
        review = result.get("citation_review") or {}
        rm = review.get("metrics", {})
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(v)) + "</td>"
                for v in [
                    name,
                    round(result["seconds"], 2),
                    m["words"],
                    m["factual_sentences"],
                    m["cited_sentences"],
                    rm.get("quote_match_rate", "pending"),
                    rm.get("citation_accuracy", "pending"),
                    rm.get("supported_coverage", "pending"),
                    rm.get("usable_with_provenance", "pending"),
                ]
            )
            + "</tr>"
        )
        blocks = []
        section = None
        for sentence in result["sentences"]:
            if sentence["section"] != section:
                blocks.append("<h3>" + html.escape(sentence["section"]) + "</h3>")
                section = sentence["section"]
            links = []
            quotations = []
            for citation in sentence["citations"]:
                label = (
                    "["
                    + citation["source_id"]
                    + "]"
                    + (
                        " ?"
                        if citation.get("quote") and not citation.get("quote_located")
                        else ""
                    )
                )
                if citation.get("revision_url"):
                    links.append(
                        '<a href="'
                        + html.escape(citation["revision_url"], quote=True)
                        + '" target="_blank" rel="noopener">'
                        + html.escape(label)
                        + "</a>"
                    )
                else:
                    links.append(html.escape(label))
                if citation.get("quote"):
                    quotations.append(
                        "<blockquote>"
                        + html.escape(citation["quote"])
                        + "</blockquote><p>"
                        + html.escape(citation.get("status", "unchecked"))
                        + "</p>"
                    )
            if not links and sentence.get("requires_citation", True):
                links = ['<span class="flag">[citation needed]</span>']
            blocks.append(
                "<p><small>"
                + html.escape(sentence["claim_id"])
                + "</small> "
                + html.escape(sentence["text"])
                + " "
                + " ".join(links)
                + "</p>"
            )
            if quotations:
                blocks.append(
                    "<details><summary>Supporting quotation(s)</summary>"
                    + "".join(quotations)
                    + "</details>"
                )
        checks = []
        for check in review.get("claims", []):
            checks.append(
                "<tr>"
                + "".join(
                    "<td>" + html.escape(str(check.get(k, ""))) + "</td>"
                    for k in ["claim_id", "support", "explanation"]
                )
                + "</tr>"
            )
        cards.append(
            "<section><h2>"
            + html.escape(name)
            + '</h2><p class="label">Editor prompt · thinking off · draft until citation support is reviewed</p>'
            + "".join(blocks)
            + "<hr><h3>Assistant citation review</h3><p>"
            + html.escape(review.get("summary", "Pending"))
            + "</p><table><tr><th>Claim</th><th>Support</th><th>Reason</th></tr>"
            + "".join(checks)
            + "</table>"
            + "<details><summary>Full pipeline trace</summary><pre>"
            + html.escape(json.dumps(result, ensure_ascii=False, indent=2))
            + "</pre></details></section>"
        )
    evidence = "".join(
        "<details><summary>["
        + html.escape(n["id"])
        + "] "
        + html.escape(n["title"])
        + " · "
        + html.escape(n["lang"])
        + "</summary><blockquote>"
        + html.escape(n["text"])
        + '</blockquote><a href="'
        + html.escape(n["revision_url"], quote=True)
        + '">Original revision</a></details>'
        for n in report["nodes"]
    )
    path.write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>WikiLens · two citation pipelines</title><style>
body{margin:0;background:#f5f4f0;color:#20302f;font:16px/1.75 system-ui,sans-serif}main{max-width:1700px;padding:36px;margin:auto}h1,h2,h3{line-height:1.3}h1{font-size:36px}table{width:100%;border-collapse:collapse;margin:24px 0;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd5d2;vertical-align:top}.cards{display:grid;grid-template-columns:1fr 1fr;gap:28px}section{background:#fff;padding:28px;border-radius:12px;border:1px solid #dfe5e2}.label,.flag{color:#855800;font-size:13px}small{color:#607674;font-size:11px}pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}blockquote{white-space:pre-wrap;margin:12px 0;padding:12px;background:#f4f6f3}details{margin:14px 0}summary{cursor:pointer}a{color:#236e64}@media(max-width:1000px){.cards{grid-template-columns:1fr}}
</style><main><h1>WikiLens · citation-aware vs post-hoc</h1><p>Same question, editor framing, pinned Qwen3-1.7B and seven cross-encoder passages. No article retrieval/reranking repeated. Citation-aware prose generation tracks inline IDs as sentences are written; it does not request a separate model quotation. Post-hoc drafting splits sentences, looks up candidates inside the same passage store, then uses Qwen for citation attribution.</p><p>Every factual sentence needs a citation; headings do not. No optional regeneration/removal is used. Timings include all inference stages, model loading and cleanup. Quote matches establish provenance only; assistant review assesses full claim support. These are one-example observations, not a calibrated benchmark.</p><p><strong>Reading the scores:</strong> citation accuracy is fully supported cited sentences / cited sentences. Supported coverage uses all factual sentences as its denominator. Support + provenance additionally requires known source IDs and a located quotation when the workflow requests one. A missing or altered quotation fails that last check even if the cited passage supports the sentence. Prose generation does not request quotations; its quote check is not applicable. Uncited sentences are not automatically false.</p><table><thead><tr><th>Pipeline</th><th>Total s</th><th>Words</th><th>Factual sentences</th><th>Cited sentences</th><th>Quotation matches</th><th>Citation accuracy</th><th>Supported coverage</th><th>Support + provenance</th></tr></thead><tbody>"""
        + "".join(rows)
        + '</tbody></table><div class="cards">'
        + "".join(cards)
        + "</div><h2>Shared passage store</h2>"
        + evidence
        + "</main></html>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["all", "citation_aware", "post_hoc"], default="all"
    )
    parser.add_argument("--baseline-report")
    args = parser.parse_args()
    template = json.loads(
        (ROOT / "notebooks/06_citation_pipeline_comparison.ipynb").read_text()
    )
    namespace = {}
    exec(
        "".join(next(c for c in template["cells"] if c["id"] == "settings")["source"]),
        namespace,
    )
    settings = namespace["SETTINGS"]
    source = ROOT / settings["source_report"]
    capture = json.loads(source.read_text())
    shared = capture["variants"]["cross_encoder"]["run"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "data/citation-comparisons" / stamp
    directory.mkdir(parents=True)
    path = ROOT / f"reports/citation-comparison-{stamp}.json"
    registry = json.dumps(
        shared["stages"]["postprocessing"]["nodes"], sort_keys=True, ensure_ascii=False
    )
    report = {
        "created_at": stamp,
        "question": shared["question"],
        "settings": settings,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "nodes": shared["stages"]["postprocessing"]["nodes"],
        "passage_registry_sha256": hashlib.sha256(registry.encode()).hexdigest(),
        "variants": {},
        "order": METHODS,
        "review_protocol": {
            "reviewer": "Codex assistant after generation",
            "citation_accuracy": "Number of cited factual sentences fully supported by their cited passage set / factual sentences with displayed citations. Partial support is not full credit.",
            "supported_coverage": "Number of factual sentences fully supported by displayed citations / all factual sentence units. Unresolved sentences remain visible.",
            "mechanical_checks": "ID membership and literal quotation match reported separately, not counted as semantic accuracy.",
            "scope": "Every factual part of compound sentences must be supported. Historical truth is separate from source entailment; this is one development example.",
        },
        "code_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "wikilens/citation_pipelines.py",
                "wikilens/prompt_comparison.py",
                "wikilens/torch_runtime.py",
            ]
        },
    }
    if args.baseline_report:
        previous = json.loads((ROOT / args.baseline_report).read_text())
        if previous["passage_registry_sha256"] != report["passage_registry_sha256"]:
            raise ValueError("Frozen passages differ from baseline")
        report["variants"] = {
            name: item
            for name, item in previous["variants"].items()
            if name != args.method
        }
        report["reused_baseline_report"] = args.baseline_report
    runner = renderer()
    for method in METHODS:
        if args.method != "all" and method != args.method:
            continue
        print("\nRUN:", method, flush=True)
        n = copy.deepcopy(template)
        n["cells"] = [
            c
            for c in n["cells"]
            if c["cell_type"] == "markdown" or c["id"] in ["settings", "setup", method]
        ]
        result_path = directory / f"{method}-run.json"
        n["cells"].append(
            {
                "cell_type": "code",
                "id": "export-citation-run",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "Path("
                    + repr(str(result_path))
                    + ").write_text(json.dumps(results["
                    + repr(method)
                    + "],ensure_ascii=False,indent=2))\n"
                ],
            }
        )
        try:
            cells, wall = runner.execute(n, directory / f"{method}-executed.ipynb")
            result = json.loads(result_path.read_text())
            result.update(cell_seconds=cells, notebook_wall_seconds=wall)
            report["variants"][method] = result
            print(
                "RESULT:",
                method,
                result["status"],
                "seconds",
                round(result["seconds"], 2),
                json.dumps(result["mechanical_metrics"]),
                flush=True,
            )
        except Exception as error:
            report.setdefault("failures", {})[method] = (
                f"{type(error).__name__}: {error}"
            )
            print("FAILED:", method, report["failures"][method], flush=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    publish(report, path.with_suffix(".html"))
    (ROOT / "reports/citation-comparison-latest.json").write_text(path.read_text())
    print("REPORT:", path, flush=True)


if __name__ == "__main__":
    main()
