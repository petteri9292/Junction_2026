"""Three real Qwen generation configurations over frozen cross-encoder passages."""

import copy
import hashlib
import html
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wikilens.prompt_comparison import VARIANTS, REVIEW_RUBRIC


def renderer():
    spec = importlib.util.spec_from_file_location(
        "notebook_runner", ROOT / "scripts/compare-postprocessing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(report, path):
    render = renderer()
    rows, cards = [], []
    for name, item in report["variants"].items():
        review = item.get("historical_review") or {}
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(v)) + "</td>"
                for v in [
                    name,
                    round(item["seconds"], 2),
                    len(item.get("final_text", "").split()),
                    item.get("reasoning_tokens"),
                    item.get("generated_tokens"),
                    item.get("finish_reason"),
                    item.get("citation_markers"),
                    review.get("score", "pending"),
                ]
            )
            + "</tr>"
        )
        checks = []
        for claim in review.get("claims", []):
            links = " ".join(
                '<a href="'
                + html.escape(s["url"], quote=True)
                + '" target="_blank" rel="noopener">'
                + html.escape(s["title"])
                + "</a>"
                for s in claim.get("sources", [])
            )
            checks.append(
                "<li><strong>"
                + html.escape(claim["verdict"])
                + "</strong> · "
                + html.escape(claim["claim"])
                + "<p>"
                + html.escape(claim["explanation"])
                + " "
                + links
                + "</p></li>"
            )
        cards.append(
            "<section><h2>"
            + html.escape(name)
            + '</h2><p class="label">AI draft · historical review by assistant · source support not automatically verified</p>'
            + render.prose_html(
                item.get("final_text") or item.get("error", "No final answer"),
                report["nodes"],
            )
            + "<hr><h3>Historical accuracy: "
            + html.escape(str(review.get("score", "pending")))
            + "/10</h3><p>"
            + html.escape(review.get("summary", "Review pending"))
            + "</p><ol>"
            + "".join(checks)
            + "</ol>"
            + "<details><summary>Exact prompt and configuration</summary><pre>"
            + html.escape(
                json.dumps(item["messages"], ensure_ascii=False, indent=2)
                + "\n"
                + json.dumps(item["config"], indent=2)
            )
            + "</pre></details>"
            + "<details><summary>Raw output (including native reasoning, if any)</summary><pre>"
            + html.escape(item.get("raw_output", ""))
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
        + '">Exact source revision</a></details>'
        for n in report["nodes"]
    )
    path.write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>WikiLens · Qwen prompt comparison</title><style>
body{margin:0;background:#f5f4f0;color:#20302f;font:16px/1.7 system-ui,sans-serif}main{max-width:1900px;padding:36px;margin:auto}h1,h2,h3{line-height:1.3}h1{font-size:36px}table{width:100%;border-collapse:collapse;margin:24px 0}td,th{padding:12px;text-align:left;border-bottom:1px solid #ccd5d2}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}section{background:#fff;padding:26px;border-radius:12px;border:1px solid #dfe5e2}.label{color:#855800;font-size:13px}p,blockquote{white-space:pre-wrap}pre{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}blockquote{margin:12px 0;padding:12px;background:#f4f6f3}details{margin:14px 0}summary{cursor:pointer}a{color:#236e64}@media(max-width:1100px){.cards{grid-template-columns:1fr}}
</style><main><h1>WikiLens · same evidence, three Qwen configurations</h1><p>"""
        + html.escape(report["question"])
        + """ · seven fixed cross-encoder passages · Qwen3-1.7B on Tesla T4.</p><p>Current and editor prompts use greedy decoding with native thinking disabled. Editor+thinking enables native thinking and sampling (temperature 0.6, top-p 0.95, top-k 20, seed 42). This compares configurations; it does not isolate thinking from decoding. Timing includes imports, model load, generation and cleanup; search/reranking are not rerun. Reasoning and the article share a 4000-token budget.</p><table><thead><tr><th>Variant</th><th>Seconds</th><th>Article words</th><th>Reasoning tokens</th><th>Total tokens</th><th>Finish</th><th>Citations</th><th>Historical accuracy /10</th></tr></thead><tbody>"""
        + "".join(rows)
        + '</tbody></table><div class="cards">'
        + "".join(cards)
        + "</div><h2>Shared evidence</h2>"
        + evidence
        + "<h2>Review rubric</h2><pre>"
        + html.escape(json.dumps(report["review_rubric"], indent=2))
        + "</pre></main></html>"
    )


def main():
    template = json.loads(
        (ROOT / "notebooks/05_qwen_prompt_comparison.ipynb").read_text()
    )
    namespace = {}
    exec(
        "".join(next(c for c in template["cells"] if c["id"] == "settings")["source"]),
        namespace,
    )
    settings = namespace["SETTINGS"]
    source = ROOT / settings["source_report"]
    capture = json.loads(source.read_text())
    source_run = capture["variants"]["cross_encoder"]["run"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "data/prompt-comparisons" / stamp
    directory.mkdir(parents=True)
    report_path = ROOT / f"reports/prompt-comparison-{stamp}.json"
    report = {
        "created_at": stamp,
        "settings": settings,
        "question": source_run["question"],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shared_user_message_sha256": hashlib.sha256(
            source_run["stages"]["generation"]["messages"][1]["content"].encode()
        ).hexdigest(),
        "nodes": source_run["stages"]["postprocessing"]["nodes"],
        "review_rubric": REVIEW_RUBRIC,
        "variants": {},
        "order": list(VARIANTS),
        "code_sha256": hashlib.sha256(
            (ROOT / "wikilens/prompt_comparison.py").read_bytes()
        ).hexdigest(),
        "limitation": "Single development question. Editor+thinking changes native thinking and decoding. Historical review is an assistant assessment, not human validation or a calibrated benchmark.",
    }
    run = renderer()
    for variant in VARIANTS:
        print("\nRUN:", variant, flush=True)
        n = copy.deepcopy(template)
        n["cells"] = [
            c
            for c in n["cells"]
            if c["cell_type"] == "markdown" or c["id"] in ("settings", "setup", variant)
        ]
        target = directory / f"{variant}-result.json"
        n["cells"].append(
            {
                "cell_type": "code",
                "id": "export-prompt-run",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "Path("
                    + repr(str(target))
                    + ").write_text(json.dumps(results["
                    + repr(variant)
                    + "],ensure_ascii=False,indent=2))\n"
                ],
            }
        )
        cell_times, wall = run.execute(n, directory / f"{variant}-executed.ipynb")
        result = json.loads(target.read_text())
        result["cell_seconds"] = cell_times
        result["notebook_wall_seconds"] = wall
        text = result.get("final_text", "")
        result["citation_markers"] = len(re.findall(r"\[S\d+\]", text))
        known = {node["id"] for node in report["nodes"]}
        result["unknown_citation_ids"] = sorted(
            set(re.findall(r"\[(S\d+)\]", text)) - known
        )
        result["user_message_sha256"] = hashlib.sha256(
            result["messages"][1]["content"].encode()
        ).hexdigest()
        report["variants"][variant] = result
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        (ROOT / f"reports/prompt-comparison-{stamp}-{variant}.md").write_text(
            "# "
            + variant
            + "\n\nUnverified AI draft; historical review recorded separately.\n\n"
            + text
            + "\n"
        )
        print(
            "RESULT:",
            variant,
            json.dumps(
                {
                    k: result.get(k)
                    for k in (
                        "status",
                        "seconds",
                        "generated_tokens",
                        "reasoning_tokens",
                        "final_tokens",
                        "finish_reason",
                        "citation_markers",
                    )
                }
            ),
            "words",
            len(text.split()),
            flush=True,
        )
    publish(report, report_path.with_suffix(".html"))
    (ROOT / "reports/prompt-comparison-latest.json").write_text(report_path.read_text())
    print("REPORT:", report_path, flush=True)


if __name__ == "__main__":
    main()
