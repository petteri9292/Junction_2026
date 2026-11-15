"""Readable exact-input timing and output comparison from actual captures."""

import argparse
import html
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("capture_directory", type=Path)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = json.loads((args.capture_directory / "report.json").read_text())
escape = html.escape
rows = []
details = []
sections = []
plain = [
    "WikiLens · L40S Transformers versus H100 vLLM · actual outputs",
    "Same model/revision and frozen messages/passages. One observation per request. Hardware and engine both change.",
    "Times exclude server/model startup. Effective tokens/sec includes request handling, tokenization, prefill, decoding and response serialization.",
    "",
]
for row in report["frozen_calls"]:
    old_seconds = row["baseline_seconds"]
    new_seconds = row["vllm_seconds"]
    old_rate = row["baseline_output_tokens"] / old_seconds
    new_rate = row["vllm_output_tokens"] / new_seconds
    title = row["case"] + " · " + row["stage"]
    baseline = json.loads((args.baseline / (row["case"] + ".json")).read_text())
    frozen = json.loads(
        (
            args.capture_directory
            / "frozen-calls"
            / (row["case"] + "-" + row["stage"] + ".json")
        ).read_text()
    )
    index = 0 if row["stage"] == "draft" else 1
    old_text = baseline["calls"][index]["raw_output"]
    new_text = frozen["call"]["raw_output"]
    rows.append(
        "<tr><td>"
        + escape(title)
        + "</td>"
        + "".join(
            f"<td>{value:.2f}</td>"
            for value in [
                old_seconds,
                new_seconds,
                old_seconds / new_seconds,
                old_rate,
                new_rate,
            ]
        )
        + f"<td>{row['baseline_output_tokens']}/{row['vllm_output_tokens']}</td><td>{row['output_equal']}</td></tr>"
    )
    details.append(
        "<details><summary>"
        + escape(title)
        + ' · exact-input outputs</summary><div class="columns"><article><h3>L40S Transformers</h3><div>'
        + escape(old_text)
        + "</div></article><article><h3>H100 vLLM</h3><div>"
        + escape(new_text)
        + "</div></article></div></details>"
    )
    plain.extend(
        [
            title,
            f"L40S {old_seconds:.3f}s; H100 {new_seconds:.3f}s; wall-time ratio {old_seconds / new_seconds:.2f}x",
            f"Effective output tokens/s {old_rate:.2f} versus {new_rate:.2f}; outputs identical: {row['output_equal']}",
            "\nL40S OUTPUT\n",
            old_text,
            "\nH100 OUTPUT\n",
            new_text,
            "",
        ]
    )
for row in report.get("pipeline_runs", []):
    case = row["case"]
    result = json.loads(
        (args.capture_directory / "pipeline" / (case + ".json")).read_text()
    )
    baseline = json.loads((args.baseline / (case + ".json")).read_text())
    old_pipeline_seconds = sum(baseline["stages"].values())
    plain.extend(
        [
            case + " · fixed-evidence pipeline",
            f"L40S {old_pipeline_seconds:.3f}s; H100 {row['seconds']:.3f}s",
            "\nH100 REVIEWED OUTPUT\n",
            result.get("reviewed_draft", ""),
            "",
        ]
    )
    nodes = {n["id"]: n for n in result["nodes"]}
    used = list(
        dict.fromkeys(
            c["source_id"] for u in result["sentences"] for c in u["citations"]
        )
    )
    sources = "".join(
        "<li>["
        + escape(sid)
        + '] <a href="'
        + escape(nodes[sid]["revision_url"], quote=True)
        + '">'
        + escape(nodes[sid]["title"])
        + "</a></li>"
        for sid in used
        if sid in nodes
    )
    sections.append(
        "<section><h2>"
        + escape(case)
        + " · final reviewed text</h2><p>Fixed-evidence draft → review → citations: L40S "
        + f"{old_pipeline_seconds:.2f}"
        + " s; H100 "
        + f"{row['seconds']:.2f}"
        + ' s. Upstream retrieval is reused, excluded from both timings.</p><div class="columns"><article><h3>L40S Transformers</h3><div>'
        + escape(baseline.get("reviewed_draft", ""))
        + "</div></article><article><h3>H100 vLLM</h3><div>"
        + escape(result.get("reviewed_draft", result.get("error", "")))
        + "</div></article></div><h3>Sources attached to H100 output</h3><ul>"
        + sources
        + '</ul><a href="h100-vllm-cited-results.html#'
        + escape(case, quote=True)
        + '">H100 final answer with citations and exact quotations →</a></section>'
    )
args.output.parent.mkdir(parents=True, exist_ok=True)
quality_note = "Observed quality issues remain: the history review copies its draft unchanged, including “aloitti soton”; the game article includes “MSX-tyypin laskimille” and “julkaisu ei olenistunut”, and asserts a failed Commodore 64 release where the saved passage only describes a planned release. Faster inference has not resolved language or grounding errors."
plain.extend(["OBSERVED QUALITY", quality_note])
sections.insert(
    0,
    "<section><h2>Observed output quality</h2><p>"
    + escape(quality_note)
    + "</p></section>",
)
args.output.with_suffix(".txt").write_text("\n".join(plain) + "\n")
style = "body{max-width:1500px;margin:40px auto;padding:0 24px;background:#f6f7f2;color:#193b32;font:16px/1.7 system-ui}section,details{background:white;border:1px solid #dbe3d7;border-radius:12px;padding:24px;margin:24px 0}.columns{display:grid;grid-template-columns:1fr 1fr;gap:30px}article>div{white-space:pre-wrap;font-family:Georgia,serif}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #dbe3d7}.note{color:#806028}summary{cursor:pointer}a{color:#176f52}@media(max-width:900px){.columns{grid-template-columns:1fr}}"
args.output.write_text(
    '<!doctype html><html lang="en"><meta charset="utf-8"><title>WikiLens · H100 vLLM comparison</title><style>'
    + style
    + '</style><h1>WikiLens · H100 + vLLM comparison</h1><p class="note">Actual unedited model outputs. Same pinned FP8 checkpoint and saved prompts/passages. Hardware and runtime change together; this does not isolate GPU-only speedup. One observation per request. Factual support remains unverified.</p><p>Server/model startup: '
    + f"{report['model_setup_seconds']:.2f}"
    + " s, excluded below. Prefix caching disabled; requests sequential.</p><table><tr><th>Frozen request</th><th>L40S s</th><th>H100 s</th><th>Wall ratio ×</th><th>L40S tokens/s</th><th>H100 tokens/s</th><th>Tokens old/new</th><th>Same text</th></tr>"
    + "".join(rows)
    + "</table><p>Rates are output tokens divided by whole request wall time, including prefill. Output lengths may differ; consider token rates alongside elapsed time. Frozen review replays the exact saved original draft; the full pipeline below reviews its newly generated draft.</p>"
    + "".join(details)
    + "".join(sections)
    + "</html>"
)
print(args.output)
