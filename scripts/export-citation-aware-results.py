"""Export raw citation-aware answers and timings without altering model text."""

import argparse
import hashlib
import html
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("capture_directory", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.capture_directory
report = json.loads((root / "report.json").read_text())
plain = [
    "WikiLens · actual FP8 citation-aware outputs · unedited",
    "Full-query timing excludes database/model startup; generation timing is answer writing only.",
    f"Model startup: {report['model_setup_seconds']:.2f} s",
    "",
]
sections = []
for row in report["runs"]:
    case = row["case"]
    result_path = root / f"{case}.json"
    if not result_path.exists():
        sections.append(
            "<section><h2>"
            + html.escape(case)
            + "</h2><pre>"
            + html.escape(row.get("error", "No answer capture"))
            + "</pre></section>"
        )
        continue
    result = json.loads(result_path.read_text())
    context = result["input_capture"]["search_focus"]
    title = "talvisota · " + (context or "ei kontekstia")
    raw = result.get("raw_article", "")
    generation = result["stages"]["generation_seconds"]
    timing = (
        f"Full query: {row['seconds']:.2f} s · Answer generation: {generation:.2f} s"
    )
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    plain.extend([title, timing, f"Capture SHA-256: {digest}", "", raw, ""])
    sources = "".join(
        "<details><summary>"
        + html.escape(n["id"] + " · " + n["title"])
        + "</summary><p>"
        + html.escape(n["text"])
        + '</p><a href="'
        + html.escape(n["revision_url"], quote=True)
        + '">Wikipedia revision</a></details>'
        for n in result["nodes"]
    )
    sections.append(
        "<section><h2>"
        + html.escape(title)
        + "</h2><p>"
        + timing
        + '</p><p class="note">Unedited model output. Citations are model suggestions; factual support has not been independently verified.</p><div class="answer">'
        + html.escape(raw)
        + "</div><h3>Retrieved source passages</h3>"
        + sources
        + "</section>"
    )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.with_suffix(".txt").write_text("\n".join(plain) + "\n")
args.output.write_text(
    '<!doctype html><html lang="fi"><meta charset="utf-8"><title>WikiLens · FP8 captured results</title><style>body{max-width:1050px;margin:40px auto;padding:0 25px;background:#f6f7f2;color:#193b32;font:17px/1.7 system-ui}section{background:white;padding:30px;border:1px solid #dbe3d7;border-radius:12px;margin:25px 0}.answer,details p{white-space:pre-wrap}.answer{font-family:Georgia,serif;font-size:19px}.note{color:#806028;font-size:14px}details{margin:16px 0}summary{cursor:pointer}a{color:#176f52}</style><h1>WikiLens · Talvisota · FP8</h1><p>Actual model outputs, without editorial changes.</p><p>One-time model startup: '
    + f"{report['model_setup_seconds']:.2f}"
    + " s. Timings below exclude startup.</p>"
    + "".join(sections)
    + "</html>"
)
print(args.output)
