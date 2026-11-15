"""Show actual original/reviewed model outputs side by side without editing text."""

import argparse
import hashlib
import html
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("capture_directory", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--cited-report", help="Relative URL of the matching final cited report"
)
args = parser.parse_args()
report = json.loads((args.capture_directory / "report.json").read_text())
sections = []
plain = [
    "WikiLens · actual draft and LLM-revised text · no manual rewriting",
    "Self-review is not independent factual verification. Query timing excludes startup.",
    "",
]
for row in report["runs"]:
    path = args.capture_directory / (row["case"] + ".json")
    if not path.exists():
        sections.append(
            "<section><pre>"
            + html.escape(row.get("error", "Missing capture"))
            + "</pre></section>"
        )
        continue
    result = json.loads(path.read_text())
    context = result["input_capture"]["search_focus"] or "ei kontekstia"
    original = result["initial_draft"]
    revised = result.get("reviewed_draft", "")
    review_seconds = result["stages"].get("draft_review_seconds", 0)
    timing = f"Full query {row['seconds']:.2f} s · Draft {result['stages']['generation_seconds']:.2f} s · Review {review_seconds:.2f} s"
    plain.extend(
        [
            "talvisota · " + context,
            timing,
            "Capture SHA-256: " + hashlib.sha256(path.read_bytes()).hexdigest(),
            "\nORIGINAL MODEL DRAFT\n",
            original,
            "\nLLM-REVISED MODEL DRAFT\n",
            revised,
            "",
        ]
    )
    nodes = {node["id"]: node for node in result["nodes"]}
    used_ids = list(
        dict.fromkeys(
            c["source_id"] for unit in result["sentences"] for c in unit["citations"]
        )
    )
    source_items = []
    plain.append("SOURCES ATTACHED AFTER REVIEW")
    for source_id in used_ids:
        node = nodes.get(source_id)
        if node is None:
            continue
        title = html.escape(node["title"])
        url = node.get("revision_url")
        linked = (
            (
                '<a href="'
                + html.escape(url, quote=True)
                + '" target="_blank" rel="noopener">'
                + title
                + "</a>"
            )
            if url
            else title
        )
        source_items.append(
            "<li>["
            + html.escape(source_id)
            + "] "
            + linked
            + " ("
            + html.escape(node["lang"])
            + ")</li>"
        )
        plain.extend(
            [
                "[" + source_id + "] " + node["title"] + " (" + node["lang"] + ")",
                url or "No source URL supplied",
            ]
        )
    final_link = (
        (
            html.escape(args.cited_report, quote=True)
            + "#"
            + html.escape(row["case"], quote=True)
        )
        if args.cited_report
        else None
    )
    final_anchor = (
        (
            '<p><a href="'
            + final_link
            + '">Open final answer with paragraph citations and exact quotations →</a></p>'
        )
        if final_link
        else ""
    )
    sources = (
        '<div class="sources"><h3>Sources attached after review</h3><ul>'
        + "".join(source_items)
        + "</ul>"
        + final_anchor
        + "</div>"
    )
    sections.append(
        '<section id="'
        + html.escape(row["case"], quote=True)
        + '"><h2>talvisota · '
        + html.escape(context)
        + "</h2><p>"
        + timing
        + '</p><div class="columns"><article><h3>Original model draft</h3><div>'
        + html.escape(original)
        + "</div></article><article><h3>LLM-revised draft</h3><div>"
        + html.escape(revised)
        + "</div></article></div>"
        + sources
        + "</section>"
    )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.with_suffix(".txt").write_text("\n".join(plain) + "\n")
args.output.write_text(
    '<!doctype html><html lang="fi"><meta charset="utf-8"><title>WikiLens · review comparison</title><style>body{max-width:1500px;margin:40px auto;padding:0 24px;background:#f6f7f2;color:#193b32;font:17px/1.7 system-ui}section{background:white;border:1px solid #dbe3d7;border-radius:12px;padding:28px;margin:25px 0}.columns{display:grid;grid-template-columns:1fr 1fr;gap:35px}article>div{white-space:pre-wrap;font-family:Georgia,serif}h1{font-size:36px}.note{color:#806028}.sources{margin-top:24px;border-top:1px solid #dbe3d7;padding-top:12px}a{color:#176f52}@media(max-width:900px){.columns{grid-template-columns:1fr}}</style><h1>WikiLens · model self-review comparison</h1><p class="note">Actual pipeline outputs. Original and revised text are preserved unchanged. These draft views precede citation attachment; source lists and links to the final cited answers appear below each comparison. Revision is performed by the same model; factual correctness is not independently verified.</p>'
    + "".join(sections)
    + "</html>"
)
print(args.output)
