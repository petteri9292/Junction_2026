"""Export actual captures as a readable local comparison, without GPU access."""

import argparse
import html
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("capture_directory", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = json.loads((args.capture_directory / "report.json").read_text())
escape = html.escape
arm = report.get("arm", "full-hyde")
model = report.get("runtime", {}).get("models", {}).get("qwen", "Qwen/Qwen3-1.7B")
run_note = (
    f"{model} · {arm}. Fixed passages reuse upstream stages; their timings exclude retrieval."
    if arm.startswith("fixed-passages")
    else f"{model} · {arm}."
)
sections = []
fixtures = {
    "format": "wikilens-demo-comparison-v1",
    "capture_run": args.capture_directory.name,
    "label": "Actual model outputs; assistant review stored separately",
    "arm": arm,
    "generator_model": model,
    "playback_is_inference_measurement": False,
    "examples": [],
}
plain = [
    "WikiLens · talvisota · Finnish · raw captured outputs",
    "Cutoff −2; 60 retrieved articles; at most 25 selected passages.",
    "Quotation provenance and factual support are separate checks.",
    run_note,
    "LLM review before citations: " + str(report.get("review_before_citations", False)),
    "",
]
for row in report["runs"]:
    name = row["case"]
    title = (
        "Ei kontekstia"
        if name == "no-context"
        else "Konteksti: " + report["contexts"][1]
    )
    plain.extend([title, f"Full run: {row['seconds']:.2f} s", ""])
    if row["status"] == "failed":
        plain.append(row["error"])
        sections.append(
            "<section><h2>"
            + title
            + "</h2><p>"
            + escape(row["error"])
            + "</p></section>"
        )
        continue
    result = json.loads((args.capture_directory / f"{name}.json").read_text())
    fixtures["examples"].append(
        {
            "id": name,
            "question": report["question"],
            "search_focus": result["input_capture"]["search_focus"],
            "answer_language": report["answer_language"],
            "status": result["status"],
            "settings": result["settings"],
            "paragraphs": result["sentences"],
            "nodes": result["nodes"],
            "hyde": result["input_capture"]["stages"]["hyde"],
            "corpus": result["input_capture"]["corpus"],
            "timings": result["interactive_timings"],
            "generator_model": model,
            "arm": arm,
            "initial_draft": result.get("initial_draft"),
            "reviewed_draft": result.get("reviewed_draft"),
            "review_status": result.get("review_status"),
            "model_revisions": result["runtime"]["revisions"],
            "claim_support_reviewed": False,
            "raw_capture": str(args.capture_directory / f"{name}.json"),
            "raw_capture_sha256": hashlib.sha256(
                (args.capture_directory / f"{name}.json").read_bytes()
            ).hexdigest(),
        }
    )
    node_map = {node["id"]: node for node in result["nodes"]}
    text = []
    evidence = []
    last_section = None
    for unit in result["sentences"]:
        if unit["section"] != last_section:
            last_section = unit["section"]
            text.append("<h3>" + escape(last_section) + "</h3>")
            plain.append(last_section)
        links = []
        for index, citation in enumerate(unit["citations"]):
            node = node_map[citation["source_id"]]
            anchor = f"{name}-{unit['claim_id']}-{index}"
            links.append(f'<a class="cite" href="#{anchor}">[{escape(node["id"])}]</a>')
            quote = citation.get("literal_quote") or citation.get("quote") or ""
            evidence.append(
                f'<article id="{anchor}" tabindex="-1"><small>{escape(unit["claim_id"])} · {escape(node["id"])} · FI</small><h4>{escape(node["title"])}</h4><blockquote>{escape(quote)}</blockquote><a href="{escape(node["revision_url"], quote=True)}" target="_blank" rel="noopener">Lähteen Wikipedia-versio ↗</a><details><summary>Koko katkelma</summary><p>{escape(node["text"])}</p></details></article>'
            )
        markers = " ".join(links) or '<span class="missing">[lähde puuttuu]</span>'
        text.append("<p>" + escape(unit["text"]) + " " + markers + "</p>")
        plain.extend(
            [
                unit["text"],
                "Citations: " + ", ".join(c["source_id"] for c in unit["citations"]),
                "",
            ]
        )
    plain.append("SOURCE PASSAGES")
    for node in result["nodes"]:
        plain.extend(
            [node["id"] + " · " + node["title"], node["revision_url"], node["text"], ""]
        )
    plain.extend(
        ["TIMINGS (seconds)", json.dumps(result["interactive_timings"], indent=2), ""]
    )
    timings = "".join(
        "<tr><td>" + escape(key) + "</td><td>" + f"{value:.2f}" + "</td></tr>"
        for key, value in result["interactive_timings"].items()
        if isinstance(value, (float, int))
    )
    hyde = (
        result["input_capture"]["stages"]["hyde"].get("text")
        or "HyDE skipped: direct query retrieval."
    )
    sections.append(
        '<section id="'
        + escape(name, quote=True)
        + '"><header><h2>'
        + title
        + "</h2><span>"
        + f"{row['seconds']:.1f} s · {len(result['nodes'])} katkelmaa"
        + '</span></header><div class="columns"><div>'
        + "".join(text)
        + "</div><aside><h3>Lähteet</h3>"
        + ("".join(evidence) or "<p>Ei viitteitä.</p>")
        + "</aside></div><details><summary>HyDE ja mitatut ajat</summary><p>HyDE on hakua varten tuotettu hypoteesi. Se ei ole lähde.</p><blockquote>"
        + escape(hyde)
        + "</blockquote><table>"
        + timings
        + "</table></details></section>"
    )
style = """body{margin:0;background:#f5f5f1;color:#243a33;font:17px/1.75 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:50px}h1{font-size:42px;letter-spacing:-1.5px}h2{font-size:30px}h3{font-size:21px}h4{margin:8px 0}header{display:flex;align-items:center;justify-content:space-between}section{background:white;border:1px solid #dce4dd;border-radius:18px;padding:32px;margin:32px 0}.columns{display:grid;grid-template-columns:1.2fr 1fr;gap:40px}.columns>div>p{white-space:pre-wrap}aside{border-left:1px solid #e2e8e3;padding-left:28px}article{padding:18px;background:#f8faf7;border:1px solid #e0e7df;border-radius:12px;margin:20px 0;scroll-margin:25px}article:target{outline:3px solid #85af94}blockquote{margin:12px 0;white-space:pre-wrap;font-size:15px}.cite{font-size:12px;padding:3px 5px;background:#edf3ec;border-radius:4px}a{color:#287659}summary{cursor:pointer}details p{white-space:pre-wrap}small,header span{color:#68796f}table{font-size:14px;width:100%}td:last-child{text-align:right}.note,.missing{color:#8b6532}.note{font-size:14px}html{scroll-behavior:smooth}@media(max-width:900px){main{padding:20px}.columns{grid-template-columns:1fr}}"""
document = (
    '<!doctype html><html lang="fi"><meta charset="utf-8"><title>WikiLens — talvisota comparison</title><style>'
    + style
    + '</style><main><h1>WikiLens · Talvisota</h1><p>Sama kysely, kaksi kontekstia. Raakatulokset aidosta ajosta.</p><p class="note">Mallin luonnokset. Tarkat lainaukset eivät yksin varmista kaikkien väitteiden oikeellisuutta. Tallennettu toisto ei ole reaaliaikainen hakumittaus.</p>'
    + '<p class="note">'
    + escape(run_note)
    + "</p>"
    + "".join(sections)
    + "</main></html>"
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(document)
args.output.with_suffix(".txt").write_text("\n".join(plain) + "\n")
args.output.with_suffix(".fixtures.json").write_text(
    json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n"
)
print(args.output)
