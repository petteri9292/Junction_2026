"""Readable detailed performance report; does not load models."""

import html, json, re
from .paragraph_citations import publish as publish_articles


def publish(report, path):
    publish_articles(report, path)
    rows = []
    profiles = []
    for item in report["runs"]:
        r = item["result"]
        s = r["stages"]
        m = r["mechanical_metrics"]
        values = [
            item["language"],
            item["repetition"],
            round(item["baseline_posthoc_seconds"], 2),
            round(r["seconds"], 3),
            round(s["generation_seconds"], 3),
            round(s["reference_lookup_seconds"], 3),
            round(s["attribution_seconds"], 3),
            m["words"],
            m["strict_exact_quotes"],
            m["proposed_citations"],
            item["identical_draft_to_t4"],
        ]
        rows.append(
            "<tr>"
            + "".join("<td>" + html.escape(str(x)) + "</td>" for x in values)
            + "</tr>"
        )
        for i, p in enumerate(item["profiles"]):
            values = [
                item["language"],
                item["repetition"],
                "draft" if i == 0 else f"attribution {i}",
                p["input_tokens"],
                p["output_tokens"],
                round(p["tokenization_seconds"], 4),
                round(p["schema_setup_seconds"], 4),
                round(p["prefill_first_forward_seconds"], 4),
                round(p["prefill_forward_gpu_seconds"], 4),
                round(p["decode_and_generation_overhead_seconds"], 4),
                round(p["decode_tokens_per_second"], 1),
                round(p["wall_seconds"], 4),
            ]
            profiles.append(
                "<tr>"
                + "".join("<td>" + html.escape(str(x)) + "</td>" for x in values)
                + "</tr>"
            )

    def table(headers, body):
        return (
            "<table><tr>"
            + "".join("<th>" + h + "</th>" for h in headers)
            + "</tr>"
            + "".join(body)
            + "</table>"
        )

    summary = "<h1>L40S · resident inference timings</h1><p>Same pinned FP16 Qwen3-1.7B, BGE reranker, editor prompt, thinking off and exact saved English/Finnish passages. Qwen and the reranker are loaded once and retained simultaneously across both passes. Retrieval is excluded for both baseline and new comparisons. Baseline T4 timings include model loading/swapping; this measures a combined hardware/runtime change, not GPU speed alone. Output equality and actual token counts are reported because greedy outputs can change across GPUs.</p>"
    summary += (
        "<h2>One-time preparation</h2><pre>"
        + html.escape(
            json.dumps(
                {
                    "downloads_seconds": report["downloads_seconds"],
                    "loads": report["loads"],
                },
                indent=2,
            )
        )
        + "</pre>"
    )
    summary += table(
        [
            "Language",
            "Pass",
            "T4 baseline s",
            "L40S resident s",
            "Draft s",
            "Lookup s",
            "Attribution s",
            "Words",
            "Exact quotes",
            "Quotes",
            "Identical draft",
        ],
        rows,
    )
    summary += "<h2>Per-call details</h2><p>Prefill wall time ends after the first model forward, with one CUDA synchronization. It is not exact time to first token. Decode time includes token selection, CPU JSON constraints and generation bookkeeping. Tokens/s uses output tokens minus one divided by this interval. The callback instrumentation adds some overhead. Startup/downloads and the small warmup are excluded from the request timings. All displayed articles remain unverified drafts.</p>"
    summary += table(
        [
            "Language",
            "Pass",
            "Call",
            "Input tokens",
            "Output tokens",
            "Tokenize s",
            "Schema setup s",
            "Prefill wall s",
            "First forward GPU s",
            "Decode + overhead s",
            "Decode tok/s",
            "Call total s",
        ],
        profiles,
    )
    summary += (
        "<h2>Timing definitions</h2><pre>"
        + html.escape(json.dumps(report["timing_definitions"], indent=2))
        + "</pre>"
    )
    text = path.read_text().replace(
        "<title>WikiLens · paragraph citations</title>",
        "<title>WikiLens · L40S resident inference</title>",
    )
    text = re.sub(
        r"<h1>.*?</h1><p>.*?</p>", lambda m: summary, text, count=1, flags=re.S
    )
    text = text.replace(
        "</style>",
        "table{border-collapse:collapse;font-size:12px;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap}body{max-width:1600px}</style>",
    )
    path.write_text(text)
