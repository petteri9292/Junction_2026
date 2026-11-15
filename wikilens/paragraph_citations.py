"""Paragraph citation examples, including actual Finnish retrieval from S."""

import copy, hashlib, html, json, time
from pathlib import Path
from .citation_pipelines import run_pipeline
from .torch_runtime import TorchRuntime


def prepare_finnish(base, settings):
    from .hyde_search import generate_focused_hyde, retrieve_hyde
    from .reranker_comparison import prepare_chunks, rank_chunks

    original = copy.deepcopy(base["variants"]["cross_encoder"]["run"])
    original["question"] = settings["finnish_question"]
    original["search_focus"] = "history"
    runtime = TorchRuntime(
        device="cuda",
        context_tokens=settings["context_tokens"],
        qwen_revision=original["runtime"]["revisions"]["qwen"],
    )
    started = time.perf_counter()
    try:
        hyde = generate_focused_hyde(
            runtime,
            original["question"],
            "history",
            max_new_tokens=256,
            max_time=60,
            language="Finnish",
        )
        retrieval = retrieve_hyde(
            runtime,
            original["question"],
            hyde,
            top_k=settings["search_top_k"],
            language="fi",
        )
        runtime.close()
        chunks = prepare_chunks(
            retrieval["articles"],
            chunk_tokens=256,
            overlap_tokens=32,
            max_tokens=512,
            skip_reference_sections=True,
        )
        ranked = rank_chunks(
            runtime,
            original["question"],
            chunks,
            method="cross_encoder",
            search_focus="history",
            max_nodes=10,
            context_chars=7000,
            max_per_article=3,
            max_tokens=512,
        )
        original["stages"] = {
            "hyde": hyde,
            "input": retrieval,
            "postprocessing": ranked,
            "generation": {
                "messages": [
                    {
                        "role": "system",
                        "content": "Prepared input; actual editor prompt is recorded in the post-hoc run.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": original["question"],
                                "search_focus": "history",
                                "passages": [
                                    {
                                        "source_id": n["id"],
                                        "title": n["title"],
                                        "lang": n["lang"],
                                        "text": n["text"],
                                    }
                                    for n in ranked["nodes"]
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
            },
        }
        original["runtime"] = runtime.stats()
        original["preparation_seconds"] = time.perf_counter() - started
        return {"variants": {"cross_encoder": {"run": original}}}
    finally:
        runtime.close()


def publish(report, path):
    blocks = []
    for name, item in report["examples"].items():
        r = item["result"]
        m = r["mechanical_metrics"]
        blocks.append(
            "<section><h2>"
            + html.escape(name)
            + "</h2><p>"
            + html.escape(r["question"])
            + "</p><p>"
            + html.escape(json.dumps(m, ensure_ascii=False))
            + "</p>"
        )
        blocks.append(
            "<p>Post-hoc seconds: "
            + str(round(r["seconds"], 2))
            + "; input preparation seconds: "
            + str(round(item.get("preparation_seconds", 0), 2))
            + "</p>"
        )
        for p in r["sentences"]:
            blocks.append(
                "<h3>"
                + html.escape(p["section"])
                + " · "
                + p["claim_id"]
                + "</h3><p>"
                + html.escape(p["text"])
                + "</p>"
            )
            links = []
            for c in p["citations"]:
                n = next(n for n in r["nodes"] if n["id"] == c["source_id"])
                links.append(
                    '<a href="'
                    + html.escape(c["revision_url"], quote=True)
                    + '">['
                    + html.escape(c["source_id"])
                    + "] "
                    + html.escape(n["title"])
                    + " · "
                    + n["lang"]
                    + "</a>"
                )
                blocks.append(
                    "<details><summary>"
                    + html.escape(c["selected_span_id"])
                    + " · "
                    + html.escape(n["title"])
                    + " · "
                    + n["lang"]
                    + "</summary><blockquote>"
                    + html.escape(c["quote"])
                    + "</blockquote></details>"
                )
            blocks.append(
                "<p>"
                + (" · ".join(dict.fromkeys(links)) or "<em>[citation needed]</em>")
                + "</p>"
            )
        blocks.append(
            "<details><summary>Selected source passages and revisions</summary>"
        )
        for n in r["nodes"]:
            blocks.append(
                "<h4>"
                + n["id"]
                + " · "
                + html.escape(n["title"])
                + " · "
                + n["lang"]
                + "</h4><blockquote>"
                + html.escape(n["text"])
                + "</blockquote>"
            )
        blocks.append("</details></section>")
    path.write_text(
        """<!doctype html><meta charset="utf-8"><title>WikiLens · paragraph citations</title><style>body{font:17px/1.8 system-ui;background:#f5f4f0;color:#20302f;max-width:1100px;margin:40px auto}section{background:white;padding:35px;margin:25px 0;border-radius:14px}p,blockquote{white-space:pre-wrap}blockquote{background:#f3f6f4;margin:15px 0;padding:18px}summary{cursor:pointer}a{color:#236e64}</style><h1>Paragraph citations · English and Finnish</h1><p>Multiple source quotations may support one paragraph. Quotes are copied by Python; relevance and full factual support remain unverified. English reuses frozen passages; Finnish performs fresh Finnish HyDE, S retrieval restricted to fi, and BGE cross-encoder ranking. These are different input sets, not an isolated language benchmark. HyDE is never evidence. Timings separate input preparation from post-hoc inference.</p>"""
        + "".join(blocks)
    )
