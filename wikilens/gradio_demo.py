"""Notebook UI for the real paragraph-citation pipeline and labelled saved runs.

Importing this module never loads models, opens MariaDB or creates cloud jobs.
"""

import hashlib
import html
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
from .torch_runtime import (
    QWEN_MODEL,
    QWEN_REVISION,
    QWEN4_FP8_MODEL,
    QWEN4_FP8_REVISION,
)

EXAMPLES = {
    "Winter War · English": ("winter war", "English", "All languages"),
    "Talvisota · suomeksi": (
        "Milloin talvisota käytiin, mistä se johtui ja miten se päättyi?",
        "Finnish",
        "Finnish",
    ),
}
SOURCE_LANGUAGES = {
    "All languages": None,
    "Finnish": "fi",
    "English": "en",
    "Swedish": "sv",
}
DEFAULTS = {
    "context_tokens": 12288,
    "target_words": 500,
    "article_tokens": 4000,
    "article_seconds": 300,
    "max_nodes": 25,
    "reference_top_k": 3,
    "reference_pair_tokens": 1024,
    "attribution_batch_size": 2,
    "attribution_tokens": 1200,
    "attribution_seconds": 90,
    "max_spans_per_unit": 8,
    "citation_unit": "paragraph",
    "attribution_mode": "source_spans",
}
# Keep notebook/deepsearch post-hoc defaults independent of the live product.
LIVE_DEFAULTS = {
    **DEFAULTS,
    "generation_method": "post_hoc",
    "posthoc_prompt": "finnish_no_length_target",
    "target_words": None,
    "review_before_citations": True,
    "review_tokens": 4000,
    "review_seconds": 300,
}


def default_score_policy():
    p = Path(__file__).with_name("static") / "reranker-cutoff-policy.json"
    return (
        json.loads(p.read_text())
        if p.exists()
        else {"min_score": None, "relative_gap": None, "status": "not_calibrated"}
    )


class ProductEngine:
    """Keep Qwen, BGE-M3 and the citation reranker resident on the selected GPU."""

    def __init__(
        self, *, qwen_model=QWEN4_FP8_MODEL, qwen_revision=None, runtime_factory=None
    ):
        if qwen_revision is None and qwen_model == QWEN4_FP8_MODEL:
            qwen_revision = QWEN4_FP8_REVISION
        if qwen_revision is None and qwen_model == QWEN_MODEL:
            qwen_revision = QWEN_REVISION
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Live search needs the SSH project GPU environment. Use Saved run locally."
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Live search needs CUDA. Run this notebook on SSH, or select Saved run."
            )
        from .database import connect
        from .torch_runtime import BGE_REVISION

        with connect() as db, db.cursor() as cursor:
            cursor.execute("SELECT DATABASE() AS database_name")
            database_name = cursor.fetchone()["database_name"]
            cursor.execute("SHOW TABLES LIKE 'wikilens_corpus_identity'")
            small_corpus = None
            if cursor.fetchone():
                cursor.execute(
                    "SELECT corpus_id,source_manifest_sha256,file_sha256_json,status FROM wikilens_corpus_identity WHERE corpus_id=%s",
                    ("s-2026-10",),
                )
                small_corpus = cursor.fetchone()
            if small_corpus is not None:
                if small_corpus["status"] != "complete":
                    raise RuntimeError(
                        "Dataset S import is still in progress. Use Saved run until it completes."
                    )
                expected = {"fi": 19285, "en": 5254, "sv": 4921}
                counts = {}
                for table in ("article", "article_vector"):
                    cursor.execute(
                        f"SELECT lang,COUNT(*) AS n FROM {table} GROUP BY lang"
                    )
                    counts[table] = {r["lang"]: int(r["n"]) for r in cursor.fetchall()}
                    if counts[table] != expected:
                        raise RuntimeError(
                            f"Dataset S {table} counts differ from the verified29460-row release."
                        )
                if (
                    small_corpus["source_manifest_sha256"]
                    != "50d484246b526cd5f181ddd4cfdbdfd93a016246e87f4e0a4cf0be1eca30ac4f"
                ):
                    raise RuntimeError(
                        "Dataset S source identity differs from the pinned release manifest."
                    )
                cursor.execute("SELECT DISTINCT model FROM article_vector")
                labels = [r["model"] for r in cursor.fetchall()]
                self.corpus_info = {
                    "dataset": "s-2026-10",
                    "database": database_name,
                    "corpus_id": small_corpus["corpus_id"],
                    "expected_rows": 29460,
                    "next_source_row": 29460,
                    "status": "complete",
                    "article_counts": counts["article"],
                    "vector_counts": counts["article_vector"],
                    "source_manifest_sha256": small_corpus["source_manifest_sha256"],
                    "file_sha256": json.loads(small_corpus["file_sha256_json"]),
                    "provided_vector_model_labels": labels,
                    "provided_vector_model_revision": "unknown; supplied S artifacts do not identify a model commit",
                    "query_embedding_model_revision": BGE_REVISION,
                }
                self.available_source_languages = frozenset(expected)
            else:
                cursor.execute("SHOW TABLES LIKE 'wikilens_corpus_import'")
                if not cursor.fetchone():
                    raise RuntimeError(
                        "No verified S or completed full L corpus is available. Use Saved run or finish an import first."
                    )
                cursor.execute(
                    "SELECT corpus_id,expected_rows,next_source_row,status,source_sha256,model_revision FROM wikilens_corpus_import"
                )
                corpus = cursor.fetchone()
                if (
                    not corpus
                    or corpus["status"] != "complete"
                    or corpus["next_source_row"] != 568416
                    or corpus["expected_rows"] != 568416
                ):
                    count = corpus["next_source_row"] if corpus else 0
                    raise RuntimeError(
                        f"Full L import is still in progress ({count}/568416). Use Saved run or select the verified S database."
                    )
                if corpus["model_revision"] != BGE_REVISION:
                    raise RuntimeError(
                        "Database vectors and query embedding model revisions differ."
                    )
                if (
                    corpus["source_sha256"]
                    != "05d5520fc272b0f879bdf62ea67e8bf853c55ce5975eb5393f20c78c086158ec"
                ):
                    raise RuntimeError(
                        "Full L source identity differs from the pinned Finnish release."
                    )
                self.corpus_info = {
                    **corpus,
                    "dataset": "l-2026-10",
                    "database": database_name,
                }
                self.available_source_languages = frozenset({"fi"})
        from huggingface_hub import snapshot_download
        from .reranker_comparison import CROSS_MODEL, CROSS_REVISION
        from .resident_inference import ResidentModels

        snapshot_download(
            CROSS_MODEL,
            revision=CROSS_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        factory = runtime_factory or ResidentModels
        self.models = factory(
            qwen_revision,
            context_tokens=DEFAULTS["context_tokens"],
            qwen_model=qwen_model,
        )
        if (
            self.models.model.config.max_position_embeddings
            < DEFAULTS["context_tokens"]
        ):
            self.models.shutdown()
            raise RuntimeError("Configured context exceeds the model window")
        try:
            self.embedding_model, self.embedding_tokenizer, self.embedding_revision = (
                self.models.backend.load("bge", QWEN_REVISION)
            )
        except Exception:
            self.models.shutdown()
            raise
        self.models.torch.cuda.synchronize()

    def embed(self, texts, *, batch_size=1, max_tokens=1024):
        return self.models.backend.embed(
            self.embedding_model,
            self.embedding_tokenizer,
            texts,
            batch_size,
            max_tokens,
            1024,
        )

    def rank(
        self, question, focus, chunks, min_score=None, relative_gap=None, max_nodes=25
    ):
        from .reranker_comparison import select_nodes, CROSS_MODEL, CROSS_REVISION
        from .reranker_policy import filter_ranked

        torch = self.models.torch
        started = time.perf_counter()
        scores = []
        query = question + "\nSearch focus: " + focus
        for i in range(0, len(chunks), 4):
            batch = chunks[i : i + 4]
            inputs = self.models.cross_tokenizer(
                [[query, c["embedding_text"]] for c in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if inputs["input_ids"].shape[1] > 512:
                raise ValueError(
                    "Question plus source chunk exceeds 512 tokens. Shorten the question."
                )
            with torch.inference_mode():
                output = self.models.cross(**inputs.to("cuda"))
            scores.extend(output.logits.view(-1).float().cpu().tolist())
        ranked = sorted(
            ({**c, "rerank_score": float(score)} for c, score in zip(chunks, scores)),
            key=lambda c: (-c["rerank_score"], c["retrieval_rank"], c["start_char"]),
        )
        passing, score_filter = filter_ranked(
            ranked, min_score=min_score, relative_gap=relative_gap
        )
        nodes, used = select_nodes(
            passing,
            max_nodes=int(max_nodes),
            context_chars=10**9,
            max_per_article=int(max_nodes),
        )
        return {
            "nodes": nodes,
            "ranked": ranked,
            "candidate_count": len(chunks),
            "context_chars": used,
            "seconds": time.perf_counter() - started,
            "score_filter": score_filter,
            "diagnostics": {
                "model": CROSS_MODEL,
                "revision": CROSS_REVISION,
                "resident": True,
                "device": "cuda:0",
            },
        }

    def run(
        self,
        question,
        answer_language,
        source_language,
        focus,
        top_k,
        words,
        tokens,
        progress,
        *,
        max_nodes=25,
        min_score=None,
        relative_gap=None,
        retrieval_mode="hyde",
        diagnostics=None,
    ):
        from .hyde_search import (
            generate_focused_hyde,
            retrieve_hyde,
            retrieve_direct,
            rerank_question,
        )

        if retrieval_mode not in {"hyde", "direct"}:
            raise ValueError("retrieval_mode must be hyde or direct")
        from .reranker_comparison import prepare_chunks
        from .citation_pipelines import run_pipeline

        if source_language not in SOURCE_LANGUAGES:
            raise ValueError("Unknown source-edition filter")
        language_code = SOURCE_LANGUAGES[source_language]
        if (
            language_code is not None
            and language_code not in self.available_source_languages
        ):
            raise RuntimeError(
                "The active L corpus contains Finnish sources only. Choose Finnish or All languages."
            )
        started = time.perf_counter()
        profile_start = len(self.models.profiles)
        progress(0.08, desc="Writing HyDE retrieval aid")
        automatic_language = answer_language == "Auto"
        if automatic_language and retrieval_mode != "hyde":
            raise ValueError("Automatic language detection requires HyDE retrieval")
        hyde = (
            generate_focused_hyde(
                self.models,
                question,
                focus,
                max_new_tokens=512 if automatic_language else 256,
                max_time=60,
                language=answer_language,
            )
            if retrieval_mode == "hyde"
            else {"status": "skipped", "seconds": 0, "reason": "direct_query_retrieval"}
        )
        if retrieval_mode == "hyde" and hyde["status"] != "completed":
            raise RuntimeError(hyde.get("error", "HyDE generation failed"))
        if automatic_language:
            answer_language = hyde["answer_language"]
        if diagnostics:
            diagnostics("hyde", hyde)
        progress(0.22, desc="Searching Wikipedia articles in MariaDB")
        retrieval = (
            retrieve_hyde(
                self, question, hyde, top_k=int(top_k), language=language_code
            )
            if retrieval_mode == "hyde"
            else retrieve_direct(
                self, question, focus, top_k=int(top_k), language=language_code
            )
        )
        if diagnostics:
            diagnostics("retrieval", retrieval)
        if not retrieval["articles"]:
            raise RuntimeError("No articles found for this language filter.")
        progress(0.36, desc="Chunking headings and ranking passages")
        t = time.perf_counter()
        chunks = prepare_chunks(
            retrieval["articles"],
            chunk_tokens=256,
            overlap_tokens=32,
            max_tokens=512,
            skip_reference_sections=True,
        )
        chunk_seconds = time.perf_counter() - t
        ranking_question = (
            rerank_question(question, hyde) if automatic_language else question
        )
        post = self.rank(
            ranking_question,
            focus,
            chunks,
            min_score=min_score,
            relative_gap=relative_gap,
            max_nodes=max_nodes,
        )
        post["ranking_question"] = ranking_question
        post["ranking_hint_role"] = (
            "retrieval_only_never_evidence" if ranking_question != question else None
        )
        if diagnostics:
            diagnostics("reranking", post)
        if not post["nodes"]:
            raise RuntimeError(
                "No passages passed the score cutoff. Lower it or try a broader question."
            )
        progress(0.47, desc="Selecting evidence for your requested context")
        from .focus_evidence import filter_focused_nodes

        before_focus = post["nodes"]
        nodes, focus_filter = filter_focused_nodes(
            self.models, question, focus, before_focus
        )
        if not nodes:
            raise RuntimeError(
                "No retrieved passages matched the question in this context. Try a more explicit question, another source edition or a broader context."
            )
        settings = {
            **LIVE_DEFAULTS,
            "target_words": None,
            "article_tokens": int(tokens),
            "answer_language": answer_language,
            "max_nodes": int(max_nodes),
            "rerank_min_score": min_score,
            "rerank_relative_gap": relative_gap,
        }
        from .reranker_policy import pack_generation_nodes

        focused_nodes = nodes
        nodes, context_packing = pack_generation_nodes(
            focused_nodes, question, focus, settings, self.models.tokenizer
        )
        if not nodes:
            raise RuntimeError(
                "No source passage fits the requested output/context budget. Reduce maximum output tokens."
            )
        post.update(
            pre_focus_nodes=before_focus,
            focused_nodes=focused_nodes,
            nodes=nodes,
            focus_filter=focus_filter,
            context_packing=context_packing,
            context_chars=sum(len(n["text"]) for n in nodes),
        )
        if diagnostics:
            diagnostics("selection", post)
        source = {
            "question": question,
            "search_focus": focus,
            "corpus": self.corpus_info,
            "runtime": self.models.stats(),
            "retrieval_mode": retrieval_mode,
            "stages": {
                "hyde": hyde,
                "input": retrieval,
                "postprocessing": post,
                "generation": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Input preparation; the actual editor prompt is retained in calls.",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "question": question,
                                    "search_focus": focus,
                                    "passages": [
                                        {
                                            "source_id": n["id"],
                                            "title": n["title"],
                                            "lang": n["lang"],
                                            "text": n["text"],
                                            "revision_url": n["revision_url"],
                                        }
                                        for n in post["nodes"]
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ]
                },
            },
        }
        progress(0.55, desc="Writing the article with Qwen")

        def stage(message, **unused):
            if message.startswith("Reviewing:"):
                progress(0.65, desc="Reviewing language and source support")
            elif message.startswith("Draft:"):
                progress(0.75, desc="Finding references for paragraphs")
            elif message.startswith("Attributed"):
                progress(0.9, desc="Selecting exact quotations")

        result = run_pipeline(
            {"variants": {"cross_encoder": {"run": source}}},
            settings["generation_method"],
            settings,
            progress=stage,
            runtime=self.models,
            reference_lookup=self.models.lookup,
        )
        result["runtime"]["records"] = self.models.profiles[profile_start:]
        result["runtime"]["revisions"]["bge"] = self.embedding_revision
        result["interactive_timings"] = {
            "hyde_seconds": hyde["seconds"],
            "retrieval_and_embedding_seconds": retrieval["seconds"],
            "chunk_seconds": chunk_seconds,
            "article_reranking_seconds": post["seconds"],
            "focus_filter_seconds": focus_filter["seconds"],
            **result["stages"],
            "total_seconds": time.perf_counter() - started,
        }
        result["input_capture"] = source
        result["answer_language"] = answer_language
        result["language_detection"] = {
            "automatic": automatic_language,
            "code": hyde.get("detected_language"),
            "fallback": hyde.get("language_fallback", False),
        }
        directory = ROOT / "data/gradio-runs"
        directory.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (directory / f"{stamp}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        if result["status"] == "failed":
            raise RuntimeError(result.get("error", "Generation failed"))
        if not result["sentences"]:
            raise RuntimeError(
                "The model returned no article. Inspect the settings and try again."
            )
        progress(1, desc="Ready")
        return result

    def close(self):
        self.embedding_model = self.embedding_tokenizer = None
        self.models.shutdown()


def saved_result(example):
    language = EXAMPLES[example][1]
    for name in ["resident-inference-latest.json", "paragraph-citations-latest.json"]:
        p = ROOT / "reports" / name
        if not p.exists():
            continue
        report = json.loads(p.read_text())
        if "runs" in report:
            entries = [r["result"] for r in report["runs"] if r["language"] == language]
            if entries:
                return entries[-1], report["created_at"]
        elif language in report.get("examples", {}):
            return report["examples"][language]["result"], report["created_at"]
    raise RuntimeError(
        "No saved example capture is available in reports/. Use Live search on SSH."
    )


def render_result(result, label):
    """One HTML component keeps citation anchors and evidence in the same DOM."""
    nodes = {n["id"]: n for n in result["nodes"]}
    answers = []
    evidence = []
    section = None
    prefix = hashlib.sha256((result["question"] + label).encode()).hexdigest()[:10]
    for p in result["sentences"]:
        pid = prefix + "-" + p["claim_id"]
        links = []
        seen = set()
        if p["section"] != section:
            answers.append("<h3>" + html.escape(p["section"]) + "</h3>")
            section = p["section"]
        for j, c in enumerate(p["citations"]):
            n = nodes.get(c["source_id"])
            if n is None:
                continue
            eid = pid + "-quote-" + str(j)
            if c["source_id"] not in seen:
                links.append(
                    '<a class="wl-cite" href="#'
                    + eid
                    + '">['
                    + html.escape(c["source_id"])
                    + "]</a>"
                )
                seen.add(c["source_id"])
            quotation = c.get("literal_quote") or c.get("quote")
            source_label = (
                "Selected quotation"
                if quotation
                else "Model-proposed source · Retrieved passage · Support not verified"
            )
            evidence.append(
                '<article class="wl-evidence" id="'
                + eid
                + '" tabindex="-1"><div class="wl-meta">'
                + html.escape(
                    p["claim_id"] + " · " + c["source_id"] + " · " + n["lang"].upper()
                )
                + "</div><h4>"
                + html.escape(n["title"])
                + '</h4><p class="wl-subtle">'
                + source_label
                + "</p><blockquote>"
                + html.escape(quotation or n["text"])
                + '</blockquote><a target="_blank" rel="noopener" href="'
                + html.escape(n["revision_url"], quote=True)
                + '">Wikipedia revision ↗</a> · <a href="#'
                + pid
                + '">Back to paragraph</a><details><summary>Full retrieved passage</summary><p>'
                + html.escape(n["text"])
                + "</p></details></article>"
            )
        markers = " ".join(links) or (
            '<span class="wl-missing">[citation needed]</span>'
            if p.get("requires_citation", True)
            else ""
        )
        answers.append(
            '<article id="'
            + pid
            + '" tabindex="-1"><p>'
            + html.escape(p["text"])
            + " "
            + markers
            + "</p></article>"
        )
    if result.get("source_list"):
        answers.append(
            '<h3>Lähteet</h3><p style="white-space:pre-wrap">'
            + html.escape(result["source_list"])
            + "</p>"
        )
    review_label = (
        "LLM-reviewed draft · Factual support not independently verified"
        if result.get("review_status") == "self_reviewed_unverified"
        else "Draft · Factual support not reviewed"
    )
    return (
        '<div class="wl-label">'
        + html.escape(label)
        + " · "
        + review_label
        + '</div><div class="wl-result"><section class="wl-answer"><h2>'
        + html.escape(result["question"])
        + "</h2>"
        + "".join(answers)
        + '</section><aside class="wl-sources"><h2>Evidence</h2><p class="wl-subtle">Click a citation to focus its source quotation.</p>'
        + ("".join(evidence) or "<p>No citations selected.</p>")
        + "</aside></div>"
    )


CSS = """
.gradio-container{max-width:1500px!important;margin:auto!important;background:#f7f8f5!important}
.wl-hero{padding:22px 4px 10px}.wl-hero h1{font-size:38px;letter-spacing:-1.5px;margin:0;color:#173f39}.wl-hero p{font-size:17px;color:#66746b}
.wl-result{display:grid;grid-template-columns:1.35fr 1fr;gap:28px;color:#253c34;font:16px/1.85 ui-sans-serif,system-ui}
.wl-answer,.wl-sources{background:white;border:1px solid #dce5dc;border-radius:16px;padding:28px;max-height:78vh;overflow:auto;scroll-behavior:smooth}
.wl-answer h2,.wl-sources h2{font-size:25px;line-height:1.35;letter-spacing:-.5px}.wl-answer h3{font-size:19px;margin-top:26px}.wl-answer p{white-space:pre-wrap;margin:20px 0}
.wl-result a{color:#267b63}.wl-cite{font-size:13px;font-weight:650;text-decoration:none;border-radius:4px;background:#edf4ee;padding:3px 5px;white-space:nowrap}
.wl-evidence{border:1px solid #e1e8df;border-radius:10px;padding:18px;margin:16px 0;scroll-margin:12px}.wl-evidence h4{margin:4px 0 10px;font-size:18px}.wl-evidence blockquote{margin:0 0 14px;white-space:pre-wrap;font-size:15px}.wl-meta,.wl-label,.wl-subtle{font-size:12px;color:#6c7a70}.wl-label{padding:15px 3px}.wl-evidence:target,.wl-answer article:target{outline:3px solid #8cbaa0;background:#f3faf3}.wl-result details p{white-space:pre-wrap;font-size:13px}.wl-result summary{cursor:pointer;margin-top:12px}.wl-missing{color:#9d661b;font-size:13px}@media(max-width:900px){.wl-result{grid-template-columns:1fr}}
"""
# Event delegation survives HTML result updates and works inside Gradio's DOM.
CITATION_JS = """element.addEventListener('click', (event) => {const a=event.target.closest('a[href^="#"]'); if(!a)return;const id=a.getAttribute('href').slice(1); const target=element.querySelector('[id="'+id+'"]');if(target){event.preventDefault();target.scrollIntoView({behavior:'smooth',block:'start'});target.focus({preventScroll:true});element.querySelectorAll('[data-focused]').forEach(x=>{x.removeAttribute('data-focused');x.style.outline='';});target.setAttribute('data-focused','true');target.style.outline='3px solid #8cbaa0';}});"""


class DemoController:
    def __init__(self, remote_url=None):
        self.engine = None
        self.lock = threading.Lock()
        self.remote_url = remote_url
        self.remote_client = None

    def close(self):
        with self.lock:
            if self.engine is not None:
                self.engine.close()
                self.engine = None

    def submit(
        self,
        question,
        mode,
        example,
        language,
        source_language,
        focus,
        top_k,
        words,
        tokens,
        max_nodes=25,
        min_score=None,
        relative_gap=None,
        progress=gr.Progress(),
    ):
        question = question.strip()
        if not question:
            raise gr.Error("Enter a Wikipedia question first.")
        try:
            if mode == "Saved run (offline)":
                if question != EXAMPLES[example][0] or language != EXAMPLES[example][1]:
                    raise ValueError(
                        "Saved run replays only the selected example and answer language. Select an example, or switch to Live search for a new question."
                    )
                result, stamp = saved_result(example)
                label = "Saved actual run · earlier S corpus · " + stamp
                recorded_focus = json.loads(
                    result["calls"][0]["messages"][1]["content"]
                ).get("search_focus", "")
                if focus.strip().casefold() != recorded_focus.strip().casefold():
                    raise ValueError(
                        f"This saved run used context {recorded_focus!r}. Select Live search to try another context."
                    )
                diagnostics = {
                    "mode": "saved",
                    "recorded_seconds": result["seconds"],
                    "timings": result.get("stages"),
                    "note": "Playback time is not inference time. Saved settings are fixed; switch to Live search to change them.",
                }
                baseline = ROOT / "reports/paragraph-citations-latest.json"
                if baseline.exists():
                    example_record = (
                        json.loads(baseline.read_text())
                        .get("examples", {})
                        .get(language, {})
                    )
                    capture = example_record.get("capture")
                    if capture:
                        diagnostics["hyde_retrieval_aid_not_evidence"] = (
                            capture["variants"]["cross_encoder"]["run"]["stages"]
                            .get("hyde", {})
                            .get("text")
                        )
                if "hyde_retrieval_aid_not_evidence" not in diagnostics:
                    source = ROOT / "reports/reranker-comparison-20261114T104507Z.json"
                    if source.exists():
                        diagnostics["hyde_retrieval_aid_not_evidence"] = (
                            json.loads(source.read_text())["variants"]["cross_encoder"][
                                "run"
                            ]["stages"]
                            .get("hyde", {})
                            .get("text")
                        )
            else:
                from .reranker_policy import filter_ranked

                filter_ranked([], min_score=min_score, relative_gap=relative_gap)
                if not 10 <= int(max_nodes) <= 40:
                    raise ValueError("Maximum answer passages must be between10 and40.")
                with self.lock:
                    if self.remote_url:
                        from gradio_client import Client

                        progress(0.1, desc="Running the live pipeline on the SSH GPU")
                        if self.remote_client is None:
                            self.remote_client = Client(self.remote_url, verbose=False)
                        return self.remote_client.predict(
                            question,
                            mode,
                            example,
                            language,
                            source_language,
                            focus,
                            top_k,
                            words,
                            tokens,
                            max_nodes,
                            min_score,
                            relative_gap,
                            api_name="/ask",
                        )
                    if self.engine is None:
                        progress(
                            0.01,
                            desc="Loading GPU models once; later questions reuse them",
                        )
                        self.engine = ProductEngine()
                    result = self.engine.run(
                        question,
                        language,
                        source_language,
                        focus,
                        top_k,
                        words,
                        tokens,
                        progress,
                        max_nodes=max_nodes,
                        min_score=min_score,
                        relative_gap=relative_gap,
                    )
                label = "Live pipeline"
                source = result["input_capture"]
                diagnostics = {
                    "mode": "live",
                    "timings": result["interactive_timings"],
                    "hyde_retrieval_aid_not_evidence": source["stages"]["hyde"]["text"],
                    "corpus": source.get("corpus"),
                    "retrieved_articles": len(source["stages"]["input"]["articles"]),
                    "selected_passages": len(result["nodes"]),
                    "language_filter": source["stages"]["input"]["language"],
                    "query_plan": source["stages"]["input"]["query_plan"],
                    "model_revisions": result["runtime"]["revisions"],
                    "focused_question": result.get("focused_question"),
                    "focus_evidence_selection": source["stages"]["postprocessing"].get(
                        "focus_filter"
                    ),
                    "reranker_score_filter": source["stages"]["postprocessing"].get(
                        "score_filter"
                    ),
                    "context_token_packing": source["stages"]["postprocessing"].get(
                        "context_packing"
                    ),
                    "focus_policy": "Use reader context to interpret the subject; ignore off-topic meanings even when they have more source text.",
                }
            count = sum(len(p["citations"]) for p in result["sentences"])
            note = (
                f"{count} model-proposed source references; factual support remains unverified."
                if result.get("method") == "citation_aware"
                else f"{count} quotations. Exact copying checks provenance; factual support remains unverified."
            )
            return (
                render_result(result, label),
                diagnostics,
                f"{label} · {len(result['sentences'])} {result.get('citation_unit', 'paragraph')} units · {note}",
            )
        except Exception as error:
            raise gr.Error(str(error)) from error


def build_demo(default_mode="Saved run (offline)", remote_url=None):
    controller = DemoController(remote_url=remote_url)
    score_policy = default_score_policy()
    with gr.Blocks(
        title="WikiLens · Wikipedia evidence", analytics_enabled=False
    ) as demo:
        gr.HTML(
            '<div class="wl-hero"><h1>WikiLens</h1><p>Ask Wikipedia. Follow the evidence.</p></div>'
        )
        gr.Markdown(
            "Live search uses the active verified corpus: S has Finnish, English and Swedish sources; full L has Finnish sources. You can ask and read answers in English or Finnish. Pipeline details identify the active corpus. Saved playback retains its earlier corpus and settings."
        )
        with gr.Row():
            mode = gr.Radio(
                ["Live search", "Saved run (offline)"], value=default_mode, label="Mode"
            )
            example = gr.Dropdown(
                list(EXAMPLES), value=next(iter(EXAMPLES)), label="Try an example"
            )
        question = gr.Textbox(
            value=EXAMPLES[next(iter(EXAMPLES))][0],
            label="Your question",
            placeholder="What would you like to understand?",
            lines=2,
        )
        with gr.Row():
            language = gr.Dropdown(
                ["English", "Finnish"], value="Finnish", label="Answer language"
            )
            sources = gr.Dropdown(
                list(SOURCE_LANGUAGES), value="Finnish", label="Source editions"
            )
            focus = gr.Textbox(
                value="history",
                label="Your perspective",
                placeholder="history, video games…",
            )
        with gr.Accordion("Answer and search settings", open=False):
            with gr.Row():
                top_k = gr.Slider(
                    10, 100, value=60, step=10, label="Articles to retrieve"
                )
                words = gr.Slider(
                    100, 1000, value=500, step=50, label="Target answer words"
                )
                tokens = gr.Slider(
                    512,
                    6000,
                    value=4000,
                    step=128,
                    label="Maximum output tokens (must fit context)",
                )
            with gr.Row():
                max_nodes = gr.Slider(
                    10, 40, value=25, step=5, label="Maximum answer passages"
                )
                min_score = gr.Number(
                    value=score_policy.get("min_score"),
                    label="Minimum BGE score (blank disables)",
                )
                relative_gap = gr.Number(
                    value=score_policy.get("relative_gap"),
                    label="Maximum gap below best score (blank disables)",
                )
            gr.Markdown(
                "Scores are relevance logits, not probabilities. Whole passages are packed into the model context with room reserved for your output. The model may use fewer passages or write less when evidence is limited."
            )
        with gr.Row():
            submit = gr.Button("Explore Wikipedia", variant="primary")
            reset = gr.Button("Reset")
        status = gr.Markdown(
            "Ready. Generated drafts may contain errors; inspect their evidence."
        )
        result = gr.HTML(
            '<p class="wl-subtle">Your answer and source quotations will appear here.</p>',
            js_on_load=CITATION_JS,
        )
        with gr.Accordion("HyDE, retrieval and timings", open=False):
            diagnostics = gr.JSON(label="Pipeline details")
        inputs = [
            question,
            mode,
            example,
            language,
            sources,
            focus,
            top_k,
            words,
            tokens,
            max_nodes,
            min_score,
            relative_gap,
        ]
        submit.click(
            controller.submit,
            inputs,
            [result, diagnostics, status],
            concurrency_limit=1,
            api_name="ask",
        )
        question.submit(
            controller.submit,
            inputs,
            [result, diagnostics, status],
            concurrency_limit=1,
            api_name=False,
        )
        example.change(
            lambda x: EXAMPLES[x], example, [question, language, sources], queue=False
        )
        reset.click(
            lambda: (
                "",
                '<p class="wl-subtle">Ready for a new question.</p>',
                None,
                "Ready.",
            ),
            outputs=[question, result, diagnostics, status],
            queue=False,
        )
    demo.queue(default_concurrency_limit=1)
    return demo, controller


def launch_demo(demo, *, port=7860, inline=True):
    return demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inline=inline,
        prevent_thread_lock=True,
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
        css=CSS,
        show_error=False,
        footer_links=[],
    )
