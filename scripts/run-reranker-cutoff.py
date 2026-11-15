"""Cache real explicitly identified S/full-L candidate pools for manually reviewed reranker cutoffs.

Inference only: no fitting, training, answer synthesis, model judge or cloud jobs.
Run on the restored GPU project environment; no data/model work runs on import.
Threshold policy counts are diagnostics, never relevance-quality measurements.
"""

from __future__ import annotations
import argparse
import gc
import hashlib
import html
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_QWEN_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def nonoverlapping(ranked, limit):
    """A review pool independent of generation context length or article budgets."""
    selected = []
    for row in ranked:
        if any(
            (n["lang"], n["page_id"], n["rev_id"])
            == (row["lang"], row["page_id"], row["rev_id"])
            and max(
                0,
                min(n["end_char"], row["end_char"])
                - max(n["start_char"], row["start_char"]),
            )
            / min(len(n["text"]), len(row["text"]))
            > 0.5
            for n in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def policy_counts(ranked, config):
    """Simulate rank/cutoff/cap selection, without pretending counts are quality."""
    if not ranked:
        return []
    policies = [("absolute", t) for t in config["absolute_logit_cutoffs"]]
    policies += [("relative_gap", t) for t in config["relative_gap_cutoffs"]]
    output = []
    top_score = ranked[0]["rerank_score"]
    for kind, value in policies:
        threshold = (
            (-math.inf if value is None else value)
            if kind == "absolute"
            else top_score - value
        )
        passing = [p for p in ranked if p["rerank_score"] >= threshold]
        for cap in config["selection_caps"]:
            selected = nonoverlapping(passing, cap)
            output.append(
                {
                    "kind": kind,
                    "value": value,
                    "cap": cap,
                    "raw_passing_chunks": len(passing),
                    "selected_passages": len(selected),
                    "selected_characters": sum(len(p["text"]) for p in selected),
                    "selected_ids": [p["id"] for p in selected],
                    "context_fit_checked": False,
                }
            )
    return output


def review_sample(ranked, config):
    """Select across score strata; remove score/rank from the labeling records."""
    size = config.get("review_sample_size", 24)
    chosen = []

    def add(row):
        if row["id"] not in {p["id"] for p in chosen}:
            chosen.append(row)

    for row in nonoverlapping(ranked, min(8, size)):
        add(row)
    thresholds = [t for t in config["absolute_logit_cutoffs"] if t is not None]
    for threshold in thresholds:
        # One above and one below each proposed boundary avoids only seeing the
        # very best passages. This is purposive, not a representative sample.
        for side in (True, False):
            options = [p for p in ranked if (p["rerank_score"] >= threshold) == side]
            if options and len(chosen) < size:
                add(min(options, key=lambda p: abs(p["rerank_score"] - threshold)))
    remaining = [
        p
        for p in nonoverlapping(ranked, config["review_pool_size"])
        if p["id"] not in {q["id"] for q in chosen}
    ]
    rng = random.Random(config.get("seed", 42))
    rng.shuffle(remaining)
    for row in remaining:
        if len(chosen) >= size:
            break
        add(row)
    rng.shuffle(chosen)
    return [
        {
            k: p[k]
            for k in [
                "id",
                "title",
                "lang",
                "revision_url",
                "page_id",
                "rev_id",
                "start_char",
                "end_char",
                "heading_path",
                "text",
            ]
        }
        for p in chosen
    ]


def export_review(directory, question, ranking, config):
    sample = review_sample(ranking["ranked"], config)
    records = [
        {
            "query_id": question["id"],
            **p,
            "grade": None,
            "uncertain": None,
            "reason": "",
            "reviewer": "",
        }
        for p in sample
    ]
    atomic_json(
        directory / "review-labels-template.json",
        {
            "query": question,
            "label_protocol": config["label_protocol"],
            "records": records,
            "review_status": "unreviewed",
        },
    )
    scores = {
        p["id"]: {"score": p["rerank_score"], "rank": i + 1}
        for i, p in enumerate(ranking["ranked"])
    }
    atomic_json(
        directory / "review-score-key.json", {p["id"]: scores[p["id"]] for p in sample}
    )
    blocks = []
    for p in sample:
        blocks.append(
            "<article><h2>"
            + html.escape(p["title"])
            + " · "
            + html.escape(p["lang"])
            + "</h2><p><small>"
            + html.escape(p["id"])
            + "</small></p><p>"
            + html.escape(" > ".join(p["heading_path"]))
            + "</p><blockquote>"
            + html.escape(p["text"])
            + '</blockquote><a href="'
            + html.escape(p["revision_url"], quote=True)
            + '">Exact Wikipedia revision</a><p>Grade: ____ · uncertain: ____ · reason: ____________________</p></article>'
        )
    (directory / "review-blind.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Passage relevance review</title><style>body{font:16px/1.7 system-ui;max-width:1000px;margin:35px auto;color:#24392f}article{border:1px solid #cdd8cf;padding:25px;margin:25px 0;border-radius:12px}blockquote{white-space:pre-wrap;margin:0;background:#f4f6f2;padding:16px}small{overflow-wrap:anywhere}</style><h1>Score-blind passage review</h1><p>"""
        + html.escape(question["question"])
        + "<br>Context: "
        + html.escape(question["search_focus"])
        + "</p><p>Grade0 unrelated/wrong meaning; grade1 background only; grade2 useful specific evidence; grade3 direct substantial evidence. Do not infer corpus recall from this purposive reviewed sample.</p>"
        + "".join(blocks)
    )


def dataset_kind(config):
    value = config.get("dataset", config.get("corpus", "L"))
    if value in ("S", "s", "s-2026-10"):
        return "s-2026-10"
    if value in ("L", "l", "l-2026-10"):
        return "l-2026-10"
    raise ValueError("Explicit dataset must be S/s-2026-10 or L/l-2026-10")


def project_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def count_map(rows):
    return {r["lang"]: int(r["n"]) for r in rows}


def verified_s_sources(config):
    required = ["source_manifest_sha256", "import_report", "import_report_sha256"]
    for field in required:
        if not config.get(field):
            raise ValueError(f"S identity requires frozen configuration field {field}")
    directory = project_path(config.get("dataset_directory", "data/s-2026-10"))
    manifest_path = project_path(
        config.get("source_manifest", directory / "manifest.json")
    )
    manifest_sha = sha(manifest_path)
    if manifest_sha != config["source_manifest_sha256"]:
        raise ValueError("S source manifest hash differs from the frozen configuration")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("release") != "s-2026-10":
        raise ValueError("S manifest is not release s-2026-10")
    file_hashes = {}
    for name, record in manifest["files"].items():
        actual = sha(directory / name)
        if actual != record["sha256"]:
            raise ValueError(f"S source file hash mismatch: {name}")
        file_hashes[name] = actual
    report_path = project_path(config["import_report"])
    if sha(report_path) != config["import_report_sha256"]:
        raise ValueError("S import report hash differs from the frozen configuration")
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "complete"
        or report.get("dataset") != "s-2026-10"
        or report.get("source_manifest_sha256") != manifest_sha
        or report.get("file_sha256") != file_hashes
    ):
        raise ValueError(
            "S importer report is incomplete or refers to different sources"
        )
    return {
        "manifest_sha256": manifest_sha,
        "file_sha256": file_hashes,
        "import_report_sha256": config["import_report_sha256"],
        "import_report": report,
    }


def database_identity(config):
    from wikilens.database import connect
    from wikilens.torch_runtime import BGE_REVISION

    corpus = dataset_kind(config)
    s_sources = verified_s_sources(config) if corpus == "s-2026-10" else None
    with connect() as db, db.cursor() as cursor:
        cursor.execute(
            "SELECT DATABASE() AS database_name, VERSION() AS server_version"
        )
        info = cursor.fetchone()
        info["dataset"] = corpus
        cursor.execute("SELECT lang,COUNT(*) AS n FROM article GROUP BY lang")
        info["article_counts"] = cursor.fetchall()
        cursor.execute("SELECT lang,COUNT(*) AS n FROM article_vector GROUP BY lang")
        info["vector_counts"] = cursor.fetchall()
        cursor.execute("SELECT DISTINCT model FROM article_vector")
        info["provided_vector_model_labels"] = [r["model"] for r in cursor.fetchall()]
        if corpus == "s-2026-10":
            canonical = {"fi": 19285, "en": 5254, "sv": 4921}
            expected_articles = config.get(
                "expected_article_counts", config.get("expected_articles_by_language")
            )
            expected_vectors = config.get("expected_vector_counts", expected_articles)
            if expected_articles != canonical or expected_vectors != canonical:
                raise ValueError(
                    "S identity requires explicit canonical expected per-language article/vector counts"
                )
            if (
                count_map(info["article_counts"]) != expected_articles
                or count_map(info["vector_counts"]) != expected_vectors
            ):
                raise ValueError(
                    "S database article/vector counts do not match all29460 supplied rows"
                )
            imported = s_sources["import_report"]
            if (
                imported.get("database") != info["database_name"]
                or imported.get("article_counts") != expected_articles
                or imported.get("vector_counts") != expected_vectors
            ):
                raise ValueError(
                    "S import report does not identify the connected database and actual counts"
                )
            cursor.execute("SHOW TABLES LIKE 'wikilens_corpus_identity'")
            if not cursor.fetchone():
                raise ValueError("S database lacks the guarded corpus-identity record")
            cursor.execute(
                "SELECT corpus_id,source_manifest_sha256,file_sha256_json,status FROM wikilens_corpus_identity WHERE corpus_id=%s",
                (corpus,),
            )
            record = cursor.fetchone()
            if (
                not record
                or record["status"] != "complete"
                or record["source_manifest_sha256"] != s_sources["manifest_sha256"]
                or json.loads(record["file_sha256_json"]) != s_sources["file_sha256"]
            ):
                raise ValueError(
                    "Connected S corpus provenance differs from its checked source/import report"
                )
            info["corpus_identity"] = record
            info["source_identity"] = s_sources
            info["provided_vector_model_revision"] = (
                "unknown; supplied S artifacts do not identify a model commit"
            )
            info["query_embedding_model_revision"] = BGE_REVISION
        else:
            expected = config.get("expected_finnish_articles", 568416)
            if expected != 568416:
                raise ValueError(
                    "L default requires the complete568416-article corpus; use explicitS configuration for the smaller corpus"
                )
            for key in ["article_counts", "vector_counts"]:
                count = count_map(info[key]).get("fi", 0)
                if count != expected:
                    raise ValueError(
                        f"Full L identity guard: {key} has {count} fi rows; expected {expected}"
                    )
            cursor.execute("SHOW TABLES LIKE 'wikilens_corpus_import'")
            if not cursor.fetchone():
                raise ValueError("Full L database lacks completed importer provenance")
            cursor.execute(
                "SELECT corpus_id,expected_rows,next_source_row,status,source_sha256,model_revision FROM wikilens_corpus_import"
            )
            states = cursor.fetchall()
            matching = [
                r
                for r in states
                if r["expected_rows"] == 568416
                and r["next_source_row"] == 568416
                and r["status"] == "complete"
                and r["source_sha256"]
                == "05d5520fc272b0f879bdf62ea67e8bf853c55ce5975eb5393f20c78c086158ec"
                and r["model_revision"] == BGE_REVISION
            ]
            if len(matching) != 1:
                raise ValueError(
                    "Full L database must have exactly one matching completed source/model provenance record"
                )
            info["corpus_identity"] = matching[0]
            info["provided_vector_model_revision"] = matching[0]["model_revision"]
            info["query_embedding_model_revision"] = BGE_REVISION
        cursor.execute("SHOW CREATE TABLE article_vector")
        info["vector_table_definition"] = cursor.fetchone()
    return info


def prepare_models(config):
    from huggingface_hub import snapshot_download
    from wikilens.torch_runtime import BGE_MODEL, BGE_REVISION
    from wikilens.reranker_comparison import CROSS_MODEL, CROSS_REVISION

    records = {}
    for name, revision in [
        (BGE_MODEL, BGE_REVISION),
        ("Qwen/Qwen3-1.7B", config.get("qwen_revision", DEFAULT_QWEN_REVISION)),
        (CROSS_MODEL, CROSS_REVISION),
    ]:
        start = time.perf_counter()
        patterns = [
            "*.json",
            "*.txt",
            "*.model",
            "*.bin" if name == BGE_MODEL else "*.safetensors",
        ]
        snapshot_download(
            name,
            revision=revision,
            allow_patterns=patterns,
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        records[name] = {
            "revision": revision,
            "download_setup_seconds": time.perf_counter() - start,
        }
    return records


def execute_question(question, config, directory, identity, resume):
    from wikilens.torch_runtime import TorchRuntime
    from wikilens.hyde_search import generate_focused_hyde, retrieve_hyde
    from wikilens.reranker_comparison import prepare_chunks, rank_chunks

    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "identity.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(
                f"Query output already exists: {directory}; use --resume explicitly"
            )
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(
                "Resume identity mismatch: code/config/corpus/query changed"
            )
    else:
        atomic_json(manifest, identity)
    runtime = TorchRuntime(
        device="cuda",
        context_tokens=8192,
        qwen_revision=config.get("qwen_revision", DEFAULT_QWEN_REVISION),
    )
    stages = {}
    started = time.perf_counter()
    try:
        hyde_path = directory / "hyde.json"
        if hyde_path.exists():
            hyde = json.loads(hyde_path.read_text())
        else:
            hyde = generate_focused_hyde(
                runtime,
                question["question"],
                question["search_focus"],
                max_new_tokens=config.get("hyde_tokens", 256),
                max_time=config.get("hyde_seconds", 60),
                language={"en": "English", "fi": "Finnish", "sv": "Swedish"}.get(
                    question["language"], question["language"]
                ),
            )
            atomic_json(hyde_path, hyde)
        if hyde.get("status") != "completed":
            raise RuntimeError(
                "HyDE failed; preserved raw output: " + str(hyde.get("error"))
            )
        stages["hyde_seconds"] = hyde["seconds"]
        retrieval_path = directory / "retrieval.json"
        if retrieval_path.exists():
            retrieval = json.loads(retrieval_path.read_text())
        else:
            retrieval = retrieve_hyde(
                runtime,
                question["question"],
                hyde,
                top_k=config["retrieval_top_k"],
                language=config.get(
                    "source_language",
                    None if dataset_kind(config) == "s-2026-10" else "fi",
                ),
            )
            atomic_json(retrieval_path, retrieval)
        stages["retrieval_seconds"] = retrieval["seconds"]
        runtime.close()
        chunks_path = directory / "chunks.json"
        if chunks_path.exists():
            cached = json.loads(chunks_path.read_text())
            chunks = cached["chunks"]
            stages["chunk_seconds"] = cached["seconds"]
        else:
            tick = time.perf_counter()
            chunks = prepare_chunks(
                retrieval["articles"],
                chunk_tokens=config["chunk_tokens"],
                overlap_tokens=config["overlap_tokens"],
                max_tokens=config["max_pair_tokens"],
                skip_reference_sections=True,
            )
            stages["chunk_seconds"] = time.perf_counter() - tick
            atomic_json(
                chunks_path, {"chunks": chunks, "seconds": stages["chunk_seconds"]}
            )
        ranking_path = directory / "ranking.json"
        if ranking_path.exists():
            ranking = json.loads(ranking_path.read_text())
        else:
            ranking = rank_chunks(
                runtime,
                question["question"],
                chunks,
                method="cross_encoder",
                search_focus=question["search_focus"],
                max_nodes=config["review_pool_size"],
                context_chars=10**9,
                max_per_article=config["review_pool_size"],
                max_tokens=config["max_pair_tokens"],
                cross_batch_size=config.get("cross_batch_size", 4),
                progress=lambda message: print(
                    question["id"] + ": " + message, flush=True
                ),
            )
            atomic_json(ranking_path, ranking)
        stages["reranking_seconds"] = ranking["seconds"]
        pool = nonoverlapping(ranking["ranked"], config["review_pool_size"])
        atomic_json(
            directory / "review-pool.json",
            {
                "query": question,
                "passages": pool,
                "note": "Up to 100 high-ranked overlap-deduplicated passages. No generation context limit applied.",
            },
        )
        policies = policy_counts(ranking["ranked"], config)
        atomic_json(
            directory / "policy-counts.json",
            {
                "policies": policies,
                "note": "Counts/IDs only; no quality metrics until original passages are manually reviewed. Generation context fit is not established.",
            },
        )
        export_review(directory, question, ranking, config)
        result = {
            "query": question,
            "status": "completed",
            "stages": stages,
            "candidate_articles": len(retrieval["articles"]),
            "candidate_chunks": len(chunks),
            "review_pool_passages": len(pool),
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "source_input_sha256": sha(retrieval_path),
            "ranked_output_sha256": sha(ranking_path),
            "score_range": [
                ranking["ranked"][-1]["rerank_score"],
                ranking["ranked"][0]["rerank_score"],
            ]
            if ranking["ranked"]
            else None,
            "human_relevance_labels": "pending",
            "quality_metrics": "not_computed",
            "runtime": runtime.stats(),
        }
        atomic_json(directory / "summary.json", result)
        print(
            "QUERY_COMPLETE",
            question["id"],
            "articles",
            result["candidate_articles"],
            "chunks",
            len(chunks),
            "pool",
            len(pool),
            flush=True,
        )
        return result
    finally:
        runtime.close()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["development", "heldout", "all"], default="development"
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Optional exact query IDs, repeat flag for several",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models-ready", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    required = [
        "questions",
        "retrieval_top_k",
        "chunk_tokens",
        "overlap_tokens",
        "max_pair_tokens",
        "review_pool_size",
        "selection_caps",
        "absolute_logit_cutoffs",
        "relative_gap_cutoffs",
        "label_protocol",
    ]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing configuration field: {field}")
    questions = [
        q
        for q in config["questions"]
        if (args.split == "all" or q["split"] == args.split)
        and (not args.query or q["id"] in args.query)
    ]
    if not questions:
        raise ValueError("No questions selected")
    config_hash = sha(args.config)
    code_files = [
        "scripts/run-reranker-cutoff.py",
        "wikilens/hyde_search.py",
        "wikilens/reranker_comparison.py",
        "wikilens/heading_postprocessing.py",
        "wikilens/torch_runtime.py",
        "wikilens/database.py",
        "wikilens/evidence.py",
    ]
    code_hashes = {p: sha(ROOT / p) for p in code_files}
    args.output.mkdir(parents=True, exist_ok=True)
    if config.get("database"):
        os.environ["DB_NAME"] = config["database"]
    database = database_identity(config)
    base_identity = {
        "configuration_sha256": config_hash,
        "code_hashes": code_hashes,
        "database": database,
    }
    report_path = args.output / ("report-" + args.split + ".json")
    if report_path.exists() and not args.resume:
        raise FileExistsError(
            "Output report exists; choose a fresh directory or --resume explicitly"
        )
    setup_path = args.output / "model-setup.json"
    if not args.models_ready and not setup_path.exists():
        atomic_json(setup_path, prepare_models(config))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "identity": base_identity,
        "split": args.split,
        "queries": {},
        "failures": {},
        "measurement_scope": f"One measured inference pass/query. Checked corpus {dataset_kind(config)}; source-language filter {config.get('source_language', None if dataset_kind(config) == 's-2026-10' else 'fi')!r}. No answer generation. Raw scores retained. Precision/coverage need manual original-passage review; no corpus-recall claim.",
        "setup": json.loads(setup_path.read_text())
        if setup_path.exists()
        else {"models_ready_flag": True},
    }
    atomic_json(report_path, report)
    for question in questions:
        try:
            identity = {**base_identity, "query": question}
            report["queries"][question["id"]] = execute_question(
                question, config, args.output / question["id"], identity, args.resume
            )
        except Exception as error:
            report["failures"][question["id"]] = f"{type(error).__name__}: {error}"
            atomic_json(
                args.output / question["id"] / "failure.json",
                {
                    "query": question,
                    "error": report["failures"][question["id"]],
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "stages_preserved": True,
                },
            )
            print(
                "QUERY_FAILED",
                question["id"],
                report["failures"][question["id"]],
                flush=True,
            )
        atomic_json(report_path, report)
    print("REPORT", report_path, flush=True)
    if report["failures"]:
        raise SystemExit(
            "One or more queries failed. Preserved checkpoints; investigate before explicit --resume."
        )


if __name__ == "__main__":
    main()
