"""Read-only, single-pass S retrieval diagnostic using already cached embeddings.

No inference, training, warmup requests, cache flushes, cloud jobs or DB writes.
The starter questions are development data; this is not answer-quality evaluation.
"""

import base64
import csv
import hashlib
import json
import math
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wikilens.database import _vector_search, connect, vector_search_sql
from wikilens.dataset import LANGUAGES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topic_order(rows):
    scores = defaultdict(float)
    for lang in LANGUAGES:
        for rank, row in enumerate((r for r in rows if r["lang"] == lang), 1):
            if row["wikidata_id"]:
                scores[row["wikidata_id"]] += 1 / (60 + rank)
    return sorted(scores, key=lambda q: (-scores[q], q))


def main():
    questions_file = ROOT / "data/evaluation/starter-questions.tsv"
    vectors_file = questions_file.with_name("question-vectors-bge-m3.json")
    cache = json.loads(vectors_file.read_text())
    with questions_file.open() as stream:
        questions = list(csv.DictReader(stream, delimiter="\t"))
    assert len(questions) == 37, "Frozen starter evaluation must have 37 questions."
    vectors = {}
    for question in questions:
        # Missing cache entries fail rather than calling a model.
        packed = base64.b64decode(cache[question["question"]], validate=True)
        assert len(packed) == 4096 and all(
            math.isfinite(x) for x in struct.unpack("<1024f", packed)
        )
        vectors[question["question"]] = packed
    diagnostics, results = {}, []
    with connect() as db, db.cursor() as cursor:
        cursor.execute("SELECT VERSION() AS version, @@mhnsw_ef_search AS ef_search")
        diagnostics["server"] = cursor.fetchone()
        for table in ("article", "article_vector"):
            cursor.execute("SHOW CREATE TABLE " + table)
            diagnostics["create_" + table] = cursor.fetchone()
            cursor.execute("SHOW INDEX FROM " + table)
            diagnostics["indexes_" + table] = cursor.fetchall()
        cursor.execute("SELECT lang,COUNT(*) AS articles FROM article GROUP BY lang")
        diagnostics["article_counts"] = cursor.fetchall()
        cursor.execute(
            "SELECT lang,COUNT(*) AS vectors FROM article_vector GROUP BY lang"
        )
        diagnostics["vector_counts"] = cursor.fetchall()
        cursor.execute("SELECT title,wikidata_id FROM article WHERE lang='fi'")
        topics = {r["title"]: r["wikidata_id"] for r in cursor.fetchall()}
        cursor.execute(
            "SELECT wikidata_id,lang,page_id FROM article WHERE wikidata_id IS NOT NULL"
        )
        availability = defaultdict(set)
        for r in cursor.fetchall():
            availability[r["wikidata_id"]].add((r["lang"], r["page_id"]))
        for i, question in enumerate(questions):
            expected = topics[question["expected_fi_title"]]
            vector = vectors[question["question"]]
            row = {
                "question": question["question"],
                "language": question["lang"],
                "expected_topic": expected,
                "available_expected_editions": sorted(availability[expected]),
                "methods": {},
            }
            # Alternate which approach is measured first to reduce fixed-order advantage.
            methods = (
                ("global9", "balanced3x3") if i % 2 == 0 else ("balanced3x3", "global9")
            )
            method_rows = {}
            for method in methods:
                started = time.perf_counter()
                if method == "global9":
                    hits = _vector_search(cursor, vector, limit=9)
                else:
                    hits = []
                    for lang in LANGUAGES:
                        hits.extend(_vector_search(cursor, vector, lang=lang, limit=3))
                elapsed = (time.perf_counter() - started) * 1000
                method_rows[method] = hits
                matched = {
                    (h["lang"], h["page_id"])
                    for h in hits
                    if h["wikidata_id"] == expected
                }
                ranked_topics = (
                    topic_order(hits)
                    if method == "balanced3x3"
                    else list(
                        dict.fromkeys(
                            h["wikidata_id"] for h in hits if h["wikidata_id"]
                        )
                    )
                )
                top_topic = ranked_topics[0] if ranked_topics else None
                row["methods"][method] = {
                    "hit_at_1_topic": top_topic == expected,
                    "hit_at_9_articles": bool(matched),
                    "languages_returned": len({h["lang"] for h in hits}),
                    "expected_edition_recall": len(matched)
                    / len(availability[expected]),
                    "retrieval_ms": round(elapsed, 3),
                    "top_topic": top_topic,
                    "topic_order": ranked_topics,
                    "articles": [
                        {
                            k: h[k]
                            for k in (
                                "lang",
                                "page_id",
                                "wikidata_id",
                                "title",
                                "distance",
                            )
                        }
                        for h in hits
                    ],
                }
            # Measure relational expansion separately; it is not counted as vector Hit@9.
            selected_topics = topic_order(method_rows["balanced3x3"])[:3]
            started = time.perf_counter()
            aligned = []
            if selected_topics:
                cursor.execute(
                    "SELECT lang,page_id,rev_id,wikidata_id,title FROM article WHERE wikidata_id IN ("
                    + ",".join(["%s"] * len(selected_topics))
                    + ") ORDER BY wikidata_id,lang,page_id",
                    tuple(selected_topics),
                )
                aligned = cursor.fetchall()
            matched = {
                (h["lang"], h["page_id"])
                for h in aligned
                if h["wikidata_id"] == expected
            }
            row["wikidata_expansion_top3"] = {
                "topics": selected_topics,
                "articles": aligned,
                "expansion_ms": round((time.perf_counter() - started) * 1000, 3),
                "expected_edition_recall": len(matched) / len(availability[expected]),
            }
            results.append(row)
            print(f"Completed {i + 1}/{len(questions)}", flush=True)
        # Diagnose the exact query shapes after measurements, not inside their timings.
        vector = vectors[questions[0]["question"]]
        diagnostics["explain"] = {}
        for lang in (None, *LANGUAGES):
            limit = 9 if lang is None else 3
            params = (vector, limit) if lang is None else (vector, lang, limit)
            sql = vector_search_sql(lang)
            cursor.execute("EXPLAIN " + sql, params)
            plan = cursor.fetchall()
            diagnostics["explain"][lang or "global"] = {
                "sql": sql,
                "limit": limit,
                "plan": plan,
                "vector_index_named_in_plan": any(
                    r.get("key") == "article_semantics" for r in plan
                ),
            }
    summary = {}
    for method in ("global9", "balanced3x3"):
        rows = [r["methods"][method] for r in results]
        times = sorted(r["retrieval_ms"] for r in rows)
        summary[method] = {
            "questions": len(rows),
            "hit_at_1_topic": statistics.mean(r["hit_at_1_topic"] for r in rows),
            "hit_at_9_articles": statistics.mean(r["hit_at_9_articles"] for r in rows),
            "mean_languages_returned": statistics.mean(
                r["languages_returned"] for r in rows
            ),
            "mean_expected_edition_recall": statistics.mean(
                r["expected_edition_recall"] for r in rows
            ),
            "median_retrieval_ms": statistics.median(times),
            "p95_retrieval_ms_nearest_rank": times[math.ceil(0.95 * len(times)) - 1],
        }
    summary["wikidata_expansion_top3"] = {
        "mean_expected_edition_recall": statistics.mean(
            r["wikidata_expansion_top3"]["expected_edition_recall"] for r in results
        ),
        "median_expansion_ms": statistics.median(
            r["wikidata_expansion_top3"]["expansion_ms"] for r in results
        ),
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_changes": True,
        "dataset_release": "s-2026-10",
        "question_source": "https://github.com/MariaDB/mariadb-vector-wikipedia-rag/blob/main/data/test-questions.tsv",
        "question_file_sha256": sha(questions_file),
        "query_vector_cache_sha256": sha(vectors_file),
        "dataset_manifest_sha256": sha(ROOT / "data/s-2026-10/manifest.json"),
        "embedding_provenance": "Previously saved bge-m3 via Ollama query vectors; no new embeddings or inference.",
        "questions": len(results),
        "question_languages": dict(Counter(q["lang"] for q in questions)),
        "config": {
            "global_article_budget": 9,
            "balanced_per_language": 3,
            "expansion_topic_limit": 3,
            "balanced_topic_score": "sum(1/(60+rank_within_language)); ties by Wikidata ID",
        },
        "summary": summary,
        "diagnostics": diagnostics,
        "results": results,
        "limitations": [
            "Starter development questions, not held-out evaluation; prior settings were influenced by related questions.",
            "Single pass, no deliberate warmup or cache flush; OS/database caches may already be warm. Not a cold-start latency benchmark.",
            "Timings exclude embedding, model loading, connection setup, hydration text, EXPLAIN and generation; balanced retrieval makes three SQL calls.",
            "Global Hit@1 uses first article topic; balanced Hit@1 uses cross-language rank-fusion topic score, so these are different ranking rules.",
            "Hit@9 is presence of the expected topic in at most nine vector candidates; expansion is measured separately.",
            "Expected edition recall counts stored article identities for the expected topic, not useful passages, factual agreement, or answer faithfulness.",
            "Only dataset S was measured. Index names in EXPLAIN are diagnostics, not an all-Wikipedia scaling claim.",
        ],
    }
    output = ROOT / "reports/language-balanced-retrieval.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
