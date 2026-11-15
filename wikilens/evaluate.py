"""Paired diagnostic retrieval evaluation on the starter authors' question set."""

import base64
import csv
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .database import connect, embed_question, search_articles
from .dataset import DEFAULT_DATA, sha256

ROOT = Path(__file__).resolve().parents[1]


def evaluate():
    questions_path = ROOT / "data/evaluation/starter-questions.tsv"
    cache_path = questions_path.with_name("question-vectors-bge-m3.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with questions_path.open() as stream:
        questions = list(csv.DictReader(stream, delimiter="\t"))
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT title,wikidata_id FROM article WHERE lang='fi'")
        topics = {row["title"]: row["wikidata_id"] for row in cursor.fetchall()}
    results = []
    for i, question in enumerate(questions, 1):
        expected = topics.get(question["expected_fi_title"])
        if not expected:
            raise ValueError(
                f"Ground truth topic unavailable: {question['expected_fi_title']}"
            )
        text = question["question"]
        if text not in cache:
            cache[text] = base64.b64encode(embed_question(text)).decode("ascii")
            cache_path.write_text(
                json.dumps(cache, indent=2, ensure_ascii=False) + "\n"
            )
        vector = base64.b64decode(cache[text], validate=True)
        if len(vector) != 4096:
            raise ValueError("Invalid cached BGE-M3 vector size")
        row = {
            "language": question["lang"],
            "question": text,
            "expected_topic": expected,
            "modes": {},
        }
        for mode in ("keyword", "semantic", "hybrid"):
            started = time.perf_counter()
            hits = search_articles(text, mode=mode, limit=10, query_vector=vector)
            rank = next(
                (
                    rank
                    for rank, hit in enumerate(hits, 1)
                    if hit["wikidata_id"] == expected
                ),
                None,
            )
            row["modes"][mode] = {
                "rank": rank,
                "retrieval_ms": round((time.perf_counter() - started) * 1000, 2),
                "top_title": hits[0]["title"] if hits else None,
            }
        results.append(row)
        print(f"Evaluated {i}/{len(questions)}", flush=True)
    summary = {}
    for mode in ("keyword", "semantic", "hybrid"):
        summary[mode] = {
            "hit_at_1": sum(row["modes"][mode]["rank"] == 1 for row in results)
            / len(results),
            "hit_at_10": sum(row["modes"][mode]["rank"] is not None for row in results)
            / len(results),
            "mrr_at_10": sum(
                1 / row["modes"][mode]["rank"] if row["modes"][mode]["rank"] else 0
                for row in results
            )
            / len(results),
        }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "questions": len(results),
        "question_languages": dict(Counter(row["language"] for row in results)),
        "question_source": "https://github.com/MariaDB/mariadb-vector-wikipedia-rag/blob/main/data/test-questions.tsv",
        "question_file_sha256": sha256(questions_path),
        "dataset_manifest_sha256": sha256(DEFAULT_DATA / "manifest.json"),
        "embedding_model": "bge-m3 via Ollama",
        "summary": summary,
        "results": results,
        "limits": "Small starter-authored development set, not a held-out benchmark. Settings were adjusted after inspecting a related Winter War question. These metrics evaluate article-topic retrieval, not answer faithfulness or cross-language coverage claims. Latency excludes embedding and is a single un-warmed run.",
    }
    output = ROOT / "reports/retrieval-baseline.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluate()
