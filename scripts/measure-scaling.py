"""Isolated S-corpus index scaling experiment using cached query embeddings.

Creates only run-owned _wl_scale_* vector tables, one corpus at a time, then
removes them after a durable report checkpoint. Production article/vector tables
are read only. No embeddings, generation, training, or cloud jobs are performed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wikilens.database import connect, vector_search_sql

SEED = 42
SIZES = (1000, 10000, 29460)
WORKERS = (1, 4)
REPETITIONS = 5
TABLE_PATTERN = re.compile(r"_wl_scale_[0-9]+_[0-9]+\Z")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_identifier(name):
    if not isinstance(name, str) or not TABLE_PATTERN.fullmatch(name):
        raise ValueError("Only run-owned scaling table identifiers are allowed")
    return "`" + name + "`"


def query_sql(table):
    return vector_search_sql().replace(
        "FROM article_vector ", "FROM " + table_identifier(table) + " "
    )


def subset_keys(articles, expected_topics, sizes=SIZES, seed=SEED):
    """Nested, language-stratified real article identities with all targets kept."""
    by_lang = defaultdict(list)
    required = defaultdict(list)
    seen = set()
    for row in articles:
        key = (row["lang"], int(row["page_id"]))
        if key in seen:
            raise ValueError("Duplicate source article identity")
        seen.add(key)
        target = required if row["wikidata_id"] in expected_topics else by_lang
        target[row["lang"]].append(key)
    languages = sorted(set(by_lang) | set(required))
    counts = {lang: len(by_lang[lang]) + len(required[lang]) for lang in languages}
    total = len(seen)
    rng = random.Random(seed)
    for lang in languages:
        by_lang[lang].sort()
        rng.shuffle(by_lang[lang])
        required[lang].sort()
    result = {}
    previous = set()
    for size in sizes:
        if type(size) is not int or not len(previous) <= size <= total:
            raise ValueError("Corpus sizes must increase and fit the original corpus")
        quotas = {
            lang: max(len(required[lang]), math.floor(size * counts[lang] / total))
            for lang in languages
        }
        while sum(quotas.values()) < size:
            lang = max(
                languages,
                key=lambda lang: (size * counts[lang] / total - quotas[lang], lang),
            )
            quotas[lang] += 1
        while sum(quotas.values()) > size:
            eligible = [
                lang for lang in languages if quotas[lang] > len(required[lang])
            ]
            if not eligible:
                raise ValueError("Subset is too small to contain every target edition")
            lang = max(
                eligible,
                key=lambda lang: (quotas[lang] - size * counts[lang] / total, lang),
            )
            quotas[lang] -= 1
        keys = set()
        for lang in languages:
            keys.update(required[lang])
            keys.update(by_lang[lang][: quotas[lang] - len(required[lang])])
        if not previous <= keys:
            raise ValueError("Requested sampling sizes do not yield nested strata")
        assert len(keys) == size
        result[size] = sorted(keys)
        previous = keys
    return result


def nearest_rank(values, quantile):
    if not values or not 0 < quantile <= 1:
        raise ValueError("Supply measurements and a quantile in (0,1]")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def summarize(rows, wall_seconds):
    if not rows or wall_seconds <= 0:
        raise ValueError("Supply completed query records and positive elapsed time")
    latencies = [r["sql_ms"] for r in rows]
    return {
        "requests": len(rows),
        "median_sql_ms": statistics.median(latencies),
        "p95_sql_ms_nearest_rank": nearest_rank(latencies, 0.95),
        "throughput_queries_per_second": len(rows) / wall_seconds,
        "cohort_wall_seconds": wall_seconds,
        "hit_at_1_topic": statistics.mean(r["expected_rank"] == 1 for r in rows),
        "hit_at_9_articles": statistics.mean(
            r["expected_rank"] is not None for r in rows
        ),
    }


def memory_observation():
    result = {"observed_at": datetime.now(UTC).isoformat()}
    pid_path = ROOT / "data/mariadb/server.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            values = {}
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                    key, value, _ = line.split()
                    values[key.rstrip(":") + "_bytes"] = int(value) * 1024
            result["database_process"] = {"pid": pid, **values}
        except (OSError, ValueError):
            result["database_process"] = {"status": "unavailable"}
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value, *_ = line.split()
        if key in ("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"):
            meminfo[key.rstrip(":") + "_bytes"] = int(value) * 1024
    result["host"] = meminfo
    disk = shutil.disk_usage(ROOT / "data/mariadb")
    result["filesystem"] = {
        "used_bytes": disk.used,
        "free_bytes": disk.free,
        "total_bytes": disk.total,
    }
    return result


def write_report(path, report):
    # Replace only this run's checkpoint; no source artifacts are overwritten.
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    temp.replace(path)


def create_index(db, table, keys, owned):
    ident = table_identifier(table)
    with db.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) AS n FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s",
            (table,),
        )
        if cursor.fetchone()["n"]:
            raise ValueError("Refusing to replace an existing benchmark table")
        cursor.execute(f"""CREATE TABLE {ident} (
            lang CHAR(2) NOT NULL, page_id INT UNSIGNED NOT NULL,
            embedding VECTOR(1024) NOT NULL,
            PRIMARY KEY(lang,page_id),
            VECTOR INDEX article_semantics(embedding) M=16 DISTANCE=cosine
        ) ENGINE=InnoDB""")
        owned.add(table)
        for offset in range(0, len(keys), 250):
            batch = keys[offset : offset + 250]
            placeholders = ",".join(["(%s,%s)"] * len(batch))
            cursor.execute(
                f"INSERT INTO {ident}(lang,page_id,embedding) SELECT lang,page_id,embedding FROM article_vector WHERE (lang,page_id) IN ({placeholders}) ORDER BY lang,page_id",
                tuple(v for key in batch for v in key),
            )
            db.commit()
        cursor.execute(f"SELECT COUNT(*) AS n FROM {ident}")
        if cursor.fetchone()["n"] != len(keys):
            raise ValueError("Some selected articles lack original vectors")
        cursor.execute(f"ANALYZE TABLE {ident}")
        cursor.fetchall()
    db.commit()


def corpus_diagnostics(db, table, vector):
    with db.cursor() as cursor:
        cursor.execute("EXPLAIN " + query_sql(table), (vector, 9))
        plan = cursor.fetchall()
        cursor.execute("SHOW CREATE TABLE " + table_identifier(table))
        ddl = cursor.fetchone()
        cursor.execute(
            "SELECT ENGINE,TABLE_ROWS,DATA_LENGTH,INDEX_LENGTH,DATA_FREE FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s",
            (table,),
        )
        storage = cursor.fetchone()
    physical_files = list((ROOT / "data/mariadb").glob("*/" + table + ".ibd"))
    return {
        "physical_ibd_bytes": sum(p.stat().st_size for p in physical_files)
        if physical_files
        else None,
        "sql": query_sql(table),
        "plan": plan,
        "vector_index_named_in_plan": any(
            r.get("key") == "article_semantics" for r in plan
        ),
        "create_table": ddl,
        "storage": storage,
    }


def measure_queries(table, questions, vectors, workers, repetitions=REPETITIONS):
    local = threading.local()
    connections = []
    lock = threading.Lock()
    barrier = threading.Barrier(workers)

    def initialize():
        local.db = connect()
        with lock:
            connections.append(local.db)

    def ready():
        barrier.wait(timeout=30)

    def query(task):
        repetition, index = task
        question = questions[index]
        started = time.perf_counter()
        with local.db.cursor() as cursor:
            cursor.execute(query_sql(table), (vectors[question["question"]], 9))
            hits = cursor.fetchall()
        elapsed = (time.perf_counter() - started) * 1000
        rank = next(
            (
                rank
                for rank, hit in enumerate(hits, 1)
                if hit["wikidata_id"] == question["expected_topic"]
            ),
            None,
        )
        return {
            "question_index": index,
            "repetition": repetition,
            "sql_ms": elapsed,
            "expected_rank": rank,
            "top_title": hits[0]["title"] if hits else None,
            "returned": len(hits),
        }

    tasks = []
    for repetition in range(repetitions):
        order = list(range(len(questions)))
        random.Random(SEED + repetition).shuffle(order)
        tasks.extend((repetition, index) for index in order)
    try:
        with ThreadPoolExecutor(
            max_workers=workers, initializer=initialize
        ) as executor:
            # Establish all persistent connections before cohort timing begins.
            futures = [executor.submit(ready) for _ in range(workers)]
            for future in futures:
                future.result()
            started = time.perf_counter()
            rows = list(executor.map(query, tasks))
            wall = time.perf_counter() - started
        return {"readers": workers, "summary": summarize(rows, wall), "results": rows}
    finally:
        for connection in connections:
            connection.close()


def draw_chart(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    colors = {1: "#285c4c", 4: "#b47a35"}
    for readers in WORKERS:
        sizes, medians, tails, throughputs, hits = [], [], [], [], []
        for corpus in report["corpora"]:
            measurement = next(
                m for m in corpus["measurements"] if m["readers"] == readers
            )
            summary = measurement["summary"]
            sizes.append(corpus["articles"])
            medians.append(summary["median_sql_ms"])
            tails.append(summary["p95_sql_ms_nearest_rank"])
            throughputs.append(summary["throughput_queries_per_second"])
            hits.append(summary["hit_at_9_articles"] * 100)
        axes[0].plot(
            sizes,
            medians,
            marker="o",
            color=colors[readers],
            label=f"{readers} reader(s): median",
        )
        axes[0].plot(
            sizes,
            tails,
            marker="s",
            ls="--",
            color=colors[readers],
            label=f"{readers} reader(s): p95",
        )
        axes[1].plot(
            sizes,
            throughputs,
            marker="o",
            color=colors[readers],
            label=f"{readers} reader(s)",
        )
        axes[2].plot(
            sizes, hits, marker="o", color=colors[readers], label=f"{readers} reader(s)"
        )
    for axis, title, ylabel in zip(
        axes,
        ("SQL latency", "Retrieval throughput", "Expected-topic retrieval"),
        ("Milliseconds", "Queries / second", "Hit@9 (%)"),
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Indexed articles")
        axis.set_xscale("log")
        axis.set_xticks(list(SIZES), ["1,000", "10,000", "29,460"])
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=7, frameon=False)
    axes[2].set_ylim(0, 105)
    figure.suptitle(
        "WikiLens · isolated S index scaling", fontsize=15, fontweight="bold"
    )
    figure.text(
        0.5,
        -0.02,
        "37 cached development queries · 5 repeats · 9 candidates · warmed SQL only; no model inference",
        ha="center",
        fontsize=8,
        color="#5a6e67",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/scaling-retrieval.json"
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(
            "Refusing to overwrite an earlier scaling report; choose --output."
        )
    questions_file = ROOT / "data/evaluation/starter-questions.tsv"
    vectors_file = ROOT / "data/evaluation/question-vectors-bge-m3.json"
    with questions_file.open() as stream:
        questions = list(csv.DictReader(stream, delimiter="\t"))
    assert len(questions) == 37
    cached = json.loads(vectors_file.read_text())
    vectors = {}
    for q in questions:
        packed = base64.b64decode(cached[q["question"]], validate=True)
        if len(packed) != 4096 or not all(
            math.isfinite(v) for v in struct.unpack("<1024f", packed)
        ):
            raise ValueError("Invalid cached query vector")
        vectors[q["question"]] = packed
    memory = memory_observation()
    if memory["host"]["MemAvailable_bytes"] < 1024**3:
        raise SystemExit(
            "Less than 1 GiB available RAM; do not build a competing benchmark index."
        )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "dataset_release": "s-2026-10",
        "seed": SEED,
        "question_file_sha256": sha(questions_file),
        "query_vector_cache_sha256": sha(vectors_file),
        "dataset_manifest_sha256": sha(ROOT / "data/s-2026-10/manifest.json"),
        "script_sha256": sha(__file__),
        "database_source_sha256": sha(ROOT / "wikilens/database.py"),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_changes": True,
        "config": {
            "sizes": list(SIZES),
            "readers": list(WORKERS),
            "repetitions": REPETITIONS,
            "warmup_passes_per_corpus": 1,
            "vector_candidate_budget": 9,
            "mhnsw_ef_search": 80,
            "index_M": 16,
            "distance": "cosine",
        },
        "memory_before": memory,
        "questions": [],
        "corpora": [],
        "limitations": [
            "These are starter development queries with pre-existing embeddings, not held-out answer-quality evaluation.",
            "Every expected-topic edition is deliberately retained at each corpus size; distractors grow with seed42 language-stratified nested sampling.",
            "This is real index growth within S only; M/L and all-Wikipedia scale are not demonstrated.",
            "One warmup pass precedes timed queries. No OS/database cache flush occurs; this is a warmed workload.",
            "SQL latency excludes client queue wait, connection setup, embedding, passage hydration, model loading and generation.",
            "Throughput includes client scheduling/fetch overhead with persistent preconnected readers; no production service capacity claim.",
            "Five repeats of37 fixed questions are correlated; percentile estimates and ANN retrieval may vary between builds.",
            "Database RSS/high-water observations are process-wide and include original tables and earlier work, not isolated per-index memory.",
            "information_schema storage statistics are MariaDB-reported estimates; native-vector/index allocation can differ from resident memory.",
        ],
    }
    created_table = None
    owned = set()
    with connect() as db, db.cursor() as cursor:
        cursor.execute(
            "SELECT VERSION() AS version, @@mhnsw_ef_search AS ef_search, @@innodb_buffer_pool_size AS innodb_buffer_pool_bytes"
        )
        report["server"] = cursor.fetchone()
        cursor.execute(
            "SELECT lang,page_id,wikidata_id,title FROM article ORDER BY lang,page_id"
        )
        articles = cursor.fetchall()
        titles = {a["title"]: a["wikidata_id"] for a in articles if a["lang"] == "fi"}
        for q in questions:
            q["expected_topic"] = titles[q["expected_fi_title"]]
        report["questions"] = questions
        selected = subset_keys(articles, {q["expected_topic"] for q in questions})
        assert len(articles) == SIZES[-1]
        write_report(output, report)
        try:
            for size in SIZES:
                table = f"_wl_scale_{os.getpid()}_{size}"
                keys = selected[size]
                corpus = {
                    "articles": size,
                    "table": table,
                    "membership_sha256": hashlib.sha256(
                        json.dumps(keys, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "language_counts": dict(Counter(lang for lang, _ in keys)),
                    "expected_target_editions": sum(
                        a["wikidata_id"] in {q["expected_topic"] for q in questions}
                        for a in articles
                    ),
                    "memory_before_build": memory_observation(),
                    "measurements": [],
                }
                report["corpora"].append(corpus)
                print(
                    f"Building isolated {size:,}-article native vector index...",
                    flush=True,
                )
                started = time.perf_counter()
                # Ownership is registered before creation; create_index refuses collisions.
                cursor.execute(
                    "SELECT COUNT(*) AS n FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s",
                    (table,),
                )
                if cursor.fetchone()["n"]:
                    raise ValueError(
                        "Benchmark table collision; refusing cleanup of unowned table"
                    )
                created_table = table
                create_index(db, table, keys, owned)
                corpus["build_seconds"] = time.perf_counter() - started
                corpus["diagnostics"] = corpus_diagnostics(
                    db, table, next(iter(vectors.values()))
                )
                corpus["memory_after_build"] = memory_observation()
                print(
                    f"Warming {size:,} index once with all37 questions...", flush=True
                )
                for q in questions:
                    cursor.execute(query_sql(table), (vectors[q["question"]], 9))
                    cursor.fetchall()
                for readers in WORKERS:
                    print(
                        f"Measuring {size:,} articles, {readers} reader(s), 5 repeats...",
                        flush=True,
                    )
                    corpus["measurements"].append(
                        measure_queries(table, questions, vectors, readers)
                    )
                    write_report(output, report)
                corpus["memory_after_queries"] = memory_observation()
                write_report(output, report)
                cursor.execute("DROP TABLE " + table_identifier(created_table))
                db.commit()
                corpus["benchmark_table_removed"] = True
                owned.remove(created_table)
                created_table = None
                write_report(output, report)
            report["status"] = "completed"
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["memory_after"] = memory_observation()
            report["finished_at"] = datetime.now(UTC).isoformat()
            write_report(output, report)
            if created_table and created_table in owned:
                cursor.execute("DROP TABLE " + table_identifier(created_table))
                db.commit()
                report["corpora"][-1]["benchmark_table_removed"] = True
                write_report(output, report)
    draw_chart(report, output.with_suffix(".png"))
    print("Saved report and chart:", output, flush=True)
    for corpus in report["corpora"]:
        for measure in corpus["measurements"]:
            print(corpus["articles"], measure["readers"], measure["summary"])


if __name__ == "__main__":
    main()
