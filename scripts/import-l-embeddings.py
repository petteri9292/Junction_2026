"""Bounded import of verified L articles + existing full BGE-M3 embeddings.

For a fresh disposable /workspace MariaDB only. Never overlays an S database.
Each transaction atomically inserts <=1000 articles, vectors, embedding provenance
and its resume cursor. Default100 rows; larger batches are an explicit option.
--resume continues only the exact same source/manifest.

Example on SSH (after setup-remote-mariadb.py creates empty base tables):
  .venv/bin/python scripts/import-l-embeddings.py \
    --articles data/l-2026-10/articles-fi.parquet \
    --vectors data/l-embeddings/l-full-bge-m3-20261114T123013Z

Cache configuration is intentionally NOT changed. MariaDB documents a default
16MiB mhnsw_max_cache_size (GLOBAL/dynamic, per index). For T4-small a separately
reviewed explicit 512MiB vector cache plus existing 512MiB InnoDB pool is a
reasonable bounded starting configuration; measure it rather than assuming it.
See https://mariadb.com/docs/server/reference/sql-structure/vectors/vector-system-variables
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_SHA = "05d5520fc272b0f879bdf62ea67e8bf853c55ce5975eb5393f20c78c086158ec"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EXPECTED_ROWS = 568416
ARTICLE_COLUMNS = [
    "lang",
    "page_id",
    "rev_id",
    "wikidata_id",
    "title",
    "url",
    "date_modified",
    "text",
]
VECTOR_COLUMNS = [
    "lang",
    "page_id",
    "rev_id",
    "source_row",
    "model",
    "truncated",
    "tokens_used",
]


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs(articles, directory):
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    config = report["config"]
    if report.get("status") != "complete" or report["rows"] != EXPECTED_ROWS:
        raise ValueError("Expected the completed 568416-row full-L embedding run")
    if config["source_sha256"] != SOURCE_SHA or config["revision"] != MODEL_REVISION:
        raise ValueError("Unexpected article revision or BGE-M3 model revision")
    if (
        config["release"] != "l-2026-10"
        or config["input_format"] != "title + two newlines + text"
    ):
        raise ValueError("Unexpected source release or model input format")
    if file_sha(articles) != SOURCE_SHA:
        raise ValueError("L article SHA256 mismatch")
    source = pq.ParquetFile(articles)
    if source.metadata.num_rows != EXPECTED_ROWS or not set(ARTICLE_COLUMNS) <= set(
        source.schema_arrow.names
    ):
        raise ValueError("L source row count/schema mismatch")
    row_count = 0
    shards = report["shards"]
    if len(shards) != 57:
        raise ValueError("Expected all 57 full-L vector shards")
    expected_type = pa.list_(pa.float32(), 1024)
    for index, shard in enumerate(shards):
        path = directory / shard["file"]
        if path.resolve().parent != directory.resolve() or shard["index"] != index:
            raise ValueError("Invalid vector manifest path or order")
        if path.stat().st_size != shard["bytes"] or file_sha(path) != shard["sha256"]:
            raise ValueError("Vector shard size/SHA256 mismatch: " + path.name)
        q = pq.ParquetFile(path)
        if (
            q.metadata.num_rows != shard["rows"]
            or q.schema_arrow.field("embedding").type != expected_type
        ):
            raise ValueError("Vector shard row count/schema mismatch: " + path.name)
        if not set(VECTOR_COLUMNS) <= set(q.schema_arrow.names):
            raise ValueError("Missing vector provenance columns")
        row_count += shard["rows"]
    if row_count != EXPECTED_ROWS:
        raise ValueError("Vector manifest does not cover all L rows")
    return report, file_sha(report_path)


def article_rows(path, start, batch_size):
    """Skip completed row groups, then decode at most batch_size source rows."""
    q = pq.ParquetFile(path)
    offset = 0
    first_offset = None
    groups = []
    for i in range(q.num_row_groups):
        count = q.metadata.row_group(i).num_rows
        if offset + count > start:
            groups.append(i)
            if first_offset is None:
                first_offset = offset
        offset += count
    position = offset if first_offset is None else first_offset
    for batch in q.iter_batches(
        batch_size=batch_size, columns=ARTICLE_COLUMNS, row_groups=groups
    ):
        for row in batch.to_pylist():
            if position >= start:
                yield position, row
            position += 1


def vector_batches(directory, shards, start, batch_size):
    offset = 0
    for shard in shards:
        if offset + shard["rows"] <= start:
            offset += shard["rows"]
            continue
        position = offset
        q = pq.ParquetFile(directory / shard["file"])
        for batch in q.iter_batches(batch_size=batch_size):
            count = batch.num_rows
            if position + count <= start:
                position += count
                continue
            skip = max(0, start - position)
            if skip:
                batch = batch.slice(skip)
            embedding = batch.column(batch.schema.get_field_index("embedding"))
            if embedding.null_count:
                raise ValueError("Null vector")
            # flatten() respects a sliced FixedSizeListArray's nonzero offset.
            matrix = (
                embedding.flatten()
                .to_numpy(zero_copy_only=False)
                .reshape(batch.num_rows, 1024)
            )
            if not np.isfinite(matrix).all() or not np.allclose(
                np.linalg.norm(matrix, axis=1), 1, atol=2e-5
            ):
                raise ValueError("Non-finite or non-unit vector")
            rows = batch.select(VECTOR_COLUMNS).to_pylist()
            yield position + skip, rows, matrix
            position += count
        offset += shard["rows"]


CORPUS_DDL = """CREATE TABLE IF NOT EXISTS wikilens_corpus_import (
 corpus_id VARCHAR(128) NOT NULL PRIMARY KEY,
 source_sha256 CHAR(64) NOT NULL,
 manifest_sha256 CHAR(64) NOT NULL,
 model_revision CHAR(40) NOT NULL,
 config_json LONGTEXT NOT NULL,
 expected_rows BIGINT UNSIGNED NOT NULL,
 next_source_row BIGINT UNSIGNED NOT NULL DEFAULT 0,
 status VARCHAR(24) NOT NULL,
 updated_at DATETIME(6) NOT NULL
) ENGINE=InnoDB"""
PROVENANCE_DDL = """CREATE TABLE IF NOT EXISTS article_embedding_provenance (
 lang CHAR(2) NOT NULL,
 page_id INT UNSIGNED NOT NULL,
 rev_id BIGINT UNSIGNED NOT NULL,
 corpus_id VARCHAR(128) NOT NULL,
 source_row BIGINT UNSIGNED NOT NULL,
 tokens_used INT UNSIGNED NOT NULL,
 truncated BOOLEAN NOT NULL,
 PRIMARY KEY(lang,page_id),
 UNIQUE KEY corpus_position(corpus_id,source_row)
) ENGINE=InnoDB"""
ARTICLE_SQL = """INSERT INTO article
 (lang,page_id,rev_id,wikidata_id,title,url,date_modified,text,characters,sections,external_links)
 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
VECTOR_SQL = """INSERT INTO article_vector (lang,page_id,model,truncated,embedding)
 VALUES (%s,%s,%s,%s,%s)"""
PROVENANCE_SQL = """INSERT INTO article_embedding_provenance
 (lang,page_id,rev_id,corpus_id,source_row,tokens_used,truncated) VALUES (%s,%s,%s,%s,%s,%s,%s)"""


def table_count(cursor, name):
    # Only fixed internal table names are passed here.
    cursor.execute("SELECT COUNT(*) AS n FROM " + name)
    return cursor.fetchone()["n"]


def write_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def import_corpus(args, manifest, manifest_sha):
    from wikilens.database import connect
    from wikilens.evidence import coverage_signals

    corpus_id = manifest["run_id"]
    started = time.perf_counter()
    summary = {
        "corpus_id": corpus_id,
        "source_sha256": SOURCE_SHA,
        "manifest_sha256": manifest_sha,
        "importer_sha256": file_sha(Path(__file__)),
        "python_version": sys.version,
        "embedding_config": manifest["config"],
        "model_revision": MODEL_REVISION,
        "expected_rows": EXPECTED_ROWS,
        "batch_size": args.batch_size,
        "invocation_started_at": datetime.now(timezone.utc).isoformat(),
        "cache_policy": "Observed only; importer changes no global cache settings.",
    }
    with connect() as db, db.cursor() as c:
        c.execute("SELECT GET_LOCK('wikilens-full-l-import',5) AS acquired")
        if c.fetchone()["acquired"] != 1:
            raise RuntimeError("Another full-L importer owns the database lock")
        try:
            counts = {t: table_count(c, t) for t in ["article", "article_vector"]}
            c.execute(
                "SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name='wikilens_corpus_import'"
            )
            state_exists = c.fetchone()["n"] == 1
            if any(counts.values()) and not state_exists:
                raise RuntimeError(
                    "Refusing to overlay existing articles/vectors. Use a fresh remote database; S is preserved."
                )
            c.execute(CORPUS_DDL)
            c.execute(PROVENANCE_DDL)
            db.commit()
            c.execute(
                "SELECT * FROM wikilens_corpus_import WHERE corpus_id=%s", (corpus_id,)
            )
            state = c.fetchone()
            if state is None:
                if (
                    any(counts.values())
                    or table_count(c, "article_embedding_provenance")
                    or table_count(c, "wikilens_corpus_import")
                ):
                    raise RuntimeError(
                        "Database is not empty and has no matching corpus checkpoint"
                    )
                c.execute(
                    """INSERT INTO wikilens_corpus_import
                 (corpus_id,source_sha256,manifest_sha256,model_revision,config_json,expected_rows,next_source_row,status,updated_at)
                 VALUES (%s,%s,%s,%s,%s,%s,0,'importing',UTC_TIMESTAMP(6))""",
                    (
                        corpus_id,
                        SOURCE_SHA,
                        manifest_sha,
                        MODEL_REVISION,
                        json.dumps(manifest["config"], sort_keys=True),
                        EXPECTED_ROWS,
                    ),
                )
                db.commit()
                next_row = 0
            else:
                if (
                    state["source_sha256"],
                    state["manifest_sha256"],
                    state["model_revision"],
                    state["expected_rows"],
                ) != (SOURCE_SHA, manifest_sha, MODEL_REVISION, EXPECTED_ROWS):
                    raise ValueError(
                        "Checkpoint source/manifest/model differs; cannot resume"
                    )
                next_row = state["next_source_row"]
                if (
                    any(n != next_row for n in counts.values())
                    or table_count(c, "article_embedding_provenance") != next_row
                ):
                    raise ValueError(
                        "Database row counts disagree with its atomic checkpoint; manual inspection required"
                    )
                if next_row != EXPECTED_ROWS and not args.resume:
                    raise ValueError(
                        "An incomplete matching checkpoint exists. Rerun with --resume explicitly."
                    )
            c.execute(
                "SHOW GLOBAL VARIABLES WHERE Variable_name IN ('mhnsw_max_cache_size','innodb_buffer_pool_size','max_allowed_packet')"
            )
            summary["observed_database_settings"] = c.fetchall()
            summary["started_at_source_row"] = next_row
            # Keep batch-tuning provenance across resumes without changing the
            # embedding configuration or the atomic database checkpoint schema.
            history = []
            if args.report.exists():
                previous = json.loads(args.report.read_text())
                if (
                    previous.get("corpus_id"),
                    previous.get("source_sha256"),
                    previous.get("manifest_sha256"),
                ) != (corpus_id, SOURCE_SHA, manifest_sha):
                    raise ValueError(
                        "Existing metrics report belongs to different inputs; choose another --report path"
                    )
                history = previous.get("batch_history", [])
                if not history and "batch_size" in previous:
                    history.append(
                        {
                            k: previous.get(k)
                            for k in [
                                "batch_size",
                                "importer_sha256",
                                "started_at_source_row",
                                "committed_rows",
                                "elapsed_seconds",
                                "invocation_started_at",
                            ]
                        }
                    )
                if history:
                    history[-1]["ended_at_source_row"] = next_row
            history.append(
                {
                    "batch_size": args.batch_size,
                    "importer_sha256": summary["importer_sha256"],
                    "started_at_source_row": next_row,
                    "invocation_started_at": summary["invocation_started_at"],
                    "committed_rows": next_row,
                }
            )
            summary.update(
                batch_history=history, committed_rows=next_row, status="importing"
            )
            write_report(args.report, summary)
            print(
                "Database settings (unchanged):",
                summary["observed_database_settings"],
                flush=True,
            )
            source = iter(article_rows(args.articles, next_row, args.batch_size))
            committed = next_row
            last_reported = next_row
            for position, vectors, matrix in vector_batches(
                args.vectors, manifest["shards"], next_row, args.batch_size
            ):
                if position != committed:
                    raise ValueError("Vector stream is not contiguous at resume cursor")
                articles, values, provenance = [], [], []
                for i, vector in enumerate(vectors):
                    try:
                        source_row, article = next(source)
                    except StopIteration as error:
                        raise ValueError(
                            "Article stream ended before its vectors"
                        ) from error
                    if source_row != position + i or vector["source_row"] != source_row:
                        raise ValueError("Article/vector source_row misalignment")
                    if (
                        any(
                            article[k] != vector[k]
                            for k in ["lang", "page_id", "rev_id"]
                        )
                        or article["lang"] != "fi"
                    ):
                        raise ValueError(
                            "Article/vector page, revision or language mismatch"
                        )
                    if (
                        vector["model"] != "bge-m3"
                        or not 1 <= vector["tokens_used"] <= 8192
                    ):
                        raise ValueError("Unexpected vector model or token count")
                    text = article["text"]
                    signals = coverage_signals(text)
                    modified = article["date_modified"]
                    if isinstance(modified, str):
                        modified = datetime.fromisoformat(modified)
                    if modified is not None and modified.tzinfo is not None:
                        modified = modified.astimezone(timezone.utc).replace(
                            tzinfo=None
                        )
                    articles.append(
                        (
                            article["lang"],
                            article["page_id"],
                            article["rev_id"],
                            article["wikidata_id"],
                            article["title"],
                            article["url"],
                            modified,
                            text,
                            signals["characters"],
                            signals["sections"],
                            signals["external_links"],
                        )
                    )
                    # Same little-endian float32 encoding as evidence.pack_vector,
                    # without constructing a 1024-element Python-float list.
                    binary = matrix[i].astype("<f4", copy=False).tobytes()
                    values.append(
                        (
                            vector["lang"],
                            vector["page_id"],
                            vector["model"],
                            vector["truncated"],
                            binary,
                        )
                    )
                    provenance.append(
                        (
                            vector["lang"],
                            vector["page_id"],
                            vector["rev_id"],
                            corpus_id,
                            source_row,
                            vector["tokens_used"],
                            vector["truncated"],
                        )
                    )
                end = position + len(vectors)
                try:
                    c.executemany(ARTICLE_SQL, articles)
                    c.executemany(VECTOR_SQL, values)
                    c.executemany(PROVENANCE_SQL, provenance)
                    c.execute(
                        "UPDATE wikilens_corpus_import SET next_source_row=%s,status=%s,updated_at=UTC_TIMESTAMP(6) WHERE corpus_id=%s",
                        (
                            end,
                            "complete" if end == EXPECTED_ROWS else "importing",
                            corpus_id,
                        ),
                    )
                    db.commit()
                except Exception:
                    db.rollback()
                    raise
                committed = end
                if committed - last_reported >= 1000 or committed == EXPECTED_ROWS:
                    summary["batch_history"][-1].update(
                        committed_rows=committed,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                    summary.update(
                        committed_rows=committed,
                        elapsed_seconds=time.perf_counter() - started,
                        status="importing",
                    )
                    write_report(args.report, summary)
                    last_reported = committed
                    print(
                        f"{committed}/{EXPECTED_ROWS} rows committed; {summary['elapsed_seconds']:.1f}s",
                        flush=True,
                    )
            if committed != EXPECTED_ROWS or next(source, None) is not None:
                raise ValueError("Final article/vector source coverage mismatch")
            final_counts = {
                t: table_count(c, t)
                for t in ["article", "article_vector", "article_embedding_provenance"]
            }
            if any(n != EXPECTED_ROWS for n in final_counts.values()):
                raise ValueError("Final database count mismatch")
            c.execute(
                """SELECT COUNT(*) AS n FROM article_embedding_provenance p
                         JOIN article a USING(lang,page_id) WHERE p.corpus_id=%s AND p.rev_id<>a.rev_id""",
                (corpus_id,),
            )
            if c.fetchone()["n"]:
                raise ValueError("Final article/embedding revision mismatch")
            c.execute("""SELECT table_name,table_rows,data_length,index_length
                         FROM information_schema.tables WHERE table_schema=DATABASE()
                         AND table_name IN ('article','article_vector','article_embedding_provenance')""")
            summary.update(
                status="complete",
                committed_rows=committed,
                counts=final_counts,
                elapsed_seconds=time.perf_counter() - started,
                table_sizes=c.fetchall(),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            write_report(args.report, summary)
            print("COMPLETE:", json.dumps(final_counts), flush=True)
        finally:
            try:
                c.execute("SELECT RELEASE_LOCK('wikilens-full-l-import')")
            except Exception:
                # Closing the connection also releases its advisory lock.
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify all input hashes/schema without connecting to a database",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/l-corpus-import.json"
    )
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1000:
        raise ValueError("Import batch size must be between 1 and 1000")
    if not args.verify_only and not str(ROOT).startswith("/workspace/"):
        raise RuntimeError(
            "Refusing a local database import. Run this only in the disposable SSH /workspace project."
        )
    started = time.perf_counter()
    manifest, manifest_sha = verify_inputs(args.articles, args.vectors)
    print(
        f"All input SHA256/schema/row checks passed in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if args.verify_only:
        return
    import_corpus(args, manifest, manifest_sha)


if __name__ == "__main__":
    main()
