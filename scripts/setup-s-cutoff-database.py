"""Restore S into isolated wikilens_s_cutoff without changing the paused L DB.

Remote /workspace only. Uses existing ignored .env credentials and the local
MariaDB root socket solely to create/grant the isolated database. Never prints
credentials or rewrites .env, never drops a database/index, never starts compute.

  .venv/bin/python scripts/setup-s-cutoff-database.py
  DB_NAME=wikilens_s_cutoff .venv/bin/python <cutoff-runner> ...
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATABASE = "wikilens_s_cutoff"
RELEASE = "s-2026-10"
COUNTS = {"fi": 19285, "en": 5254, "sv": 4921}
ESSENTIAL_HASHES = {
    "articles-fi.parquet": "fe12b09b8b9b4c62a25b812a685d20514e1b0d85ad8c14b50e35eb6c6408f6ab",
    "articles-en.parquet": "559f08ce84767c787e42afb606e9c8220952cf2b6948487b16b404e79bad52f9",
    "articles-sv.parquet": "3a3c73c9e6969d53ad885da84c28649688fc8a91636927d613cdef0dd4652cbd",
    "vectors-bge-m3-fi.parquet": "5e108578122f8436df18c256a617ef832b10d156fe91cc30918da9c95ea2a7b6",
    "vectors-bge-m3-en.parquet": "c2a16a5b85652615b19271a3cc29df90d17e18a6c4dd6c97f379c46c1b236446",
    "vectors-bge-m3-sv.parquet": "62341a6e83e95e33fa1ccd33f561433d83640bf296ebe680754ad79a7726bec8",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory):
    import pyarrow.parquet as pq

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("release") != RELEASE:
        raise ValueError("Expected the versioned S release")
    observed = {}
    for name, entry in manifest["files"].items():
        p = directory / name
        if p.resolve().parent != directory.resolve():
            raise ValueError("Unexpected manifest path")
        actual = digest(p)
        if actual != entry["sha256"] or p.stat().st_size != entry["bytes"]:
            raise ValueError("S file checksum/size mismatch: " + name)
        observed[name] = actual
    if any(
        observed.get(name) != expected for name, expected in ESSENTIAL_HASHES.items()
    ):
        raise ValueError("Unexpected official S article/vector revision")
    for lang, count in COUNTS.items():
        for kind in ("articles", "vectors-bge-m3"):
            p = directory / f"{kind}-{lang}.parquet"
            if pq.ParquetFile(p).metadata.num_rows != count:
                raise ValueError("S row count mismatch: " + p.name)
    return digest(manifest_path), observed


def has_table(cursor, name):
    cursor.execute(
        "SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name=%s",
        (name,),
    )
    return bool(cursor.fetchone()["n"])


def count_table(cursor, name):
    if not has_table(cursor, name):
        return 0
    cursor.execute("SELECT COUNT(*) AS n FROM " + name)
    return cursor.fetchone()["n"]


def write_report(path, report):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/s-2026-10")
    parser.add_argument("--socket", default="/tmp/junction-mariadb.sock")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/s-cutoff-database.json"
    )
    args = parser.parse_args()
    if not str(ROOT).startswith("/workspace/"):
        raise RuntimeError(
            "Run this only inside the disposable SSH /workspace project; no local DB writes allowed"
        )
    started = time.perf_counter()
    manifest_sha, hashes = verify(args.data)
    print("S manifest and all article/vector checksums verified.", flush=True)
    from dotenv import dotenv_values
    import pymysql
    from pymysql.cursors import DictCursor

    cfg = dotenv_values(ROOT / ".env")
    if not cfg.get("DB_PASSWORD") or not cfg.get("DB_USER"):
        raise ValueError("Existing remote database credentials are required")
    if cfg.get("DB_HOST", "127.0.0.1") not in ("127.0.0.1", "localhost"):
        raise ValueError("Expected loopback MariaDB host")
    with (
        pymysql.connect(
            unix_socket=args.socket, user="root", cursorclass=DictCursor
        ) as admin,
        admin.cursor() as c,
    ):
        c.execute(
            "CREATE DATABASE IF NOT EXISTS `wikilens_s_cutoff` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        c.execute(
            "GRANT ALL PRIVILEGES ON `wikilens_s_cutoff`.* TO %s@%s",
            (cfg["DB_USER"], "127.0.0.1"),
        )
        admin.commit()
    # These settings affect this process only. The default .env and all other
    # existing processes keep their original DB_NAME (the paused L database).
    for key in ("DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD"):
        if cfg.get(key) is not None:
            os.environ[key] = cfg[key]
    os.environ["DB_NAME"] = DATABASE
    from wikilens.database import connect, initialize, load_dataset

    identity = {
        "dataset": RELEASE,
        "source_manifest_sha256": manifest_sha,
        "file_sha256": hashes,
        "expected_article_counts": COUNTS,
        "expected_vector_counts": COUNTS,
    }
    with connect() as db, db.cursor() as c:
        c.execute("SELECT DATABASE() AS name")
        if c.fetchone()["name"] != DATABASE:
            raise ValueError("Database isolation failed")
        existing = {
            name: count_table(c, name) for name in ("article", "article_vector")
        }
        identity_exists = has_table(c, "wikilens_corpus_identity")
        if any(existing.values()) and not identity_exists:
            raise RuntimeError(
                "Nonempty isolated database has no matching identity; inspect it manually"
            )
        c.execute("""CREATE TABLE IF NOT EXISTS wikilens_corpus_identity (
          corpus_id VARCHAR(128) NOT NULL PRIMARY KEY,
          source_manifest_sha256 CHAR(64) NOT NULL,
          file_sha256_json LONGTEXT NOT NULL,
          status VARCHAR(24) NOT NULL,
          updated_at DATETIME(6) NOT NULL
        ) ENGINE=InnoDB""")
        db.commit()
        c.execute(
            "SELECT * FROM wikilens_corpus_identity WHERE corpus_id=%s", (RELEASE,)
        )
        old = c.fetchone()
        if old:
            if (
                old["source_manifest_sha256"] != manifest_sha
                or json.loads(old["file_sha256_json"]) != hashes
            ):
                raise ValueError(
                    "Isolated database has a different S identity; no overwrite performed"
                )
            completed = old["status"] == "complete"
        else:
            if any(existing.values()) or count_table(c, "wikilens_corpus_identity"):
                raise RuntimeError("Isolated database contains another corpus identity")
            c.execute(
                "INSERT INTO wikilens_corpus_identity VALUES (%s,%s,%s,'importing',UTC_TIMESTAMP(6))",
                (RELEASE, manifest_sha, json.dumps(hashes, sort_keys=True)),
            )
            db.commit()
            completed = False
    initialize()
    # Existing bounded S loader is restartable via upsert. Completed, matching
    # databases are only verified, avoiding a second native vector-index import.
    loaded = None
    if not completed:
        loaded = load_dataset(args.data)
    with connect() as db, db.cursor() as c:
        c.execute("SELECT lang,COUNT(*) AS n FROM article GROUP BY lang")
        articles = {r["lang"]: r["n"] for r in c.fetchall()}
        c.execute("SELECT lang,COUNT(*) AS n FROM article_vector GROUP BY lang")
        vectors = {r["lang"]: r["n"] for r in c.fetchall()}
        if articles != COUNTS or vectors != COUNTS:
            raise ValueError("Isolated S database count mismatch")
        c.execute("""SELECT COUNT(*) AS n FROM article a LEFT JOIN article_vector v USING(lang,page_id)
                     WHERE v.page_id IS NULL""")
        if c.fetchone()["n"]:
            raise ValueError("Article without matching S vector")
        c.execute("SHOW CREATE TABLE article_vector")
        ddl = str(c.fetchone()).upper()
        if "VECTOR KEY" not in ddl and "VECTOR INDEX" not in ddl:
            raise ValueError("Missing native vector index")
        c.execute(
            "UPDATE wikilens_corpus_identity SET status='complete',updated_at=UTC_TIMESTAMP(6) WHERE corpus_id=%s",
            (RELEASE,),
        )
        db.commit()
    try:
        directory = str(args.data.resolve().relative_to(ROOT))
    except ValueError:
        directory = str(args.data.resolve())
    report = {
        **identity,
        "release": RELEASE,
        "status": "complete",
        "database": DATABASE,
        "dataset_directory": directory,
        "article_counts": articles,
        "vector_counts": vectors,
        "counts": [
            {"lang": lang, "articles": articles[lang], "vectors": vectors[lang]}
            for lang in sorted(COUNTS)
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "load_result": loaded,
        "reused_complete_database": completed,
        "importer_sha256": digest(Path(__file__)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_path": str(args.data / "manifest.json"),
        "policy": "Isolated S database; original .env and paused L database unchanged.",
    }
    write_report(args.report, report)
    print(
        "Isolated database ready:",
        DATABASE,
        "counts",
        json.dumps(report["counts"]),
        flush=True,
    )
    print("Report:", args.report, flush=True)
    print(
        "Use DB_NAME=wikilens_s_cutoff for the cutoff runner; .env was not changed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
