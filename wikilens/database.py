"""MariaDB stores articles, native vectors and the evidence comparison signals."""

import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor

from .dataset import DEFAULT_DATA, LANGUAGES
from .evidence import (
    coverage_signals,
    excerpt,
    fuse_rankings,
    pack_vector,
    revision_url,
)
from .model_client import model_request

ROOT = Path(__file__).resolve().parents[1]


def connect():
    load_dotenv(ROOT / ".env")
    password = os.environ.get("DB_PASSWORD")
    if not password:
        raise ValueError("Set DB_PASSWORD in .env; see docs/mariadb-prototype.md")
    connection = pymysql.connect(
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("DB_PORT", "3307")),
        user=os.environ.get("DB_USER", "wikilens"),
        password=password,
        database=os.environ.get("DB_NAME", "wikilens"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        connect_timeout=10,
        read_timeout=120,
    )
    with connection.cursor() as cursor:
        cursor.execute("SET SESSION mhnsw_ef_search = 80")
    return connection


def initialize():
    with connect() as connection, connection.cursor() as cursor:
        for statement in (Path(__file__).with_name("schema.sql").read_text()).split(
            ";"
        ):
            if statement.strip():
                cursor.execute(statement)
        connection.commit()


def load_dataset(directory=DEFAULT_DATA):
    """Bounded Arrow batches; upserts make interrupted imports safe to rerun."""
    directory = Path(directory)
    started = time.perf_counter()
    with connect() as connection, connection.cursor() as cursor:
        for lang in LANGUAGES:
            article_file = pq.ParquetFile(directory / f"articles-{lang}.parquet")
            for batch in article_file.iter_batches(batch_size=100):
                rows = []
                for row in batch.to_pylist():
                    signals = coverage_signals(row["text"])
                    modified = row["date_modified"]
                    if isinstance(modified, str):
                        modified = datetime.fromisoformat(modified)
                    if modified is not None:
                        if modified.tzinfo is not None:
                            modified = modified.astimezone(UTC)
                        modified = modified.replace(tzinfo=None)
                    rows.append(
                        (
                            row["lang"],
                            row["page_id"],
                            row["rev_id"],
                            row["wikidata_id"],
                            row["title"],
                            row["url"],
                            modified,
                            row["text"],
                            signals["characters"],
                            signals["sections"],
                            signals["external_links"],
                        )
                    )
                cursor.executemany(
                    """INSERT INTO article
                    (lang,page_id,rev_id,wikidata_id,title,url,date_modified,text,
                     characters,sections,external_links)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE rev_id=VALUES(rev_id),
                    wikidata_id=VALUES(wikidata_id),title=VALUES(title),url=VALUES(url),
                    date_modified=VALUES(date_modified),text=VALUES(text),
                    characters=VALUES(characters),sections=VALUES(sections),
                    external_links=VALUES(external_links)""",
                    rows,
                )
                connection.commit()
            print(f"Loaded {lang} articles", flush=True)
            vector_file = pq.ParquetFile(directory / f"vectors-bge-m3-{lang}.parquet")
            for batch in vector_file.iter_batches(batch_size=100):
                rows = [
                    (
                        r["lang"],
                        r["page_id"],
                        r["model"],
                        r["truncated"],
                        pack_vector(r["embedding"]),
                    )
                    for r in batch.to_pylist()
                ]
                cursor.executemany(
                    """INSERT INTO article_vector (lang,page_id,model,truncated,embedding)
                    VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE
                    model=VALUES(model),truncated=VALUES(truncated),embedding=VALUES(embedding)""",
                    rows,
                )
                connection.commit()
            print(f"Loaded {lang} native vectors", flush=True)
        cursor.execute("""SELECT a.lang,COUNT(*) AS articles,
                       COUNT(v.page_id) AS vectors FROM article a
                       LEFT JOIN article_vector v USING (lang,page_id) GROUP BY a.lang""")
        counts = cursor.fetchall()
    return {
        "languages": counts,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def embed_question(question):
    """Pretrained inference only; the model must match the supplied vectors."""
    response = model_request(
        "/api/embed",
        {
            "model": "bge-m3",
            "input": [question],
            "truncate": False,
            "options": {"num_ctx": 8192, "num_batch": 8192, "num_thread": 2},
        },
    )
    return pack_vector(response["embeddings"][0])


def vector_search_sql(lang=None):
    """Single query shape shared by retrieval and its EXPLAIN diagnostics."""
    where = "WHERE lang = %s" if lang else ""
    return f"""SELECT a.lang,a.page_id,a.wikidata_id,a.title,a.url,v.distance,
                   a.characters,a.sections,a.external_links
        FROM (SELECT lang,page_id,VEC_DISTANCE_COSINE(embedding,%s) AS distance
              FROM article_vector {where} ORDER BY distance LIMIT %s) v
        JOIN article a USING (lang,page_id) ORDER BY v.distance,a.lang,a.page_id"""


def _vector_search(cursor, vector, lang=None, limit=12):
    cursor.execute(vector_search_sql(lang), (vector, *([lang] if lang else []), limit))
    return cursor.fetchall()


def search_articles(query, mode="keyword", lang=None, limit=12, query_vector=None):
    query = query.strip()
    if not query or len(query) > 2000:
        raise ValueError("Enter a query of 1–2000 characters")
    if lang is not None and lang not in LANGUAGES:
        raise ValueError("Unknown language")
    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100")
    with connect() as connection, connection.cursor() as cursor:
        vector_rows = []
        if mode in ("semantic", "hybrid"):
            vector_rows = _vector_search(
                cursor,
                query_vector if query_vector is not None else embed_question(query),
                lang=lang,
                limit=max(limit, 30) if mode == "hybrid" else limit,
            )
            if mode == "semantic":
                return vector_rows
        if mode not in ("keyword", "hybrid"):
            raise ValueError("Unknown search mode")
        language_sql = "AND lang = %s" if lang else ""
        cursor.execute(
            f"""SELECT lang,page_id,wikidata_id,title,url,characters,sections,
                       external_links,MATCH(title,text) AGAINST (%s) AS score
            FROM article WHERE MATCH(title,text) AGAINST (%s) {language_sql}
            ORDER BY score DESC,lang,page_id LIMIT %s""",
            (
                query,
                query,
                *([lang] if lang else []),
                max(limit, 30) if mode == "hybrid" else limit,
            ),
        )
        keyword_rows = cursor.fetchall()
        return (
            fuse_rankings(vector_rows, keyword_rows, limit=limit)
            if mode == "hybrid"
            else keyword_rows
        )


def related_articles(lang, page_id, limit=12):
    if lang not in LANGUAGES or not 1 <= limit <= 100:
        raise ValueError("Invalid language or limit")
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT embedding FROM article_vector WHERE lang=%s AND page_id=%s",
            (lang, int(page_id)),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError("Article vector not found")
        # Each language gets candidates so one language cannot dominate the comparison.
        result = []
        for target in LANGUAGES:
            rows = _vector_search(cursor, row["embedding"], target, limit + 1)
            result.extend(
                [r for r in rows if (r["lang"], r["page_id"]) != (lang, int(page_id))][
                    :limit
                ]
            )
        return result


def compare_topic(wikidata_id):
    if not re.fullmatch(r"Q[1-9][0-9]*", wikidata_id):
        raise ValueError("Use a Wikidata topic ID such as Q33")
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT a.*,v.truncated FROM article a
            LEFT JOIN article_vector v USING (lang,page_id)
            WHERE wikidata_id=%s ORDER BY lang,page_id""",
            (wikidata_id,),
        )
        articles = cursor.fetchall()
        for article in articles:
            text = article.pop("text")
            signals = coverage_signals(text)
            article["excerpt"] = excerpt(text)
            article["headings"] = signals["headings"]
            article["links"] = signals["links"]
            article["revision_url"] = revision_url(article["lang"], article["rev_id"])
        cursor.execute(
            """SELECT x.lang AS language_a,y.lang AS language_b,
            VEC_DISTANCE_COSINE(x.embedding,y.embedding) AS distance
            FROM article_vector x JOIN article a ON a.lang=x.lang AND a.page_id=x.page_id
            JOIN article b ON b.wikidata_id=a.wikidata_id AND b.lang>a.lang
            JOIN article_vector y ON y.lang=b.lang AND y.page_id=b.page_id
            WHERE a.wikidata_id=%s ORDER BY x.lang,y.lang""",
            (wikidata_id,),
        )
        distances = cursor.fetchall()
    present = {article["lang"] for article in articles}
    return {
        "wikidata_id": wikidata_id,
        "articles": articles,
        "missing_from_dataset": sorted(set(LANGUAGES) - present),
        "semantic_distances": distances,
        "caveat": "Coverage signals and embedding distance do not establish bias, accuracy, or missing facts. Missing means absent from this dataset snapshot, not absent from Wikipedia.",
    }
