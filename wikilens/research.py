"""Language-balanced native-vector retrieval with explicit edition provenance."""

import base64
import json
import math
import struct
import re
import time
from collections import defaultdict
from pathlib import Path

from .database import _vector_search, connect, vector_search_sql
from .dataset import LANGUAGES
from .evidence import fuse_rankings, revision_url

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/evaluation/question-vectors-bge-m3.json"


def cached_vector(question):
    """Reuse measured baseline query embeddings; never silently call cloud compute."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    value = cache.get(question)
    if value is None:
        raise ValueError(
            "This saved-query explorer supports the listed example questions. Use the pipeline notebook for a new question."
        )
    vector = base64.b64decode(value, validate=True)
    if len(vector) != 4096 or not all(
        math.isfinite(x) for x in struct.unpack("<1024f", vector)
    ):
        raise ValueError("Invalid cached 1024-dimensional query vector")
    return vector


def retrieve_multilingual(question, vectors, per_language=3, topic_limit=3):
    """Retrieve separately by language; align editions by identity, not similarity.

    vectors maps retrieval branch names (e.g. direct/hyde) to packed vectors.
    Branch ranks are fused within each language; distances across branches are
    never interpreted as comparable relevance scores.
    """
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise ValueError("Enter a question of 1–2000 characters")
    if not 1 <= per_language <= 10 or not 1 <= topic_limit <= 10:
        raise ValueError("Retrieval limits must be between 1 and 10")
    if not vectors or len(vectors) > 2:
        raise ValueError("Provide one or two named query vectors")
    if any(not isinstance(v, bytes) or len(v) != 4096 for v in vectors.values()):
        raise ValueError("Expected packed 1024-dimensional vectors")
    if any(
        not all(math.isfinite(x) for x in struct.unpack("<1024f", v))
        for v in vectors.values()
    ):
        raise ValueError("Expected finite query vector values")
    started = time.perf_counter()
    rankings, candidates, query_plans = {}, [], {}
    with connect() as db, db.cursor() as cursor:
        for lang in LANGUAGES:
            rankings[lang] = {}
            for name, vector in vectors.items():
                rows = _vector_search(cursor, vector, lang=lang, limit=per_language)
                rankings[lang][name] = rows
                cursor.execute(
                    "EXPLAIN " + vector_search_sql(lang), (vector, lang, per_language)
                )
                query_plans.setdefault(lang, {})[name] = cursor.fetchall()
            merged = fuse_rankings(*rankings[lang].values(), limit=per_language)
            for rank, row in enumerate(merged, 1):
                candidates.append({**row, "language_rank": rank})
        topic_scores = defaultdict(float)
        for row in candidates:
            if row["wikidata_id"]:
                topic_scores[row["wikidata_id"]] += 1 / (60 + row["language_rank"])
        topics = sorted(topic_scores, key=lambda q: (-topic_scores[q], q))[:topic_limit]
        keys = {(r["lang"], r["page_id"]) for r in candidates}
        predicates, params = [], []
        if keys:
            ordered = sorted(keys)
            predicates.append(
                "(a.lang,a.page_id) IN (" + ",".join(["(%s,%s)"] * len(ordered)) + ")"
            )
            params.extend(v for key in ordered for v in key)
        if topics:
            predicates.append(
                "a.wikidata_id IN (" + ",".join(["%s"] * len(topics)) + ")"
            )
            params.extend(topics)
        articles = []
        if predicates:
            cursor.execute(
                "SELECT a.lang,a.page_id,a.rev_id,a.wikidata_id,a.title,a.url,a.text,v.truncated FROM article a LEFT JOIN article_vector v USING(lang,page_id) WHERE "
                + " OR ".join(predicates)
                + " ORDER BY a.wikidata_id,a.lang,a.page_id",
                tuple(params),
            )
            articles = cursor.fetchall()
        for article in articles:
            article["origin"] = (
                "vector_candidate"
                if (article["lang"], article["page_id"]) in keys
                else "wikidata_counterpart"
            )
            article["revision_url"] = revision_url(article["lang"], article["rev_id"])
    return {
        "question": question,
        "dataset_release": "s-2026-10",
        "languages": list(LANGUAGES),
        "per_language": per_language,
        "retrieval_rankings": rankings,
        "candidates": candidates,
        "aligned_topics": topics,
        "articles": articles,
        "explain_vector_queries": query_plans,
        "retrieval_seconds": time.perf_counter() - started,
        "limitation": "Language-balanced candidates and Wikidata counterparts are source discovery, not proof of relevance, agreement, or claim support.",
    }


def edition_dossier(result, wikidata_id=None):
    """Display exact article introductions; no translation or invented comparison."""
    topic = wikidata_id or next(iter(result["aligned_topics"]), None)
    if topic is not None and not re.fullmatch(r"Q[1-9][0-9]*", topic):
        raise ValueError("Invalid Wikidata ID")
    if topic is not None and topic not in result["aligned_topics"]:
        raise ValueError(
            "Topic editions have not been fully loaded; select an aligned topic"
        )
    articles = (
        [a for a in result["articles"] if a["wikidata_id"] == topic] if topic else []
    )
    editions = []
    for lang in LANGUAGES:
        rows = [a for a in articles if a["lang"] == lang]
        editions.append(
            {
                "lang": lang,
                "status": "present" if rows else "absent_from_dataset",
                "sources": [
                    {
                        k: a[k]
                        for k in (
                            "lang",
                            "page_id",
                            "rev_id",
                            "wikidata_id",
                            "title",
                            "origin",
                            "revision_url",
                            "truncated",
                        )
                    }
                    | {
                        "text": a["text"][:1800],
                        "start_char": 0,
                        "end_char": min(1800, len(a["text"])),
                        "selection": "article_introduction",
                    }
                    for a in rows
                ],
            }
        )
    return {
        "wikidata_id": topic,
        "editions": editions,
        "limitation": "Introductions shown verbatim. Presence is not claim support; absence is only absence from dataset S.",
    }
