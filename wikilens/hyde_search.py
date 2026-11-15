"""Personalized HyDE → BGE-M3 → native MariaDB top-ten article retrieval."""

import json
import time

from pydantic import BaseModel, ConfigDict, Field

from .evidence import pack_vector, revision_url
from .personalization import normalize_focus


class HyDEWithLanguage(BaseModel):
    """Passage first, optional ISO 639-1 language second; neither is evidence."""

    model_config = ConfigDict(extra="forbid")
    hypothetical_passage: str = Field(
        min_length=1,
        description="Hypothetical encyclopedia passage used only for vector retrieval",
    )
    language: str | None = Field(
        pattern=r"^[a-z]{2}$",
        description="Detected question language as an ISO 639-1 code, or null if uncertain",
    )


def rerank_question(question, hyde):
    """A short Finnish retrieval hint bridges cross-language reranking, never evidence."""
    if hyde.get("detected_language") in (None, "fi"):
        return question
    first_sentence = hyde["text"].split(". ", 1)[0].split("\n", 1)[0].strip()
    hint = first_sentence[:160]
    if len(first_sentence) > 160:
        hint = hint.rsplit(" ", 1)[0]
    return question + "\nFinnish retrieval hint: " + hint if hint else question


def generate_focused_hyde(
    runtime,
    question,
    search_focus="",
    *,
    max_new_tokens=192,
    max_time=45,
    language="English",
):
    focus = normalize_focus(search_focus)
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise ValueError("Question must contain 1–2000 characters")
    from .finnish_prompts import (
        HYDE_LANGUAGE_PROMPT_FI,
        HYDE_PROMPT_FI,
        LANGUAGE_NAMES,
        answer_language_name,
    )

    automatic = language == "Auto"
    system = (
        HYDE_LANGUAGE_PROMPT_FI
        if automatic
        else HYDE_PROMPT_FI.format(language=LANGUAGE_NAMES.get(language, language))
    )
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "search_focus": focus}, ensure_ascii=False
            ),
        },
    ]
    start = time.perf_counter()
    result = {
        "question": question,
        "search_focus": focus,
        "messages": messages,
        "role": "retrieval_only_never_evidence",
        "status": "pending",
        "text": None,
        "raw_output": None,
        "config": {
            "max_new_tokens": max_new_tokens,
            "max_time": max_time,
            "enable_thinking": False,
            "language": language,
        },
    }
    try:
        options = (
            {"output_schema": HyDEWithLanguage.model_json_schema()} if automatic else {}
        )
        raw = runtime.generate(
            messages, max_new_tokens=max_new_tokens, max_time=max_time, **options
        )
        result.update(raw_output=raw, text=raw, status="completed")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Empty HyDE output")
        if automatic:
            parsed = HyDEWithLanguage.model_validate_json(raw)
            if not parsed.hypothetical_passage.strip():
                raise ValueError("Empty hypothetical passage")
            result.update(
                text=parsed.hypothetical_passage.strip(),
                detected_language=parsed.language,
                answer_language=answer_language_name(parsed.language or "fi"),
                language_fallback=parsed.language is None,
                structured_output=parsed.model_dump(),
                output_schema=HyDEWithLanguage.model_json_schema(),
            )
    except Exception as error:  # noqa: BLE001 -- preserve inference/validation failures without retry
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    result["seconds"] = time.perf_counter() - start
    return result


def retrieve_hyde(runtime, question, hyde, *, top_k=10, language=None):
    from .database import _vector_search, connect, vector_search_sql

    if hyde.get("status") != "completed" or hyde.get("question") != question:
        raise ValueError("Use completed HyDE output for the current question")
    if (
        type(top_k) is not int
        or not 1 <= top_k <= 100
        or language not in (None, "fi", "en", "sv")
    ):
        raise ValueError("top_k must be 1–100 and language None/fi/en/sv")
    started = time.perf_counter()
    vector = runtime.embed([hyde["text"]], batch_size=1, max_tokens=1024)
    if len(vector) != 1:
        raise ValueError("Expected one HyDE embedding")
    packed = pack_vector(vector[0])
    articles = []
    with connect() as db, db.cursor() as cursor:
        options = {"limit": top_k}
        if language is not None:
            options["lang"] = language
        candidates = _vector_search(cursor, packed, **options)
        cursor.execute(
            "EXPLAIN " + vector_search_sql(language),
            (packed, *([language] if language else []), top_k),
        )
        plan = cursor.fetchall()
        if candidates:
            keys = [(r["lang"], r["page_id"]) for r in candidates]
            cursor.execute(
                "SELECT lang,page_id,rev_id,wikidata_id,title,url,text FROM article WHERE (lang,page_id) IN ("
                + ",".join(["(%s,%s)"] * len(keys))
                + ")",
                tuple(v for key in keys for v in key),
            )
            full = {(r["lang"], r["page_id"]): r for r in cursor.fetchall()}
            for rank, row in enumerate(candidates, 1):
                key = (row["lang"], row["page_id"])
                if key not in full:
                    raise ValueError("Retrieved article disappeared during hydration")
                article = {
                    **full[key],
                    "distance": row["distance"],
                    "retrieval_rank": rank,
                    "origin": "hyde_vector_candidate",
                }
                article["revision_url"] = revision_url(
                    article["lang"], article["rev_id"]
                )
                articles.append(article)
    return {
        "status": "completed",
        "question": question,
        "search_focus": hyde["search_focus"],
        "articles": articles,
        "candidates": candidates,
        "query_plan": plan,
        "method": "hyde_only",
        "embedding_input": hyde["text"],
        "embedding": [float(v) for v in vector[0]],
        "seconds": time.perf_counter() - started,
        "limit": top_k,
        "language": language,
    }


def retrieve_direct(runtime, question, search_focus="", *, top_k=10, language=None):
    """Embed the question and optional reader context without model generation."""
    focus = normalize_focus(search_focus)
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise ValueError("Question must contain 1–2000 characters")
    text = question.strip() + ("\nSearch focus: " + focus if focus else "")
    result = retrieve_hyde(
        runtime,
        question,
        {
            "status": "completed",
            "question": question,
            "search_focus": focus,
            "text": text,
        },
        top_k=top_k,
        language=language,
    )
    result["method"] = "direct_query"
    for article in result["articles"]:
        article["origin"] = "direct_vector_candidate"
    return result
