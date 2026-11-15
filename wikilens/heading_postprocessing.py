"""Heading-aware literal source chunks ranked with BGE-M3 dense similarity."""

import re
import time
from .evidence import revision_url
from .personalization import normalize_focus

_HEADING = re.compile(
    r"^(?:[ \t]{0,3}(#{1,6})[ \t]+(.+?)\s*|[ \t]*(={2,6})[ \t]*(.+?)[ \t]*\3[ \t]*)$",
    re.M,
)

_REFERENCE_HEADINGS = {
    "references",
    "general and cited references",
    "references and notes",
    "notes and references",
    "references and sources",
    "sources",
    "bibliography",
    "external links",
    "further reading",
    "see also",
    "notes",
    "footnotes",
    "citations",
    "lähteet",
    "kirjallisuus",
    "viitteet",
    "aiheesta muualla",
    "katso myös",
    "referenser",
    "källor",
    "litteratur",
    "externa länkar",
    "se även",
    "noter",
}


def is_reference_section(section):
    return any(
        title.strip().casefold() in _REFERENCE_HEADINGS
        for title in section["heading_path"]
    )


def heading_sections(article):
    """Separate leaf text ranges at every heading; retain ancestor breadcrumbs.

    The range includes its own heading. Parent introductory text is a separate
    section, rather than duplicating full parent bodies in each subsection.
    """
    text = article["text"]
    headings = list(_HEADING.finditer(text))
    boundaries = [(m.start(), m) for m in headings]
    if not boundaries or boundaries[0][0] != 0:
        boundaries.insert(0, (0, None))
    stack = []
    for i, (start, match) in enumerate(boundaries):
        if match:
            level = len(match[1] or match[3])
            title = (match[2] or match[4]).strip().rstrip("#").strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        # Skip headings with no body; the title remains in descendants' paths.
        body_start = match.end() if match else start
        if text[body_start:end].strip():
            yield {
                "section_start_char": start,
                "section_end_char": end,
                "heading_path": [title for _, title in stack],
            }


def heading_chunks(
    articles,
    tokenizer,
    *,
    chunk_tokens=384,
    overlap_tokens=48,
    max_tokens=512,
    skip_reference_sections=True,
):
    """Subdivide oversized sections at token offsets, keeping exact article spans."""
    if not 0 <= overlap_tokens < chunk_tokens < max_tokens:
        raise ValueError("Require 0 <= overlap_tokens < chunk_tokens < max_tokens")
    chunks, seen = [], set()
    for rank, article in enumerate(articles, 1):
        key = (article["lang"], article["page_id"], article["rev_id"])
        if key in seen:
            continue
        seen.add(key)
        for section in heading_sections(article):
            if skip_reference_sections and is_reference_section(section):
                continue
            start, end = section["section_start_char"], section["section_end_char"]
            text = article["text"][start:end]
            prefix = (
                article["title"] + "\n" + " > ".join(section["heading_path"]) + "\n\n"
            )
            prefix_offsets = tokenizer(
                prefix, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            if len(prefix_offsets) > 80:
                prefix = prefix[: prefix_offsets[79][1]] + "\n\n"
            offsets = tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            i = 0
            while i < len(offsets):
                j = min(i + chunk_tokens, len(offsets))
                lo = 0 if i == 0 else offsets[i][0]
                hi = len(text) if j == len(offsets) else offsets[j - 1][1]
                while (
                    len(tokenizer(prefix + text[lo:hi])["input_ids"]) > max_tokens
                    and j > i + 1
                ):
                    j -= 1
                    hi = offsets[j - 1][1]
                literal = text[lo:hi]
                if literal.strip():
                    a, b = start + lo, start + hi
                    chunks.append(
                        {
                            **{
                                k: article[k]
                                for k in (
                                    "lang",
                                    "page_id",
                                    "rev_id",
                                    "title",
                                    "wikidata_id",
                                )
                                if k in article
                            },
                            **section,
                            "id": f"{key[0]}-{key[1]}-{key[2]}-{a}-{b}",
                            "text": literal,
                            "start_char": a,
                            "end_char": b,
                            "retrieval_rank": article.get("retrieval_rank", rank),
                            "revision_url": revision_url(
                                article["lang"], article["rev_id"]
                            ),
                            "embedding_text": prefix + literal,
                        }
                    )
                if j == len(offsets):
                    break
                i = max(i + 1, j - overlap_tokens)
    return chunks


def bge_postprocess(
    runtime,
    question,
    articles,
    *,
    search_focus="",
    max_nodes=10,
    context_chars=7000,
    max_per_article=3,
    chunk_tokens=384,
    overlap_tokens=48,
    max_tokens=512,
    skip_reference_sections=True,
    progress=None,
):
    """Embed the focused question and every heading chunk, then rank by cosine.

    BGE outputs normalized vectors. This is bi-encoder reranking, not a dedicated
    BGE cross-encoder. Similarity is not a probability or a support judgment.
    """
    started = time.perf_counter()
    from transformers import AutoTokenizer
    from .torch_runtime import BGE_MODEL, BGE_REVISION

    focus = normalize_focus(search_focus)
    tokenizer = AutoTokenizer.from_pretrained(
        BGE_MODEL, revision=BGE_REVISION, use_fast=True
    )
    chunks = heading_chunks(
        articles,
        tokenizer,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        max_tokens=max_tokens,
        skip_reference_sections=skip_reference_sections,
    )
    excluded_sections = (
        sum(
            is_reference_section(section)
            for article in articles
            for section in heading_sections(article)
        )
        if skip_reference_sections
        else 0
    )
    chunk_seconds = time.perf_counter() - started
    query = question + ("\nSearch focus: " + focus if focus else "")
    if progress:
        progress(f"{len(chunks)} heading chunks; embedding question and all chunks")
    embeddings_started = time.perf_counter()
    query_vector = runtime.embed([query], batch_size=1, max_tokens=max_tokens)[0]
    vectors = (
        runtime.embed(
            [c["embedding_text"] for c in chunks], batch_size=2, max_tokens=max_tokens
        )
        if chunks
        else []
    )
    embedding_seconds = time.perf_counter() - embeddings_started
    for chunk, vector in zip(chunks, vectors):
        chunk["bge_similarity"] = sum(a * b for a, b in zip(query_vector, vector))
    ranked = sorted(
        chunks,
        key=lambda c: (-c["bge_similarity"], c["retrieval_rank"], c["start_char"]),
    )
    nodes, counts, used = [], {}, 0
    for chunk in ranked:
        key = (chunk["lang"], chunk["page_id"], chunk["rev_id"])
        if counts.get(key, 0) >= max_per_article:
            continue
        if any(
            (s["lang"], s["page_id"], s["rev_id"]) == key
            and max(
                0,
                min(s["end_char"], chunk["end_char"])
                - max(s["start_char"], chunk["start_char"]),
            )
            / min(len(s["text"]), len(chunk["text"]))
            > 0.5
            for s in nodes
        ):
            continue
        # Whole literal chunks only: do not cut a section just to fill the budget.
        if used + len(chunk["text"]) > context_chars:
            continue
        nodes.append({**chunk, "node_id": chunk["id"], "id": f"S{len(nodes) + 1}"})
        counts[key] = counts.get(key, 0) + 1
        used += len(chunk["text"])
        if len(nodes) >= max_nodes:
            break
    return {
        "status": "completed" if nodes else "insufficient_evidence",
        "method": "heading_chunks_bge_m3_cosine",
        "question": question,
        "search_focus": focus,
        "nodes": nodes,
        "ranked": ranked,
        "candidate_count": len(chunks),
        "excluded_reference_sections": excluded_sections,
        "context_chars": used,
        "seconds": time.perf_counter() - started,
        "chunk_seconds": chunk_seconds,
        "embedding_seconds": embedding_seconds,
        "config": {
            "chunk_tokens": chunk_tokens,
            "overlap_tokens": overlap_tokens,
            "max_tokens": max_tokens,
            "max_nodes": max_nodes,
            "context_chars": context_chars,
            "max_per_article": max_per_article,
            "query_embedding_input": query,
            "all_chunks_scored": True,
            "skip_reference_sections": skip_reference_sections,
        },
        "limitation": "Dense similarity is not calibrated relevance or verified claim support.",
    }
