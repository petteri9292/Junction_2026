"""Literal evidence slices and strict output schemas; no entailment judgement.

These helpers do not run a model or modify the database. A valid citation or a
verbatim quotation establishes provenance only. A human must still decide if
the evidence supports the answer, and whether the answer is complete.
"""

import json
import re
from numbers import Integral

from .evidence import revision_url

LIMITATION = (
    "Citation identifiers and quotation provenance do not establish claim support, "
    "relevance, or completeness. Review every claim against its evidence."
)


class GroundingError(ValueError):
    """An evidence input or generated response failed validation."""


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise GroundingError(f"{label} must be a positive integer")
    return int(value)


def _string(value, label, max_length=None):
    if not isinstance(value, str) or not value.strip():
        raise GroundingError(f"{label} must be a nonempty string")
    if max_length is not None and len(value) > max_length:
        raise GroundingError(f"{label} exceeds {max_length} characters")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GroundingError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _payload(value):
    """Accept parsed JSON or one optional fenced JSON response, never eval."""
    if isinstance(value, str):
        raw = value.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
        if fenced:
            raw = fenced[1]
        try:
            return json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    GroundingError(f"Invalid JSON constant: {token}")
                ),
            )
        except json.JSONDecodeError as error:
            raise GroundingError("Model response is not valid JSON") from error
    return value


def chunk_articles(articles, max_chars=1400, overlap_chars=300):
    """Deduplicate article revisions and slice their *entire* texts exactly.

    Offsets count Python Unicode characters, with an exclusive end. No stripping,
    translation, heading removal, or truncation occurs. Whitespace-only windows
    are omitted. Stable IDs depend on language/page/revision and character range,
    so changing retrieval order does not change a passage's ID. Different source
    revisions remain distinct. A repeated revision with conflicting text fails.
    """
    max_chars = _positive_integer(max_chars, "max_chars")
    if (
        isinstance(overlap_chars, bool)
        or not isinstance(overlap_chars, Integral)
        or not 0 <= overlap_chars < max_chars
    ):
        raise GroundingError("Require 0 <= overlap_chars < max_chars")
    overlap_chars = int(overlap_chars)
    seen = {}
    passages = []
    for rank, article in enumerate(articles, 1):
        if not isinstance(article, dict):
            raise GroundingError("Each article must be an object")
        lang = article.get("lang")
        if lang not in ("fi", "en", "sv"):
            raise GroundingError("Unsupported source language")
        page = _positive_integer(article.get("page_id"), "page_id")
        revision = _positive_integer(article.get("rev_id"), "rev_id")
        text = article.get("text")
        if not isinstance(text, str):
            raise GroundingError("Article text must be a string")
        key = (lang, page, revision)
        if key in seen:
            if seen[key] != text:
                raise GroundingError("Same article revision has conflicting text")
            continue
        seen[key] = text
        for start in range(0, len(text), max_chars - overlap_chars):
            end = min(start + max_chars, len(text))
            if text[start:end].strip():
                passage = {
                    field: article[field]
                    for field in (
                        "title",
                        "url",
                        "wikidata_id",
                        "distance",
                        "retrieval_rank",
                    )
                    if field in article
                }
                passage.update(
                    id=f"{lang}-{page}-{revision}-{start}-{end}",
                    lang=lang,
                    page_id=page,
                    rev_id=revision,
                    revision_url=revision_url(lang, revision),
                    start_char=start,
                    end_char=end,
                    text=text[start:end],
                )
                passage.setdefault("retrieval_rank", rank)
                passages.append(passage)
            if end == len(text):
                break
    return passages


def _passages_by_id(passages):
    lookup = {}
    for passage in passages:
        if not isinstance(passage, dict):
            raise GroundingError("Each passage must be an object")
        source_id = _string(passage.get("id"), "Passage ID")
        _string(passage.get("text"), "Passage text")
        if source_id in lookup:
            raise GroundingError("Duplicate passage ID")
        lookup[source_id] = passage
    return lookup


def validate_quotations(selections, passages):
    """Verify up to three 6–100-word quotations, rejecting the whole response.

    Whitespace variation is accepted. Returned `quote` is the corresponding
    literal source slice, including its original whitespace. `quote_start_char`
    and `quote_end_char` are relative to the cited passage; article offsets are
    added when the passage supplies `start_char`. This does not assess relevance.
    Duplicate selections of the same quotation are rejected; multiple distinct
    quotations from one passage are allowed.
    """
    selections = _payload(selections)
    lookup = _passages_by_id(passages)
    if not isinstance(selections, list) or len(selections) > 3:
        raise GroundingError("Expected at most three quotation selections")
    verified = []
    seen = set()
    for selection in selections:
        if not isinstance(selection, dict) or set(selection) != {"source_id", "quote"}:
            raise GroundingError("Expected source_id and quote fields")
        source_id = _string(selection["source_id"], "Source ID")
        quote = _string(selection["quote"], "Quote")
        if source_id not in lookup:
            raise GroundingError("Unknown quotation source")
        words = quote.split()
        if not 6 <= len(words) <= 100:
            raise GroundingError("Quote must contain 6 to 100 words")
        passage = lookup[source_id]
        match = re.search(
            r"\s+".join(re.escape(word) for word in words), passage["text"]
        )
        if match is None:
            raise GroundingError("Quote does not occur in its cited passage")
        identity = (source_id, match.start(), match.end())
        if identity in seen:
            raise GroundingError("Duplicate quotation selection")
        seen.add(identity)
        source = {
            field: passage[field]
            for field in (
                "title",
                "lang",
                "page_id",
                "rev_id",
                "wikidata_id",
                "revision_url",
                "start_char",
                "end_char",
                "text",
            )
            if field in passage
        }
        source.update(
            id=source_id,
            source_id=source_id,
            quote=match[0],
            quote_start_char=match.start(),
            quote_end_char=match.end(),
            provenance_checked=True,
        )
        if "start_char" in passage:
            start = passage["start_char"]
            if isinstance(start, bool) or not isinstance(start, Integral) or start < 0:
                raise GroundingError("Invalid passage character offset")
            source["quote_article_start_char"] = int(start) + match.start()
            source["quote_article_end_char"] = int(start) + match.end()
        verified.append(source)
    return verified


def validate_synthesis(payload, sources):
    """Check draft structure and citation membership, never claim entailment.

    `sources` should come from `validate_quotations`. Minimal {id, quote} source
    dictionaries are accepted, but their provenance cannot be established here.
    Model-authored URLs/HTML/inline citation markup are rejected: a renderer must
    escape text and build citation links from trusted source metadata itself.
    """
    payload = _payload(payload)
    allowed = set()
    for source in sources:
        if not isinstance(source, dict):
            raise GroundingError("Each source must be an object")
        allowed.add(_string(source.get("id"), "Source ID"))
        _string(source.get("quote"), "Source quote")
    if not isinstance(payload, dict) or set(payload) != {
        "insufficient_evidence",
        "sentences",
    }:
        raise GroundingError("Expected insufficient_evidence and sentences fields")
    insufficient = payload["insufficient_evidence"]
    sentences = payload["sentences"]
    if type(insufficient) is not bool or not isinstance(sentences, list):
        raise GroundingError("Invalid synthesis field types")
    if insufficient and sentences:
        raise GroundingError("Abstention must contain no sentences")
    if not insufficient and not 1 <= len(sentences) <= 4:
        raise GroundingError("Expected one to four answer sentences")
    checked = []
    for sentence in sentences:
        if not isinstance(sentence, dict) or set(sentence) != {"text", "sources"}:
            raise GroundingError("Expected text and sources fields for each sentence")
        text = _string(sentence["text"], "Answer text", 700).strip()
        citations = sentence["sources"]
        if not isinstance(citations, list) or not citations:
            raise GroundingError("Every sentence requires citations")
        for source_id in citations:
            if not isinstance(source_id, str) or source_id not in allowed:
                raise GroundingError("Unknown answer citation")
        if len(set(citations)) != len(citations):
            raise GroundingError("Duplicate answer citation")
        if re.search(r"https?://|[\[\]<>]", text, flags=re.IGNORECASE):
            raise GroundingError(
                "Answer text must not contain URLs, HTML, or inline citations"
            )
        checked.append({"text": text, "sources": list(citations)})
    return {
        "status": "insufficient_evidence" if insufficient else "draft",
        "sentences": checked,
        "claim_support_reviewed": False,
        "limitation": LIMITATION,
    }
