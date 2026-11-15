"""Promote one captured inference run after explicit human claim review.

Usage: ``.venv/bin/python -m wikilens.capture report.json review.json output.json``
The output is standalone schema v2, not a replacement for the placeholder UI's
schema v1 fixtures. No inference is run and existing output files are never
overwritten. Review uses zero-based sentence indices, for example::

    {"reviewer":"Petteri", "reviewed_at":"2026-11-14T12:00:00+02:00",
     "claims":[{"index":0,"supported":true,"source_ids":["fi-12-34-0-700"]}]}

Validation checks consistency against the captured raw articles, model outputs,
and metadata. JSON files are unsigned: these checks cannot prove that a model
actually ran or that the stored articles are authentic. Claim support is the
reviewer's explicit judgement, never inferred from citation validity.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from .evidence import revision_url
from .grounding import GroundingError, validate_quotations, validate_synthesis

MODELS = {"bge": "BAAI/bge-m3", "qwen": "Qwen/Qwen3-1.7B"}
LANGUAGE_LABELS = {"fi": "Finnish", "en": "English", "sv": "Swedish"}


def _require(condition, message):
    if not condition:
        raise GroundingError(message)


def _timestamp(value, label):
    _require(isinstance(value, str), f"{label} must be an ISO datetime")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise GroundingError(f"{label} must be an ISO datetime") from error
    _require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed


def _seconds(value, label):
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0,
        f"{label} must be measured nonnegative seconds",
    )
    return value


def _canonical(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise GroundingError("Capture must contain finite JSON values") from error


def _article_key(source):
    lang, page, revision = (
        source.get("lang"),
        source.get("page_id"),
        source.get("rev_id"),
    )
    _require(
        isinstance(lang, str)
        and lang in LANGUAGE_LABELS
        and type(page) is int
        and page > 0
        and type(revision) is int
        and revision > 0,
        "Invalid article identity",
    )
    return lang, page, revision


def _checked_sources(stages):
    articles = stages["retrieval"].get("articles")
    passages = stages["ranking"].get("selected")
    _require(
        isinstance(articles, list) and articles, "Raw retrieved articles are required"
    )
    _require(
        isinstance(passages, list) and passages, "Selected raw passages are required"
    )
    lookup = {}
    for article in articles:
        _require(
            isinstance(article, dict) and isinstance(article.get("text"), str),
            "Raw article text is required",
        )
        key = _article_key(article)
        _require(key not in lookup, "Ambiguous duplicate raw article revision")
        lookup[key] = article
    checked_passages = []
    for passage in passages:
        _require(isinstance(passage, dict), "Selected passage must be an object")
        article = lookup.get(_article_key(passage))
        _require(article is not None, "Selected passage has no raw source article")
        start, end = passage.get("start_char"), passage.get("end_char")
        _require(
            type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(article["text"]),
            "Invalid passage offsets",
        )
        _require(
            passage.get("text") == article["text"][start:end],
            "Selected passage differs from its raw article slice",
        )
        for field in ("title", "lang", "page_id", "rev_id", "wikidata_id"):
            _require(
                passage.get(field) == article.get(field),
                f"Passage metadata differs from raw article: {field}",
            )
        _require(passage.get("lang") in LANGUAGE_LABELS, "Unsupported passage language")
        _require(
            type(passage.get("page_id")) is int
            and passage["page_id"] > 0
            and type(passage.get("rev_id")) is int
            and passage["rev_id"] > 0,
            "Invalid source identifiers",
        )
        _require(
            isinstance(passage.get("title"), str) and passage["title"].strip(),
            "Source title is required",
        )
        expected_url = revision_url(passage["lang"], passage["rev_id"])
        _require(passage.get("revision_url") == expected_url, "Invalid revision link")
        checked_passages.append(passage)
    quotations = stages["quotations"]
    _require(
        isinstance(quotations.get("raw_output"), str),
        "Raw quotation model output is required",
    )
    canonical = validate_quotations(quotations["raw_output"], checked_passages)
    _require(
        canonical and quotations.get("sources") == canonical,
        "Stored quotations differ from validated raw output",
    )
    return canonical


def _checked_review(review, sentences, sources, captured_at):
    _require(
        isinstance(review, dict)
        and set(review) == {"reviewer", "reviewed_at", "claims"},
        "Review must contain reviewer, reviewed_at, and claims",
    )
    _require(
        isinstance(review["reviewer"], str) and review["reviewer"].strip(),
        "A named reviewer is required",
    )
    reviewed_at = _timestamp(review["reviewed_at"], "reviewed_at")
    _require(reviewed_at >= captured_at, "Review predates captured run")
    claims = review["claims"]
    _require(
        isinstance(claims, list) and len(claims) == len(sentences),
        "Every displayed sentence needs one explicit review",
    )
    by_index, allowed = {}, {s["id"] for s in sources}
    for claim in claims:
        _require(
            isinstance(claim, dict)
            and set(claim) == {"index", "supported", "source_ids"},
            "Invalid claim review schema",
        )
        index = claim["index"]
        _require(
            type(index) is int
            and 0 <= index < len(sentences)
            and index not in by_index,
            "Claim index must be unique and zero-based",
        )
        _require(
            claim["supported"] is True,
            "Every displayed claim must be reviewed as supported",
        )
        ids = claim["source_ids"]
        _require(
            isinstance(ids, list) and ids and all(isinstance(s, str) for s in ids),
            "Reviewed claims require source IDs",
        )
        _require(
            len(ids) == len(set(ids))
            and set(ids) <= allowed
            and set(ids) <= set(sentences[index]["sources"]),
            "Reviewed source IDs must cite the sentence's verified quotations",
        )
        by_index[index] = claim
    return {
        "reviewer": review["reviewer"].strip(),
        "reviewed_at": review["reviewed_at"],
        "claims": [by_index[i] for i in range(len(sentences))],
    }


def build_fixture(record, review):
    """Return standalone v2 replay JSON only for a consistent, reviewed draft.

    Raises GroundingError for incomplete/unreviewed/placeholder captures. Raw
    record data is hashed canonically and included in the result for traceability.
    The review approves every displayed sentence; it does not turn the remaining
    captured metadata into verified claims or assert general model quality.
    """
    _require(
        isinstance(record, dict) and record.get("status") == "draft",
        "Only completed draft pipeline captures can be promoted",
    )
    _require(
        record.get("origin") not in ("hand_authored", "placeholder", "mocked")
        and record.get("hand_authored") is not True
        and record.get("mocked") is not True,
        "Hand-authored, placeholder, or mocked records cannot be promoted",
    )
    _require(
        record.get("dataset_release") == "s-2026-10", "Expected dataset S s-2026-10"
    )
    question = record.get("question")
    _require(
        isinstance(question, str) and question.strip(), "Original question is required"
    )
    captured_at = _timestamp(record.get("created_at"), "created_at")
    runtime = record.get("runtime")
    _require(
        isinstance(runtime, dict)
        and runtime.get("mocked") is not True
        and runtime.get("models") == MODELS,
        "Actual model runtime metadata is required",
    )
    revisions = runtime.get("revisions")
    _require(
        isinstance(revisions, dict)
        and all(
            isinstance(revisions.get(key), str) and revisions[key].strip()
            for key in MODELS
        ),
        "Resolved model revisions are required",
    )
    operations = runtime.get("records")
    _require(
        isinstance(operations, list) and operations,
        "Runtime operation records are required",
    )
    completed = set()
    for operation in operations:
        _require(
            isinstance(operation, dict)
            and operation.get("status") == "completed"
            and isinstance(operation.get("stage"), str)
            and operation["stage"] in MODELS,
            "Runtime contains an incomplete operation",
        )
        _seconds(operation.get("seconds"), "Runtime operation seconds")
        completed.add(operation["stage"])
    _require(completed == set(MODELS), "Both model stages must have executed")
    stages = record.get("stages")
    _require(
        isinstance(stages, dict)
        and all(
            isinstance(stages.get(key), dict)
            for key in ("hyde", "retrieval", "ranking", "quotations", "synthesis")
        ),
        "All pipeline stages are required",
    )
    for key in ("hyde", "retrieval", "ranking", "quotations"):
        _require(
            stages[key].get("status") == "completed",
            f"Incomplete pipeline stage: {key}",
        )
    hyde = stages["hyde"]
    _require(
        hyde.get("role") == "retrieval_only_never_evidence"
        and isinstance(hyde.get("raw_output"), str)
        and hyde["raw_output"].strip()
        and hyde.get("text") == hyde["raw_output"],
        "HyDE provenance must be retrieval-only",
    )
    sources = _checked_sources(stages)
    synthesis = stages["synthesis"]
    _require(
        synthesis.get("status") == "draft"
        and isinstance(synthesis.get("raw_output"), str),
        "Raw draft synthesis model output is required",
    )
    checked = validate_synthesis(synthesis["raw_output"], sources)
    _require(
        checked["status"] == "draft"
        and checked["sentences"] == synthesis.get("sentences"),
        "Stored synthesis differs from validated raw output",
    )
    sentences = checked["sentences"]
    review = _checked_review(review, sentences, sources, captured_at)
    timings = {"total_seconds": _seconds(record.get("seconds"), "Total seconds")}
    for key in ("hyde", "ranking", "quotations", "synthesis"):
        timings[key + "_seconds"] = _seconds(
            stages[key].get("seconds"), key + " seconds"
        )
    timings["embedding_and_retrieval_seconds"] = _seconds(
        stages["retrieval"].get("embedding_and_retrieval_seconds"), "Retrieval seconds"
    )
    raw_record = json.loads(_canonical(record))
    digest = hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()
    cited = {source_id for sentence in sentences for source_id in sentence["sources"]}
    evidence_by_id = {}
    for source in sources:
        if source["id"] not in cited:
            continue
        item = {
            key: source[key]
            for key in (
                "id",
                "title",
                "lang",
                "page_id",
                "rev_id",
                "wikidata_id",
                "revision_url",
                "start_char",
                "end_char",
                "quote",
                "quote_start_char",
                "quote_end_char",
            )
            if key in source
        }
        item.update(
            passage=source["text"],
            language_label=LANGUAGE_LABELS[source["lang"]],
            provenance="Literal captured source passage; quotation provenance checked and claims manually reviewed.",
        )
        if source["id"] not in evidence_by_id:
            item["quotes"] = []
            evidence_by_id[source["id"]] = item
        evidence_by_id[source["id"]]["quotes"].append(source["quote"])
    evidence = list(evidence_by_id.values())
    fixture = {
        "id": "capture-" + digest[:12],
        "label": question,
        "question": question,
        "title": "A source-supported answer",
        "theme": "Reviewed pipeline capture",
        "status": "reviewed_pipeline_capture",
        "dataset_release": record["dataset_release"],
        "captured_at": record["created_at"],
        "hypothetical_passage": hyde["text"],
        "timing_basis": record.get(
            "timing_basis", "Measured wall time including model loading and cleanup."
        ),
        "hyde_role": "retrieval_only_never_evidence",
        "timings": timings,
        "answer": [{"text": s["text"], "citations": s["sources"]} for s in sentences],
        "evidence": evidence,
        "citation_mappings": {s["id"]: s["revision_url"] for s in evidence},
        "verification": {
            "pipeline_capture_checked": True,
            "claim_support_reviewed": True,
            "review": review,
            "scope": "Only the displayed sentences were reviewed; no general quality claim.",
        },
        "runtime": runtime,
        "raw_record_sha256": digest,
    }
    return {"schema_version": 2, "examples": [fixture], "raw_record": raw_record}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    fixture = build_fixture(
        json.loads(args.report.read_text()), json.loads(args.review.read_text())
    )
    with args.output.open("x", encoding="utf-8") as output:
        output.write(
            json.dumps(fixture, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print("Wrote reviewed schema-v2 capture:", args.output)


if __name__ == "__main__":
    main()
