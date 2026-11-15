"""Append-only MariaDB evidence trail for individually reviewed schema-v2 captures.

Imports revalidate the raw capture through ``build_fixture`` before touching the
database. Normalized rows are inserted atomically and are never updated. The
report hash is the identity; a different review for that same report is rejected
rather than silently changing its original audit trail. Stored sources remain
immutable even when the working ``article`` table is refreshed.

The additive migration also creates update/delete rejection triggers. It never
alters or replaces the existing article/vector tables. Keep DB write credentials
private; a privileged administrator can still deliberately remove DB safeguards.
"""

import hashlib
import json
import re
from pathlib import Path

import pymysql

from .capture import build_fixture
from .database import connect

TABLES = (
    "provenance_run",
    "provenance_evidence",
    "provenance_claim",
    "provenance_citation",
    "provenance_review",
    "provenance_claim_review",
)

# One relational query reconstructs every claim/citation and its original quote.
INSPECTION_SQL = """SELECT c.claim_index,c.claim_text,e.evidence_index,e.source_id,
    e.lang,e.page_id,e.rev_id,e.wikidata_id,e.title,e.revision_url,
    e.discovery_origin,e.passage_start,e.passage_end,e.passage,e.quotation,
    e.quote_start,e.quote_end,cr.supported,cr.source_ids_json,
    r.reviewer,r.reviewed_at
    FROM provenance_claim c
    JOIN provenance_citation x USING(report_hash,claim_index)
    JOIN provenance_evidence e ON e.report_hash=x.report_hash
        AND e.evidence_index=x.evidence_index
    JOIN provenance_claim_review cr ON cr.report_hash=c.report_hash
        AND cr.claim_index=c.claim_index
    JOIN provenance_review r ON r.report_hash=c.report_hash
    WHERE c.report_hash=%s
    ORDER BY c.claim_index,e.evidence_index"""


def _json(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def validate_capture(capture):
    """Rebuild the fixture from raw inputs/review; reject any tampered display."""
    if (
        not isinstance(capture, dict)
        or capture.get("schema_version") != 2
        or not isinstance(capture.get("examples"), list)
        or len(capture["examples"]) != 1
        or not isinstance(capture["examples"][0], dict)
    ):
        raise ValueError("Import one reviewed schema-v2 capture with its raw record")
    fixture = capture["examples"][0]
    if fixture.get("status") not in {
        "reviewed_pipeline_capture",
        "reviewed_abstention_capture",
    }:
        raise ValueError("Only explicitly reviewed pipeline captures can be imported")
    verification = fixture.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("Explicit verification and review metadata are required")
    review = verification.get("review")
    rebuilt = build_fixture(capture.get("raw_record"), review)
    if _json(rebuilt) != _json(capture):
        raise ValueError("Capture differs from its revalidated raw record and review")
    return rebuilt


def initialize_provenance():
    """Install only additive evidence tables and append-only DB safeguards."""
    with connect() as db, db.cursor() as cursor:
        for statement in (
            Path(__file__).with_name("provenance.sql").read_text().split(";")
        ):
            if statement.strip():
                cursor.execute(statement)
        for table in TABLES:
            for operation in ("UPDATE", "DELETE"):
                trigger = f"{table}_reject_{operation.lower()}"
                cursor.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {trigger} BEFORE {operation} ON {table} "
                    "FOR EACH ROW SIGNAL SQLSTATE '45000' "
                    "SET MESSAGE_TEXT='Evidence provenance is append-only'"
                )
        db.commit()


def _existing(cursor, report_hash, capture_hash):
    cursor.execute(
        "SELECT report_hash,capture_hash FROM provenance_run WHERE report_hash=%s",
        (report_hash,),
    )
    existing = cursor.fetchone()
    if existing is None:
        return None
    if existing["capture_hash"] != capture_hash:
        raise ValueError("This report already has a different immutable review/capture")
    return {"report_hash": report_hash, "capture_hash": capture_hash, "imported": False}


def _insert_capture(cursor, capture):
    """Insert inside the caller's transaction; caller owns commit/rollback."""
    fixture, record = capture["examples"][0], capture["raw_record"]
    report_hash, capture_hash = fixture["raw_record_sha256"], _hash(capture)
    existing = _existing(cursor, report_hash, capture_hash)
    if existing:
        return existing
    stages = record["stages"]
    diagnostics = {
        "timings": fixture["timings"],
        "timing_basis": fixture.get("timing_basis"),
        "hyde": stages.get("hyde"),
        "retrieval": {k: v for k, v in stages["retrieval"].items() if k != "articles"},
        "ranking_config": stages["ranking"].get("config"),
        "generation_config": stages["synthesis"].get("config"),
    }
    cursor.execute(
        """INSERT INTO provenance_run
        (report_hash,capture_hash,question,dataset_release,capture_status,captured_at,
         runtime_json,diagnostics_json,capture_json) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            report_hash,
            capture_hash,
            fixture["question"],
            fixture["dataset_release"],
            fixture["status"],
            fixture["captured_at"],
            _json(record["runtime"]),
            _json(diagnostics),
            _json(capture),
        ),
    )
    review = fixture["verification"]["review"]
    cursor.execute(
        "INSERT INTO provenance_review (report_hash,reviewer,reviewed_at,review_json) VALUES (%s,%s,%s,%s)",
        (report_hash, review["reviewer"], review["reviewed_at"], _json(review)),
    )
    origins = {
        (a["lang"], a["page_id"], a["rev_id"]): a.get("origin", "unknown")
        for a in stages["retrieval"].get("articles", [])
    }
    evidence_indices = {}
    for index, source in enumerate(stages["quotations"].get("sources", [])):
        evidence_indices.setdefault(source["id"], []).append(index)
        identity = (source["lang"], source["page_id"], source["rev_id"])
        cursor.execute(
            """INSERT INTO provenance_evidence
            (report_hash,evidence_index,source_id,lang,page_id,rev_id,wikidata_id,title,
             revision_url,discovery_origin,passage_start,passage_end,passage,quotation,
             quote_start,quote_end) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                report_hash,
                index,
                source["id"],
                *identity,
                source.get("wikidata_id"),
                source["title"],
                source["revision_url"],
                origins.get(identity, "unknown"),
                source["start_char"],
                source["end_char"],
                source["text"],
                source["quote"],
                source["quote_start_char"],
                source["quote_end_char"],
            ),
        )
    reviews = {r["index"]: r for r in review.get("claims", [])}
    for index, claim in enumerate(fixture["answer"]):
        cursor.execute(
            "INSERT INTO provenance_claim (report_hash,claim_index,claim_text) VALUES (%s,%s,%s)",
            (report_hash, index, claim["text"]),
        )
        checked = reviews[index]
        cursor.execute(
            """INSERT INTO provenance_claim_review
            (report_hash,claim_index,supported,source_ids_json) VALUES (%s,%s,%s,%s)""",
            (report_hash, index, checked["supported"], _json(checked["source_ids"])),
        )
        for source_id in claim["citations"]:
            for evidence_index in evidence_indices[source_id]:
                cursor.execute(
                    "INSERT INTO provenance_citation (report_hash,claim_index,evidence_index) VALUES (%s,%s,%s)",
                    (report_hash, index, evidence_index),
                )
    return {"report_hash": report_hash, "capture_hash": capture_hash, "imported": True}


def import_capture(capture):
    """Revalidate then atomically insert one reviewed capture, idempotently."""
    capture = validate_capture(capture)
    initialize_provenance()
    db = connect()
    try:
        db.begin()
        with db.cursor() as cursor:
            try:
                result = _insert_capture(cursor, capture)
            except pymysql.err.IntegrityError as error:
                # Concurrent identical imports may race on the unique report hash.
                # Roll back the loser and verify the winner; never ignore partial rows.
                if error.args[0] != 1062:
                    raise
                db.rollback()
                with db.cursor() as winner:
                    result = _existing(
                        winner,
                        capture["examples"][0]["raw_record_sha256"],
                        _hash(capture),
                    )
                if result is None:
                    raise
        db.commit()
        return result
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def inspect_run(report_hash):
    """Return the reconstructed evidence trail; this operation never writes."""
    if not isinstance(report_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", report_hash
    ):
        raise ValueError("Use a lowercase SHA-256 report hash")
    with connect() as db, db.cursor() as cursor:
        cursor.execute(
            """SELECT report_hash,capture_hash,question,dataset_release,capture_status,
               captured_at,runtime_json,diagnostics_json,imported_at
               FROM provenance_run WHERE report_hash=%s""",
            (report_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        runtime, diagnostics = (
            json.loads(row.pop("runtime_json")),
            json.loads(row.pop("diagnostics_json")),
        )
        cursor.execute(
            "SELECT reviewer,reviewed_at,review_json FROM provenance_review WHERE report_hash=%s",
            (report_hash,),
        )
        review = cursor.fetchone()
        review["details"] = json.loads(review.pop("review_json"))
        cursor.execute(INSPECTION_SQL, (report_hash,))
        evidence_rows = cursor.fetchall()
    claims = {}
    for evidence in evidence_rows:
        index = evidence["claim_index"]
        reviewed_ids = json.loads(evidence["source_ids_json"])
        claim = claims.setdefault(
            index,
            {
                "index": index,
                "text": evidence["claim_text"],
                "reviewed_support": bool(evidence["supported"]),
                "citations": [],
            },
        )
        citation = {
            k: evidence[k]
            for k in (
                "source_id",
                "lang",
                "page_id",
                "rev_id",
                "wikidata_id",
                "title",
                "revision_url",
                "discovery_origin",
                "passage_start",
                "passage_end",
                "passage",
                "quotation",
                "quote_start",
                "quote_end",
            )
        }
        citation["reviewed_support"] = evidence["source_id"] in reviewed_ids
        claim["citations"].append(citation)
    return {
        **row,
        "runtime": runtime,
        "diagnostics": diagnostics,
        "review": review,
        "claims": [claims[i] for i in sorted(claims)],
        "limitation": "Review is explicitly attributed; citation validity alone does not establish claim support.",
    }
