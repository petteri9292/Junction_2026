"""Submit/poll full-L work using short-lived clients, independent of log streams.

The original embedding implementation is deliberately unchanged for resume
compatibility. Deploy it once, then use this controller instead of a long
blocking local `.remote()` wait.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import modal
import pyarrow.parquet as pq

from wikilens.corpus_embeddings import digest_config

ROOT = Path(__file__).resolve().parents[1]
APP = "wikilens-full-l-embeddings"
VOLUME = "wikilens-l-embedding-benchmark"


def read_remote(volume, run_id, name):
    return json.loads(b"".join(volume.read_file(f"/{run_id}/{name}")))


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "resume", "verify", "collect"))
    parser.add_argument("run_id")
    args = parser.parse_args()
    volume = modal.Volume.from_name(VOLUME)
    output = ROOT / "data" / "l-embeddings" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    control_path = output / "control.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else {}
    if args.action == "resume":
        if control.get("embedding_call_id"):
            raise ValueError(
                "An embedding call is already recorded; inspect status first"
            )
        ledger_path = ROOT / "data" / "modal-compute-budget.json"
        ledger = json.loads(ledger_path.read_text())
        original = next(r for r in ledger["runs"] if r.get("run_id") == args.run_id)
        provenance = original["provenance"]
        for name, expected in provenance["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Resume implementation changed: {name}")
        inputs = read_remote(volume, args.run_id, "inputs.json")
        if not inputs["complete"] or inputs["config"][
            "implementation_sha256"
        ] != digest_config(provenance["source_sha256"]):
            raise ValueError(
                "Prepared input configuration differs from original implementation"
            )
        if (
            sum(r["reserved_upper_bound_usd"] for r in ledger["runs"]) + 10
            > ledger["credit_allowance_usd"]
        ):
            raise ValueError("Insufficient remaining Modal credit reservation")
        checks = []
        source = ROOT / "data" / "s-2026-10"
        for lang in ("fi", "en", "sv"):
            refs = {
                r["page_id"]: r
                for r in pq.read_table(
                    source / f"vectors-bge-m3-{lang}.parquet"
                ).to_pylist()
            }
            for row in pq.read_table(source / f"articles-{lang}.parquet").to_pylist():
                ref = refs[row["page_id"]]
                if not ref["truncated"] and 200 <= len(row["text"]) <= 900:
                    checks.append(
                        {k: row[k] for k in ("lang", "page_id", "title", "text")}
                        | {"embedding": ref["embedding"]}
                    )
                    if sum(r["lang"] == lang for r in checks) == 3:
                        break
        record = {
            "run_id": args.run_id,
            "status": "submitting_async_resume",
            "created_at": datetime.now(UTC).isoformat(),
            "reserved_upper_bound_usd": 10,
            "gpu": "H100",
            "attempts": 1,
            "gpu_timeout_seconds": 7200,
            "application_retries": 0,
            "purpose": "Resume original user-authorized full-L run after expired client JWT",
            "deployed_app": APP,
            "provenance": provenance,
        }
        ledger["runs"].append(record)
        save(ledger_path, ledger)
        call = modal.Function.from_name(APP, "embed_all").spawn(args.run_id, checks)
        control["embedding_call_id"] = call.object_id
        record.update(status="embedding_async", function_call_id=call.object_id)
        save(control_path, control)
        save(ledger_path, ledger)
        print(json.dumps({"submitted": call.object_id, "run_id": args.run_id}))
        return
    if args.action == "status":
        progress = read_remote(volume, args.run_id, "progress.json")
        result = {
            "run_id": args.run_id,
            "durable_rows": progress["rows"],
            "shards": len(progress["shards"]),
            "vector_bytes": sum(s["bytes"] for s in progress["shards"]),
        }
        for key in ("embedding_call_id", "verification_call_id"):
            if control.get(key):
                try:
                    returned = modal.FunctionCall.from_id(control[key]).get(timeout=0)
                    result[key] = "completed"
                    if key == "embedding_call_id":
                        save(ROOT / "reports" / f"{args.run_id}.json", returned)
                except (TimeoutError, modal.exception.TimeoutError):
                    result[key] = "running"
        print(json.dumps(result))
        return
    if args.action == "verify":
        report = read_remote(volume, args.run_id, "report.json")
        if report["status"] != "complete":
            raise ValueError("Corpus is incomplete")
        if control.get("verification_call_id"):
            raise ValueError("A verification call is already recorded")
        call = modal.Function.from_name(APP, "verify_corpus").spawn(args.run_id)
        control["verification_call_id"] = call.object_id
        save(control_path, control)
        print(json.dumps({"verification_submitted": call.object_id}))
        return
    report = read_remote(volume, args.run_id, "report.json")
    verification = read_remote(volume, args.run_id, "verification.json")
    if report["status"] != "complete" or verification["status"] != "verified":
        raise ValueError("Corpus must be complete and verified before collection")
    for name in (
        "inputs.json",
        "report.json",
        "verification.json",
        "compatibility.json",
    ):
        save(output / name, read_remote(volume, args.run_id, name))
    save(ROOT / "reports" / f"{args.run_id}.json", report)
    save(ROOT / "reports" / f"{args.run_id}-verification.json", verification)

    def download_shard(spec):
        destination = output / spec["file"]
        if not destination.exists() or sha(destination) != spec["sha256"]:
            temporary = destination.with_suffix(".part")
            with temporary.open("wb") as stream:
                for chunk in volume.read_file(f"/{args.run_id}/{spec['file']}"):
                    stream.write(chunk)
            if sha(temporary) != spec["sha256"]:
                raise ValueError("Downloaded shard checksum mismatch")
            temporary.replace(destination)
        print("DOWNLOADED", spec["file"], flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_shard, spec) for spec in report["shards"]]
        for future in as_completed(futures):
            future.result()
    print("COLLECTED", output, flush=True)


if __name__ == "__main__":
    main()
