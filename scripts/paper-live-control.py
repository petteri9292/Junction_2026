"""Reserve/start/status/stop one warm Paper session; no implicit GPU restarts."""

import argparse
import datetime
import json
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["reserve", "start", "status", "stop", "ask", "collect"]
    )
    parser.add_argument("--question")
    parser.add_argument("--context", default="")
    parser.add_argument(
        "--job", type=Path, default=ROOT / "data/modal-paper-live-job.json"
    )
    args = parser.parse_args()
    if args.action == "reserve":
        if args.job.exists():
            raise SystemExit(
                "Job record already exists; inspect it before another session"
            )
        combined = json.loads((ROOT / "data/project-compute-budget.json").read_text())
        hf = json.loads((ROOT / "data/compute-budget.json").read_text())
        modal_ledger = json.loads((ROOT / "data/modal-compute-budget.json").read_text())
        assert (
            hf["approved_cap_usd"]
            == modal_ledger["authorized_combined_project_cap_usd"]
            == combined["approved_combined_cap_usd"]
        )
        reserve = 10.0
        if (
            combined["current_combined_reserved_usd"] + reserve
            > combined["approved_combined_cap_usd"]
        ):
            raise SystemExit("Combined cap would be exceeded")
        now = datetime.datetime.now(datetime.UTC)
        job = {
            "run_id": "paper-live-" + now.strftime("%Y%m%dT%H%M%SZ"),
            "app": "wikilens-paper-live",
            "source_run_id": "l-search-20261114T185620Z",
            "reserved_upper_bound_usd": reserve,
            "gpu": "H100!",
            "cpu_physical": 8,
            "memory_gib": 32,
            "timeout_seconds": 7200,
            "attempts": 1,
            "status": "reserved",
            "created_at_utc": now.isoformat(),
            "keep_open": True,
            "purpose": "User requested live Paper demo with persistent Modal model and database",
            "funding_note": "Modal promotional credits tracked separately; provider charges not yet reconciled",
        }
        modal_ledger["runs"].append(job)
        combined["modal_reserved_usd"] += reserve
        combined["current_combined_reserved_usd"] += reserve
        combined["updated_at_utc"] = now.isoformat()
        save(ROOT / "data/modal-compute-budget.json", modal_ledger)
        save(ROOT / "data/project-compute-budget.json", combined)
        save(args.job, job)
        print(json.dumps(job))
        return
    import sys

    sys.path.insert(0, str(ROOT))
    import modal

    from wikilens.paper_live import QUEUE, ModalPaperBridge

    job = json.loads(args.job.read_text())
    if args.action == "start":
        if job.get("call_id") or job["status"] != "reserved":
            raise SystemExit("Session already started; no automatic retry")
        job["deadline_unix"] = time.time() + 7170
        # Save intent before spawning, so an ambiguous network failure is not retried.
        job["status"] = "starting_once"
        save(args.job, job)
        call = modal.Function.from_name(job["app"], "serve").spawn(
            job["run_id"], job["source_run_id"], job["deadline_unix"]
        )
        job.update(call_id=call.object_id, status="submitted")
        save(args.job, job)
        ledger_path = ROOT / "data/modal-compute-budget.json"
        ledger = json.loads(ledger_path.read_text())
        ledger["runs"] = [
            job if row.get("run_id") == job["run_id"] else row for row in ledger["runs"]
        ]
        save(ledger_path, ledger)
        print(json.dumps(job))
    elif args.action == "status":
        print(json.dumps(ModalPaperBridge(args.job).status(), indent=2))
    elif args.action == "stop":
        modal.Queue.from_name(QUEUE).put({"action": "stop"}, partition=job["run_id"])
        if job.get("parent_session_id") and ModalPaperBridge(args.job).status()[
            "status"
        ] in {"failed", "closed", "unavailable"}:
            modal.Queue.from_name(QUEUE).put(
                {"action": "stop"}, partition=job["parent_session_id"]
            )
        print("Stop queued; worker will close after its current question")
    elif args.action == "ask":
        rid = uuid.uuid4().hex
        bridge = ModalPaperBridge(args.job)
        print(
            bridge.submit(
                {"request_id": rid, "question": args.question, "context": args.context}
            )
        )
        print("request_id=" + rid)
    else:
        volume = modal.Volume.from_name("wikilens-l-embedding-benchmark")
        base = "/paper-live/" + job["run_id"]
        out = ROOT / "data/paper-live" / job["run_id"]
        for entry in volume.iterdir(base, recursive=True):
            if entry.type != 1:
                continue
            relative = Path(entry.path.lstrip("/")).relative_to(base.lstrip("/"))
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"".join(volume.read_file(entry.path)))
        print(out)


if __name__ == "__main__":
    main()
