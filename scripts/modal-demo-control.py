"""Submit, inspect, or download the bounded demo capture job."""

import argparse
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["submit", "status", "collect", "ask", "stop"])
parser.add_argument(
    "--job", type=Path, default=ROOT / "data/modal-demo-capture-job.json"
)
parser.add_argument("--question")
parser.add_argument("--context", default="")
args = parser.parse_args()
action = args.action
path = args.job
job = json.loads(path.read_text())
volume = modal.Volume.from_name("wikilens-l-embedding-benchmark")
base = "/demo-captures/" + job["run_id"]
if action == "submit":
    if job.get("call_id"):
        raise SystemExit("Already submitted; inspect existing call instead")
    call = modal.Function.from_name(job["app"], "capture").spawn(
        job["run_id"],
        job["source_run_id"],
        **({"keep_open": True} if job.get("keep_open") else {}),
    )
    job.update(call_id=call.object_id, status="submitted")
    path.write_text(json.dumps(job, indent=2) + "\n")
    print(call.object_id)
elif action == "status":
    try:
        result = modal.FunctionCall.from_id(job["call_id"]).get(timeout=0)
        print(json.dumps(result, indent=2))
    except TimeoutError:
        print("RUNNING")
    for name in ["progress.json", "report.json", "session.json"]:
        try:
            value = json.loads(b"".join(volume.read_file(base + "/" + name)))
            if name == "report.json":
                value = {
                    k: value[k]
                    for k in [
                        "model_setup_seconds",
                        "runs",
                        "arms",
                        "total_seconds",
                        "frozen_calls",
                        "pipeline_runs",
                        "total_worker_seconds",
                    ]
                    if k in value
                }
            print(name, json.dumps(value))
        except (modal.exception.NotFoundError, FileNotFoundError):
            pass
elif action in {"ask", "stop"}:
    if not job.get("keep_open"):
        raise SystemExit("This job is not a warm prompt session")
    import uuid

    queue = modal.Queue.from_name("wikilens-fp8-prompts", create_if_missing=True)
    request = {"action": action, "request_id": uuid.uuid4().hex}
    if action == "ask":
        session = json.loads(b"".join(volume.read_file(base + "/session.json")))
        if (
            session.get("status") not in {"ready", "answering"}
            or time.time() > session["accepts_requests_until_unix"]
        ):
            raise SystemExit(
                "This warm session is closed or too near its deadline; no request was queued."
            )
        if not args.question or not args.question.strip():
            raise SystemExit("--question is required")
        request.update(question=args.question.strip(), context=args.context)
    queue.put(request, partition=job["run_id"])
    job.setdefault("followup_requests", []).append(request)
    path.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps(request, ensure_ascii=False))
else:
    out = ROOT / "data/demo-captures" / job["run_id"]
    for entry in volume.iterdir(base, recursive=True):
        if entry.type != 1:
            continue
        relative = Path(entry.path.lstrip("/")).relative_to(base.lstrip("/"))
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            for block in volume.read_file(entry.path):
                stream.write(block)
        print(relative)
    print(out)
