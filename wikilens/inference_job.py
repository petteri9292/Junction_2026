"""Bounded, token-protected Hugging Face CPU inference; no model training."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import set_key
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "data" / "inference-job.json"
LEDGER = ROOT / "data" / "compute-budget.json"
IMAGE = "docker.io/ollama/ollama@sha256:910d0d8107b538fb0894e63163eb195de0cc7aefcbc666dc62d3b18f71124592"
TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "DELETED", "TIMEOUT"}
SERVER_COMMAND = """
set -euo pipefail
ollama serve &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
    if ollama list >/dev/null 2>&1; then break; fi
    sleep 1
done
ollama pull bge-m3 >/tmp/bge-pull.log 2>&1
echo BGE_M3_READY
ollama pull qwen3:1.7b >/tmp/qwen-pull.log 2>&1
echo RAG_MODELS_READY
wait "$server"
"""


def start():
    api = HfApi()
    if RECORD.exists():
        current = json.loads(RECORD.read_text())
        job = api.inspect_job(job_id=current["job_id"])
        if job.status.stage not in TERMINAL:
            print(f"Existing inference job: {job.url}")
            return current
    if LEDGER.exists():
        ledger = json.loads(LEDGER.read_text())
    else:
        ledger = {
            "approved_cap_usd": 10,
            "preparation_cost_upper_bound_usd": 0.0025,
            "jobs": [],
        }
    reservation = 0.02  # 30 minutes of CPU + port, with rounding margin.
    already_reserved = ledger["preparation_cost_upper_bound_usd"] + sum(
        job["reserved_upper_bound_usd"] for job in ledger["jobs"]
    )
    if already_reserved + reservation > ledger["approved_cap_usd"]:
        raise ValueError("The cumulative USD 10 allowance requires fresh approval")
    hardware = next(
        item for item in api.list_jobs_hardware() if item.name == "cpu-basic"
    )
    if hardware.unit_label != "minute" or hardware.unit_cost_usd > 0.0002:
        raise ValueError("CPU price changed; review pricing before launching")
    job = api.run_job(
        image=IMAGE,
        command=["bash", "-lc", SERVER_COMMAND],
        env={
            "OLLAMA_HOST": "0.0.0.0:11434",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
        },
        flavor="cpu-basic",
        timeout="30m",
        attempts=1,
        expose=[11434],
        name="wikilens-pretrained-inference",
    )
    record = {
        "job_id": job.id,
        "url": job.url,
        "endpoint": f"https://{job.id}--11434.hf.jobs",
        "created_at": datetime.now(UTC).isoformat(),
        "reserved_upper_bound_usd": reservation,
        "timeout_minutes": 30,
        "purpose": "pretrained BGE-M3 / Qwen inference, no training",
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    ledger["jobs"].append(record)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    set_key(ROOT / ".env", "OLLAMA_URL", record["endpoint"])
    print(
        f"Started {job.url}; hard timeout 30 minutes, reserved maximum USD {reservation:.2f}"
    )
    return record


def stop():
    record = json.loads(RECORD.read_text())
    HfApi().cancel_job(job_id=record["job_id"])
    print(f"Requested cancellation: {record['url']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"])
    args = parser.parse_args()
    if args.action == "start":
        start()
    elif args.action == "stop":
        stop()
    else:
        record = json.loads(RECORD.read_text())
        job = HfApi().inspect_job(job_id=record["job_id"])
        print(job.status.stage, record["url"])


if __name__ == "__main__":
    main()
