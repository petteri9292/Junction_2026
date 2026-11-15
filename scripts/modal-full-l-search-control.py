"""Short-lived client for bounded full-L import/search and artifact collection."""

import argparse, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import modal

ROOT = Path(__file__).resolve().parents[1]
APP = "wikilens-full-l-search"
VOLUME = "wikilens-l-embedding-benchmark"
p = argparse.ArgumentParser()
p.add_argument(
    "action", choices=["import", "status", "collect", "development", "heldout"]
)
args = p.parse_args()
control_path = ROOT / "data/modal-full-l-search-job.json"
volume = modal.Volume.from_name(VOLUME)
if args.action == "import":
    if control_path.exists():
        raise SystemExit(
            "Import record exists; inspect rather than automatically retrying"
        )
    ledger_path = ROOT / "data/modal-compute-budget.json"
    ledger = json.loads(ledger_path.read_text())
    if (
        sum(r["reserved_upper_bound_usd"] for r in ledger["runs"]) + 3.8
        > ledger["credit_allowance_usd"]
    ):
        raise SystemExit("Credit reservation exceeded")
    config = json.loads(
        (
            ROOT / "data/evaluation/modal-full-l-cutoff-20261114/questions.json"
        ).read_text()
    )
    run_id = "l-search-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "run_id": run_id,
        "reserved_upper_bound_usd": 3.8,
        "status": "submitting",
        "app": APP,
        "attempts": 1,
        "cpu_import": {"cores": 8, "memory_gib": 32, "timeout_seconds": 7200},
        "gpu_search": {
            "gpu": "T4",
            "cores": 4,
            "memory_gib": 32,
            "timeout_seconds_per_split": 1800,
            "combined_timeout_seconds": 3600,
        },
        "purpose": "Full-L native MariaDB import and cutoff inference; no training",
        "config": config,
        "rates_source": "https://modal.com/pricing",
        "maximum_compute_quote_usd": 2.301264,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "scripts/modal-full-l-search.py",
                "scripts/import-l-embeddings.py",
                "scripts/run-reranker-cutoff.py",
                "uv.lock",
            ]
        },
    }
    ledger["runs"].append(record.copy())
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    control_path.write_text(json.dumps(record, indent=2) + "\n")
    call = modal.Function.from_name(APP, "import_full").spawn(run_id, config)
    record.update(import_call_id=call.object_id, status="import_submitted")
    control_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"run_id": run_id, "import_call_id": call.object_id}))
    raise SystemExit
c = json.loads(control_path.read_text())
base = "/full-l-search/" + c["run_id"]
if args.action in ["development", "heldout"]:
    key = args.action + "_call_id"
    if key in c:
        raise SystemExit("Call already recorded; no automatic retry")
    if args.action == "heldout":
        policy = (
            ROOT / "data/evaluation/modal-full-l-cutoff-20261114/frozen-policy.json"
        )
        if not policy.exists():
            raise SystemExit("Freeze full-L development policy before heldout")
        with volume.batch_upload() as batch:
            batch.put_file(policy, base + "/frozen-policy.json")
    call = modal.Function.from_name(APP, "search").spawn(
        c["run_id"], c["config"], args.action
    )
    c[key] = call.object_id
    control_path.write_text(json.dumps(c, indent=2) + "\n")
    print(call.object_id)
    raise SystemExit
if args.action == "status":
    import subprocess

    try:
        containers = json.loads(
            subprocess.check_output(
                [str(ROOT / ".venv/bin/modal"), "container", "list", "--json"],
                text=True,
                timeout=15,
            )
        )
        for container in containers:
            if container.get("app_name") != APP:
                continue
            probe = "from pathlib import Path;import json;p=Path('/workspace/Junction_2026/reports/l-corpus-import.json');d=json.loads(p.read_text()) if p.exists() else {'stage':'database setup'};print(json.dumps({k:d[k] for k in ['stage','status','committed_rows','elapsed_seconds'] if k in d}))"
            value = subprocess.check_output(
                [
                    str(ROOT / ".venv/bin/modal"),
                    "container",
                    "exec",
                    container["container_id"],
                    "--",
                    "/workspace/Junction_2026/.venv/bin/python",
                    "-c",
                    probe,
                ],
                text=True,
                timeout=15,
            )
            print("live_checkpoint", value.strip())
    except (subprocess.SubprocessError, OSError):
        print("Live checkpoint unavailable; durable results checked below.")
    for name in [
        "import-result.json",
        "import-failure.json",
        "reports/l-corpus-import.json",
        "captures/report-development.json",
        "captures/report-heldout.json",
    ]:
        try:
            d = json.loads(b"".join(volume.read_file(base + "/" + name)))
            print(
                name,
                json.dumps(
                    {
                        k: d[k]
                        for k in [
                            "status",
                            "committed_rows",
                            "elapsed_seconds",
                            "failures",
                        ]
                        if k in d
                    }
                ),
            )
        except (modal.exception.NotFoundError, FileNotFoundError):
            pass
    for key, value in c.items():
        if key.endswith("_call_id"):
            try:
                print(key, modal.FunctionCall.from_id(value).get(timeout=0))
            except TimeoutError:
                print(key, "running")
    raise SystemExit
out = ROOT / "data/modal-full-l-search" / c["run_id"]
out.mkdir(parents=True, exist_ok=True)
entries = [
    e
    for e in volume.iterdir(base, recursive=True)
    if e.type == 1 and not e.path.endswith("mariadb-clean.tar")
]


def download(e):
    rel = Path(e.path.lstrip("/")).relative_to(base.lstrip("/"))
    target = out / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        for b in volume.read_file(e.path):
            f.write(b)
    return str(rel)


with ThreadPoolExecutor(max_workers=6) as pool:
    for name in pool.map(download, entries):
        print(name)
print("COLLECTED", out)
