"""Apply a checked request-handler update inside the existing warm allocation.

The original bounded job owns vLLM and MariaDB until its unchanged deadline.
The replacement controller loads only BGE/reranker and uses those warm services.
No new GPU job is created; the original reservation covers this process.
"""

import base64
import datetime
import json
import subprocess
import time
import zlib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "wikilens/admin_tools.py",
    "wikilens/hyde_search.py",
    "wikilens/finnish_prompts.py",
    "wikilens/text_review.py",
    "wikilens/gradio_demo.py",
    "wikilens/paper_live.py",
    "scripts/run-paper-live.py",
]
BOOTSTRAP = """import base64,json,os,subprocess,sys,zlib
from pathlib import Path
p=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));root=Path('/workspace/Junction_2026')
if p.get('stop_worker_pid'):
 import signal
 pid=p['stop_worker_pid'];command=Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode().replace('\\x00',' ')
 if 'run-paper-live.py' not in command or p['previous_session'] not in command:raise RuntimeError('Controller identity mismatch')
 os.kill(pid,signal.SIGTERM)
for name,data in p['files'].items():(root/name).write_bytes(base64.b64decode(data))
out=Path('/data/paper-live')/p['run_id'];out.mkdir(parents=True,exist_ok=True)
(out/'controller-attempt.json').write_text(json.dumps({'parent_session':p['parent_session'],'deadline_unix':p['deadline_unix'],'attempts':1})+'\\n')
with (out/'worker.log').open('a') as log:
 process=subprocess.Popen([str(root/'inference/vllm/.venv/bin/python'),'-u',str(root/'scripts/run-paper-live.py'),'--output',str(out),'--source-run',p['source_run_id'],'--deadline',str(p['deadline_unix']),'--gpu',p['gpu'],'--reuse-services','--parent-session',p['parent_session']],cwd=root,env=dict(os.environ,PYTHONPATH=str(root),HF_HOME='/data/hf'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':process.pid,'run_id':p['run_id']}))
"""


def main():
    job_path = ROOT / "data/modal-paper-live-job.json"
    old = json.loads(job_path.read_text())
    results = modal.Dict.from_name("wikilens-paper-results")
    session = results.get(old["run_id"])
    if session["status"] != "ready":
        raise SystemExit(
            "Wait until the current question finishes before updating the controller"
        )
    if old["deadline_unix"] - time.time() < 1000:
        raise SystemExit("Session is too near its existing deadline")
    now = datetime.datetime.now(datetime.UTC)
    run_id = "paper-language-" + now.strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "files": {
            name: base64.b64encode((ROOT / name).read_bytes()).decode()
            for name in FILES
        },
        "run_id": run_id,
        "parent_session": old.get("parent_session_id", old["run_id"]),
        "source_run_id": old["source_run_id"],
        "deadline_unix": old["deadline_unix"],
        "gpu": "H200" if old["gpu"] == "H200" else "H100",
        "previous_session": old["run_id"],
        "stop_worker_pid": session["worker_pid"]
        if old.get("parent_session_id")
        else None,
    }
    # Persist intent before exec; an ambiguous transport failure is never retried automatically.
    attempt_path = ROOT / "data" / ("controller-update-" + old["run_id"] + ".json")
    if attempt_path.exists():
        raise SystemExit(
            "Controller update already attempted; inspect saved attempt before another action"
        )
    attempt_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "container_id": session["container_id"],
                "created_at": now.isoformat(),
            }
        )
        + "\n"
    )
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/modal"),
            "container",
            "exec",
            session["container_id"],
            "--",
            "python",
            "-c",
            BOOTSTRAP,
            encoded,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    print(completed.stdout, flush=True)
    for _ in range(60):
        value = results.get(run_id)
        if value and value["status"] == "ready":
            break
        if value and value["status"] == "failed":
            raise RuntimeError(value.get("error", "Controller failed"))
        time.sleep(1)
    else:
        raise RuntimeError(
            "Controller did not become ready; original warm service remains available"
        )
    assert value["container_id"] == session["container_id"]
    backup = ROOT / "data" / ("before-controller-" + old["run_id"] + ".json")
    backup.write_text(json.dumps(old, indent=2) + "\n")
    billing_run = old.get("billing_run_id", old["run_id"])
    new = {
        **old,
        "run_id": run_id,
        "parent_session_id": old.get("parent_session_id", old["run_id"]),
        "previous_session_ids": list(
            dict.fromkeys([old["run_id"], *old.get("previous_session_ids", [])])
        ),
        "billing_run_id": billing_run,
        "reserved_upper_bound_usd": 0,
        "status": "ready_left_warm_at_user_request",
        "startup_seconds": value["startup_seconds"],
        "worker_pid": value["worker_pid"],
        "language_mode": "auto_from_structured_hyde",
        "reason": "Controller-only update reusing the same warm GPU/vLLM/DB and original bounded allocation",
    }
    job_path.write_text(json.dumps(new, indent=2) + "\n")
    ledger_path = ROOT / "data/modal-compute-budget.json"
    ledger = json.loads(ledger_path.read_text())
    for row in ledger["runs"]:
        if row.get("run_id") == billing_run:
            row.setdefault("controller_updates", []).append(
                {
                    "run_id": run_id,
                    "reserved_upper_bound_usd": 0,
                    "container_id": value["container_id"],
                    "startup_seconds": value["startup_seconds"],
                }
            )
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "container_id": value["container_id"],
                "startup_seconds": value["startup_seconds"],
                "deadline_unix": old["deadline_unix"],
                "gpu": "H200" if old["gpu"] == "H200" else "H100",
            }
        )
    )


if __name__ == "__main__":
    main()
