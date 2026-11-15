"""One explicitly started warm H100 session, private queue, two-hour timeout."""

import json
import os
import runpy
import subprocess
import time
from pathlib import Path

import modal

ROOT = (
    Path(__file__).resolve().parents[1]
    if modal.is_local()
    else Path("/workspace/Junction_2026")
)
image = (
    runpy.run_path(str(ROOT / "scripts/modal-vllm-h100.py"))["image"]
    .apt_install("libaio1", "libnuma1", "libcrypt1", "curl")
    .run_commands(
        "cd /workspace/Junction_2026 && python -c \"from wikilens.gradio_demo import ProductEngine; from wikilens.database import connect; from wikilens.paper_live import result_for_paper; print('Live imports OK')\""
    )
)
app = modal.App("wikilens-paper-live")
volume = modal.Volume.from_name("wikilens-l-embedding-benchmark")


@app.function(
    image=image,
    gpu="H100!",
    cpu=(8, 8),
    memory=(32768, 32768),
    volumes={"/data": volume},
    timeout=7200,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def serve(run_id, source_run_id, deadline_unix):
    volume.reload()
    root = Path("/workspace/Junction_2026")
    out = Path("/data/paper-live") / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "attempt.json").exists():
        raise RuntimeError("Session already attempted; no automatic retry")
    (out / "attempt.json").write_text(
        json.dumps(
            {
                "started_at_unix": time.time(),
                "deadline_unix": deadline_unix,
                "timeout_seconds": 7200,
                "attempts": 1,
            }
        )
        + "\n"
    )
    volume.commit()
    executable = root / "inference/vllm/.venv/bin/python"
    env = dict(os.environ, PYTHONPATH=str(root))
    try:
        with (out / "worker.log").open("a") as log:
            worker = subprocess.Popen(
                [
                    str(executable),
                    "-u",
                    str(root / "scripts/run-paper-live.py"),
                    "--output",
                    str(out),
                    "--source-run",
                    source_run_id,
                    "--deadline",
                    str(deadline_unix),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while worker.poll() is None:
                time.sleep(10)
                volume.commit()
            if worker.returncode:
                raise RuntimeError(
                    f"Live worker exited {worker.returncode}; inspect worker.log"
                )
        return {"status": "closed", "run_id": run_id}
    finally:
        volume.commit()
