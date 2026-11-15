"""Public CPU frontend for an explicitly reserved six-hour H200 session."""

import json
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("wikilens-public-demo")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("modal==1.6.4", "pydantic==2.14.3")
    .add_local_dir(
        ROOT / "wikilens", remote_path="/app/wikilens", ignore=["__pycache__", "*.pyc"]
    )
)
# Only the session identifier, deadline and allocation metadata are embedded. No credentials.
job = (
    json.loads((ROOT / "data/modal-public-demo-job.json").read_text())
    if modal.is_local()
    else {}
)
public_origin = os.environ.get("WIKILENS_PUBLIC_ORIGIN", "")
if modal.is_local() and not public_origin:
    raise RuntimeError("Set WIKILENS_PUBLIC_ORIGIN to the deployed HTTPS origin")


@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    timeout=300,
    max_containers=1,
    scaledown_window=30,
    env={
        "WIKILENS_PUBLIC_JOB": json.dumps(job),
        "WIKILENS_PUBLIC_ORIGIN": public_origin,
        "PYTHONPATH": "/app",
    },
)
@modal.concurrent(max_inputs=20)
@modal.web_server(8000, startup_timeout=30, label="wikilens")
def web():
    subprocess.Popen(["python", "-m", "wikilens.public_demo"], cwd="/app")
