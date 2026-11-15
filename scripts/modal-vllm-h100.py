"""One bounded H100/vLLM comparison on existing saved passages; no database job."""

import json
import os
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
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("uv")
    .add_local_file(
        ROOT / "inference/vllm/pyproject.toml",
        "/workspace/Junction_2026/inference/vllm/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        ROOT / "inference/vllm/uv.lock",
        "/workspace/Junction_2026/inference/vllm/uv.lock",
        copy=True,
    )
    .run_commands(
        "cd /workspace/Junction_2026/inference/vllm && uv sync --locked --no-dev --default-index https://pypi.org/simple"
    )
    .env(
        {
            "PATH": "/workspace/Junction_2026/inference/vllm/.venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "FLASHINFER_CUDA_ARCH_LIST": "9.0a",
        }
    )
    .run_commands(
        "ninja --version",
        'python -c "from flashinfer.jit.sampling import gen_sampling_module; gen_sampling_module().build(verbose=True)"',
    )
    .add_local_dir(
        ROOT / "wikilens",
        "/workspace/Junction_2026/wikilens",
        copy=True,
        ignore=["**/__pycache__/**"],
    )
    .add_local_dir(
        ROOT / "scripts",
        "/workspace/Junction_2026/scripts",
        copy=True,
        ignore=["**/__pycache__/**"],
    )
    .env(
        {
            "HF_HOME": "/data/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_NO_USAGE_STATS": "1",
            "OMP_NUM_THREADS": "8",
        }
    )
)
if modal.is_local():
    image = image.env(
        {
            "WIKILENS_SOURCE_COMMIT": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        }
    )
app = modal.App("wikilens-vllm-h100-benchmark")
volume = modal.Volume.from_name("wikilens-l-embedding-benchmark")


@app.function(
    image=image,
    gpu="H100!",
    cpu=(8, 8),
    memory=(32768, 32768),
    volumes={"/data": volume},
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def capture(run_id, source_run_id):
    volume.reload()
    root = Path("/workspace/Junction_2026")
    os.chdir(root)
    out = Path("/data/demo-captures") / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "attempt.json").exists():
        raise RuntimeError("Attempt already exists; refusing automatic retry")
    (out / "attempt.json").write_text(
        json.dumps(
            {
                "started_at_unix": time.time(),
                "timeout_seconds": 900,
                "attempts": 1,
                "baseline_run": source_run_id,
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
                    str(root / "scripts/run-vllm-comparison.py"),
                    "--output",
                    str(out),
                    "--baseline",
                    str(Path("/data/demo-captures") / source_run_id),
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
                    f"Comparison worker exited {worker.returncode}; inspect worker.log/server.log"
                )
        return json.loads((out / "report.json").read_text())
    finally:
        volume.commit()
