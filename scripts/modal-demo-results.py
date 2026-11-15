"""One bounded private GPU job restoring the existing full-L snapshot."""

import json
import os
import runpy
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

ROOT = (
    Path(__file__).resolve().parents[1]
    if modal.is_local()
    else Path("/workspace/Junction_2026")
)
base = runpy.run_path(str(ROOT / "scripts/modal-full-l-search.py"))["image"]
image = base.run_commands(
    "cd /workspace/Junction_2026 && uv sync --locked --no-dev --group demo --group fp8 --default-index https://pypi.org/simple"
)
if modal.is_local():
    image = image.env(
        {
            "WIKILENS_SOURCE_COMMIT": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        }
    )
app = modal.App("wikilens-demo-captures")
volume = modal.Volume.from_name("wikilens-l-embedding-benchmark")


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L40S",
    cpu=(4, 4),
    memory=(32768, 32768),
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def capture(run_id, source_run_id, keep_open=False):
    started = time.perf_counter()
    session_deadline = time.time() + 1770
    volume.reload()
    root = Path("/workspace/Junction_2026")
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / ".venv/lib/python3.12/site-packages")]
    out = Path("/data/demo-captures") / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "attempt.json").exists():
        raise RuntimeError("Attempt already exists; no automatic retry")
    (out / "attempt.json").write_text(
        json.dumps(
            {
                "source_run_id": source_run_id,
                "started_at": time.time(),
                "timeout_seconds": 1800,
                "attempts": 1,
            }
        )
        + "\n"
    )
    volume.commit()
    data = root / "data"
    data.mkdir(exist_ok=True)
    for target, source in [
        (data / "l-2026-10", Path("/data/l-2026-10")),
        (data / "l-embeddings", Path("/data")),
        (root / "reports", out / "reports"),
        (data / "gradio-runs", out / "raw-runs"),
    ]:
        source.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
    source = Path("/data/full-l-search") / source_run_id
    if json.loads((source / "import-result.json").read_text())["status"] != "complete":
        raise RuntimeError("Verified database snapshot required")
    with tarfile.open(source / "mariadb-clean.tar") as archive:
        archive.extractall(data, filter="data")
    # The installer still verifies the pinned SHA-256 before using this cache.
    cached_runtime = Path(
        "/data/mariadb-runtime-cache/mariadb-12.3.3-linux-systemd-x86_64.tar.gz"
    )
    if cached_runtime.exists():
        runtime = data / "runtime"
        runtime.mkdir(exist_ok=True)
        (runtime / cached_runtime.name).symlink_to(cached_runtime)
    with (out / "bootstrap.log").open("a") as log:
        subprocess.run(
            [str(root / ".venv/bin/python"), "scripts/setup-remote-mariadb.py"],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    import pymysql

    with (
        pymysql.connect(unix_socket="/tmp/junction-mariadb.sock", user="root") as db,
        db.cursor() as cursor,
    ):
        cursor.execute("SET GLOBAL innodb_buffer_pool_size=4294967296")
        cursor.execute("SET GLOBAL mhnsw_max_cache_size=8589934592")
    (out / "setup.json").write_text(
        json.dumps(
            {"database_and_workspace_setup_seconds": time.perf_counter() - started}
        )
        + "\n"
    )
    volume.commit()
    try:
        with (out / "capture.log").open("a") as log:
            process = subprocess.Popen(
                [
                    str(root / ".venv/bin/python"),
                    "-u",
                    "scripts/run-demo-captures.py",
                    "--output",
                    str(out),
                    "--context",
                    "videopeli",
                    "--keep-open-until",
                    str(session_deadline if keep_open else 0),
                ],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                time.sleep(15)
                volume.commit()
            if process.returncode:
                raise RuntimeError(
                    f"Capture worker exited {process.returncode}; inspect capture.log"
                )
        return json.loads((out / "report.json").read_text())
    finally:
        volume.commit()
