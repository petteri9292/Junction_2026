"""Bounded full-L MariaDB import on CPU, then separately invoked GPU search study.

Local database files use ephemeral disk; only clean stopped snapshots go to the
Volume. No training, no application retries, no GPU allocation during import.
"""

import json, os, runpy, subprocess, sys, tarfile, time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
APP = "wikilens-full-l-search"
app = modal.App(APP)
volume = modal.Volume.from_name("wikilens-l-embedding-benchmark")
base = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.6.0")
image = (
    base.apt_install("libaio1t64", "libnuma1", "libcrypt1", "curl", "ca-certificates")
    .pip_install("uv")
    .add_local_file(
        ROOT / "pyproject.toml", "/workspace/Junction_2026/pyproject.toml", copy=True
    )
    .add_local_file(ROOT / "uv.lock", "/workspace/Junction_2026/uv.lock", copy=True)
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
    .run_commands(
        "cd /workspace/Junction_2026 && uv venv --system-site-packages .venv && uv sync --locked --no-dev --default-index https://pypi.org/simple -v"
    )
    .env({"HF_HOME": "/data/hf", "TOKENIZERS_PARALLELISM": "false"})
)


def save(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def workspace(run_id, config):
    root = Path("/workspace/Junction_2026")
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / ".venv/lib/python3.12/site-packages"))
    out = Path("/data/full-l-search") / run_id
    out.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    data.mkdir(exist_ok=True)
    for target, source in [
        (data / "l-2026-10", Path("/data/l-2026-10")),
        (data / "l-embeddings", Path("/data")),
        (root / "reports", out / "reports"),
    ]:
        source.mkdir(exist_ok=True, parents=True)
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
    cfg = out / "questions.json"
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError("Run config changed")
    save(cfg, config)
    return root, out, cfg


def command(root, out, args, log):
    with (out / log).open("a") as stream:
        subprocess.run(
            [str(root / ".venv/bin/python"), "-u", *args],
            cwd=root,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def stop_db(root):
    import pymysql

    with (
        pymysql.connect(unix_socket="/tmp/junction-mariadb.sock", user="root") as db,
        db.cursor() as c,
    ):
        c.execute("SHUTDOWN")
    pid = (
        int((root / "data/mariadb/server.pid").read_text())
        if (root / "data/mariadb/server.pid").exists()
        else None
    )
    import psutil

    for _ in range(100):
        if (
            pid is None
            or not psutil.pid_exists(pid)
            or psutil.Process(pid).status() == "zombie"
        ):
            return
        time.sleep(0.1)
    raise RuntimeError("MariaDB did not stop cleanly")


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(8, 8),
    memory=(32768, 32768),
    timeout=7200,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def import_full(run_id, config):
    volume.reload()
    root, out, cfg = workspace(run_id, config)
    marker = out / "import-attempt.json"
    if marker.exists():
        raise RuntimeError(
            "Import attempt already recorded; inspect saved state before an explicit retry"
        )
    save(
        marker,
        {
            "started_at": time.time(),
            "cpu_physical": 8,
            "memory_mib": 32768,
            "timeout_seconds": 7200,
            "attempts": 1,
        },
    )
    volume.commit()
    started = time.perf_counter()
    try:
        command(root, out, ["scripts/setup-remote-mariadb.py"], "setup.log")
        import pymysql
        from pymysql.cursors import DictCursor

        with (
            pymysql.connect(
                unix_socket="/tmp/junction-mariadb.sock",
                user="root",
                cursorclass=DictCursor,
            ) as db,
            db.cursor() as c,
        ):
            c.execute("SET GLOBAL innodb_buffer_pool_size=8589934592")
            c.execute("SET GLOBAL mhnsw_max_cache_size=8589934592")
            c.execute(
                "SHOW GLOBAL VARIABLES WHERE Variable_name IN ('innodb_buffer_pool_size','mhnsw_max_cache_size')"
            )
            settings = c.fetchall()
        command(
            root,
            out,
            [
                "scripts/import-l-embeddings.py",
                "--batch-size",
                "1000",
                "--articles",
                "data/l-2026-10/articles-fi.parquet",
                "--vectors",
                "data/l-embeddings/l-full-bge-m3-20261114T123013Z",
            ],
            "import.log",
        )
        stop_db(root)
        with tarfile.open(out / "mariadb-clean.tar", "w") as tar:
            tar.add(root / "data/mariadb", arcname="mariadb")
        result = {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "settings": settings,
            "snapshot_bytes": (out / "mariadb-clean.tar").stat().st_size,
        }
        save(out / "import-result.json", result)
        volume.commit()
        return result
    except BaseException as error:
        save(
            out / "import-failure.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        volume.commit()
        raise


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",
    cpu=(4, 4),
    memory=(32768, 32768),
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def search(run_id, config, split):
    volume.reload()
    root, out, cfg = workspace(run_id, config)
    if json.loads((out / "import-result.json").read_text())["status"] != "complete":
        raise RuntimeError("Full import required")
    marker = out / f"{split}-attempt.json"
    if marker.exists():
        raise RuntimeError(
            "Search attempt already recorded; inspect before explicit retry"
        )
    save(
        marker,
        {
            "started_at": time.time(),
            "split": split,
            "gpu": "T4",
            "cpu": 4,
            "memory_mib": 32768,
            "timeout_seconds": 1800,
            "attempts": 1,
        },
    )
    volume.commit()
    with tarfile.open(out / "mariadb-clean.tar") as tar:
        tar.extractall(root / "data", filter="data")
    command(root, out, ["scripts/setup-remote-mariadb.py"], "gpu-db-setup.log")
    import pymysql
    from pymysql.cursors import DictCursor

    with (
        pymysql.connect(
            unix_socket="/tmp/junction-mariadb.sock",
            user="root",
            cursorclass=DictCursor,
        ) as db,
        db.cursor() as c,
    ):
        c.execute("SET GLOBAL innodb_buffer_pool_size=4294967296")
        c.execute("SET GLOBAL mhnsw_max_cache_size=8589934592")
    try:
        command(
            root,
            out,
            [
                "scripts/run-reranker-cutoff.py",
                "--config",
                str(cfg),
                "--output",
                str(out / "captures"),
                "--split",
                split,
            ],
            f"{split}-search.log",
        )
        result = json.loads((out / "captures" / f"report-{split}.json").read_text())
        return {"queries": len(result["queries"]), "failures": result["failures"]}
    finally:
        volume.commit()
