"""Resident vLLM + BGE + verified Dataset L DB; one attempt per question."""

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-run", required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--reuse-services", action="store_true")
    p.add_argument("--parent-session")
    args = p.parse_args()
    import modal
    import requests

    from wikilens.paper_live import QUEUE, RESULTS, result_for_paper
    from wikilens.torch_runtime import QWEN4_FP8_MODEL, QWEN4_FP8_REVISION
    from wikilens.vllm_runtime import VLLMRuntime

    root = Path.cwd()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    results = modal.Dict.from_name(RESULTS, create_if_missing=True)
    inbox = modal.Queue.from_name(QUEUE, create_if_missing=True)
    lock = threading.Lock()
    state = {
        "status": "starting",
        "stage": "Restoring Dataset L database",
        "session_id": out.name,
        "deadline_unix": args.deadline,
        "accepts_requests_until_unix": args.deadline - 900,
        "worker_pid": os.getpid(),
    }
    done = threading.Event()

    def status(**values):
        with lock:
            state.update(values, heartbeat_unix=time.time())
            results[out.name] = dict(state)
            save(out / "session.json", state)

    def heartbeat():
        while not done.wait(10):
            status()

    status()
    threading.Thread(target=heartbeat, daemon=True).start()
    server = engine = None
    started = time.perf_counter()
    try:
        hardware = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        if "H100" not in hardware or len(hardware.splitlines()) != 1:
            raise RuntimeError("Expected exactly one H100: " + hardware)
        save(
            out / "provenance.json",
            {
                "hardware": hardware,
                "source_run": args.source_run,
                "qwen": QWEN4_FP8_MODEL,
                "revision": QWEN4_FP8_REVISION,
                "source_sha256": {
                    str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in [
                        *Path("wikilens").glob("*.py"),
                        Path(__file__),
                        Path("inference/vllm/uv.lock"),
                    ]
                },
            },
        )
        if args.reuse_services:
            status(stage="Connecting to the existing warm vLLM server and database")
            requests.get("http://127.0.0.1:8000/health", timeout=5).raise_for_status()
        else:
            data = root / "data"
            data.mkdir(exist_ok=True)
            for target, source in [
                (data / "l-2026-10", Path("/data/l-2026-10")),
                (data / "l-embeddings", Path("/data")),
                (data / "gradio-runs", out / "raw-runs"),
            ]:
                source.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source, target_is_directory=True)
            source = Path("/data/full-l-search") / args.source_run
            if (
                json.loads((source / "import-result.json").read_text())["status"]
                != "complete"
            ):
                raise RuntimeError("Completed Dataset L snapshot required")
            with tarfile.open(source / "mariadb-clean.tar") as archive:
                archive.extractall(data, filter="data")
            runtime = data / "runtime"
            runtime.mkdir(exist_ok=True)
            archive = Path(
                "/data/mariadb-runtime-cache/mariadb-12.3.3-linux-systemd-x86_64.tar.gz"
            )
            if archive.exists():
                (runtime / archive.name).symlink_to(archive)
            subprocess.run(
                [sys.executable, "scripts/setup-remote-mariadb.py"], check=True
            )
            import pymysql

            with (
                pymysql.connect(
                    unix_socket="/tmp/junction-mariadb.sock", user="root"
                ) as db,
                db.cursor() as c,
            ):
                c.execute("SET GLOBAL innodb_buffer_pool_size=4294967296")
                c.execute("SET GLOBAL mhnsw_max_cache_size=8589934592")
            status(stage="Loading and warming the H100 model")
            command = [
                str(Path(sys.executable).parent / "vllm"),
                "serve",
                QWEN4_FP8_MODEL,
                "--revision",
                QWEN4_FP8_REVISION,
                "--tokenizer-revision",
                QWEN4_FP8_REVISION,
                "--dtype",
                "auto",
                "--max-model-len",
                "12288",
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                "0.7",
                "--max-num-seqs",
                "4",
                "--max-num-batched-tokens",
                "12288",
                "--no-enable-prefix-caching",
                "--generation-config",
                "vllm",
                "--seed",
                "42",
                "--chat-template-content-format",
                "string",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ]
            save(out / "server-config.json", {"command": command})
            with (out / "server.log").open("a") as log:
                server = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            launch = time.perf_counter()
            while time.perf_counter() - launch < 600:
                if server.poll() is not None:
                    raise RuntimeError("vLLM server stopped during startup")
                try:
                    if (
                        requests.get(
                            "http://127.0.0.1:8000/health", timeout=2
                        ).status_code
                        == 200
                    ):
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError("vLLM startup exceeded 600 seconds; no retry")

        class LiveModels(VLLMRuntime):
            def __init__(self, revision, context_tokens, qwen_model):
                assert revision == QWEN4_FP8_REVISION and qwen_model == QWEN4_FP8_MODEL
                super().__init__()
                self.load_references()
                from transformers import AutoConfig

                from wikilens.torch_runtime import _TransformersBackend

                self.model = SimpleNamespace(
                    config=AutoConfig.from_pretrained(
                        qwen_model, revision=revision, local_files_only=True
                    )
                )
                self.backend = _TransformersBackend("cuda", qwen_model=qwen_model)

        status(stage="Loading the resident embedding model and reranker")
        from wikilens.gradio_demo import ProductEngine

        engine = ProductEngine(runtime_factory=LiveModels)
        container_id = os.environ.get("MODAL_TASK_ID", "see provider task inventory")
        status(
            status="ready",
            stage="Ready",
            startup_seconds=time.perf_counter() - started,
            container_id=container_id,
            gpu=hardware,
            corpus=engine.corpus_info,
            runtime=engine.models.stats(),
        )
        while time.time() < args.deadline:
            try:
                request = inbox.get(
                    partition=out.name,
                    timeout=min(5, max(0.01, args.deadline - time.time())),
                )
            except queue.Empty:
                continue
            if request.get("action") == "stop":
                break
            rid = request.get("request_id", "")
            if not re.fullmatch("[a-f0-9]{32}", rid):
                continue
            key = out.name + ":" + rid
            destination = out / "requests" / (rid + ".json")
            if destination.exists():
                continue
            save(
                destination,
                {
                    "status": "started",
                    "request": request,
                    "started_at_unix": time.time(),
                },
            )
            base = {**request, "worker_pid": os.getpid(), "container_id": container_id}
            if time.time() > args.deadline - 900:
                results[key] = {
                    **base,
                    "status": "failed",
                    "error": "Warm session is ending",
                }
                continue
            before = time.perf_counter()
            status(status="answering", stage="Researching", current_request=rid)
            inspector = None
            try:
                from wikilens.admin_tools import PaperSettings, RetrievalInspector

                controls = PaperSettings.model_validate(
                    request.get("settings", {})
                ).model_dump()
                inspector = RetrievalInspector(
                    request["question"], request.get("context", ""), controls
                )

                def inspect_retrieval(
                    stage, data, key=key, inspector=inspector, destination=destination
                ):
                    snapshot = inspector.capture(stage, data)
                    items = list(inspector.documents.items())
                    for offset in range(0, len(items), 8):
                        results.update(
                            {
                                key + ":document:" + did: document
                                for did, document in items[offset : offset + 8]
                            }
                        )
                    results[key + ":diagnostics"] = snapshot
                    save(
                        destination.with_suffix(".diagnostics.json"),
                        {"summary": snapshot, "documents": inspector.documents},
                    )

                def progress(value, desc="", key=key, base=base, before=before):
                    results[key] = {
                        **base,
                        "status": "running",
                        "progress": value,
                        "stage": desc,
                        "elapsed_seconds": time.perf_counter() - before,
                    }

                result = engine.run(
                    request["question"],
                    "Auto",
                    "Finnish",
                    request.get("context", ""),
                    controls["retrieval_top_k"],
                    500,
                    controls["article_tokens"],
                    progress,
                    min_score=controls["score_cutoff"],
                    relative_gap=controls["relative_gap"],
                    max_nodes=controls["max_nodes"],
                    diagnostics=inspect_retrieval,
                )
                save(destination, result)
                results[key] = {
                    **base,
                    "status": "complete",
                    "result": result_for_paper(result),
                    "seconds": time.perf_counter() - before,
                }
            except Exception as error:
                failure = {
                    **base,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                save(destination.with_suffix(".failure.json"), failure)
                results[key] = failure
                if inspector is not None:
                    inspector.snapshot["error"] = failure["error"]
                    results[key + ":diagnostics"] = inspector.snapshot
                    save(
                        destination.with_suffix(".diagnostics.json"),
                        {
                            "summary": inspector.snapshot,
                            "documents": inspector.documents,
                        },
                    )
                if engine.models.dead:
                    raise RuntimeError(
                        "Inference server failed; refusing automatic restart"
                    ) from error
            status(
                status="ready", stage="Ready", current_request=None, last_request_id=rid
            )
            # Raw generation calls have been saved; avoid accumulating every prompt in RAM.
            engine.models.profiles.clear()
            engine.models.lookup_profiles.clear()
        status(status="closed", stage="Warm session ended")
    except Exception as error:
        status(
            status="failed",
            stage="Session stopped",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        done.set()
        if args.parent_session and state["status"] == "closed":
            inbox.put({"action": "stop"}, partition=args.parent_session)
        if engine is not None:
            engine.close()
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


if __name__ == "__main__":
    main()
