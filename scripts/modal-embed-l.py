"""Embed full L with bounded, verified, resumable float32 shards.

Run: .venv/bin/modal run --detach scripts/modal-embed-l.py
Resume (explicit new bounded invocation): add --resume-run-id <existing run id>.
No training and no MariaDB import are performed by this script.
"""

import hashlib
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
SOURCE_SHA = "05d5520fc272b0f879bdf62ea67e8bf853c55ce5975eb5393f20c78c086158ec"
BASE_CONFIG = {
    "model": MODEL,
    "revision": REVISION,
    "release": "l-2026-10",
    "source_sha256": SOURCE_SHA,
    "population": 568416,
    "input_format": "title + two newlines + text",
    "max_tokens": 8192,
    "pooling": "normalized CLS",
    "precision": "float16",
    "attention": "sdpa",
    "max_batch": 32,
    "padded_token_budget": 16384,
    "shard_size": 10000,
}
app = modal.App("wikilens-full-l-embeddings")
volume = modal.Volume.from_name("wikilens-l-embedding-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.6.0",
        "transformers==4.57.6",
        "numpy==2.5.4",
        "pyarrow==26.0.1",
        "requests==2.34.2",
        "huggingface-hub==0.36.2",
    )
    .env({"HF_HOME": "/data/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("wikilens")
)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=4,
    memory=16384,
    timeout=1200,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    env={"TOKENIZERS_PARALLELISM": "true", "RAYON_NUM_THREADS": "4"},
)
def prepare(run_id, config, provenance):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from wikilens.corpus_embeddings import digest_config

    start = time.perf_counter()
    volume.reload()
    source = Path("/data/l-2026-10/articles-fi.parquet")
    if file_sha(source) != SOURCE_SHA:
        raise ValueError("Cached official L source checksum mismatch")
    if pq.ParquetFile(source).metadata.num_rows != config["population"]:
        raise ValueError("Unexpected source population")
    directory = Path("/data") / run_id
    directory.mkdir(exist_ok=True)
    config_hash = digest_config(config)
    manifest_path = directory / "inputs.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old and old["config_hash"] != config_hash:
        raise ValueError("Resume configuration differs from original run")
    manifest = old or {
        "run_id": run_id,
        "config": config,
        "config_hash": config_hash,
        "provenance": provenance,
        "complete": False,
        "shards": [],
    }
    model_path = f"/data/hf/hub/models--BAAI--bge-m3/snapshots/{REVISION}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    source_row = 0
    known = {s["index"]: s for s in manifest["shards"]}
    for index, batch in enumerate(
        pq.ParquetFile(source).iter_batches(batch_size=config["shard_size"])
    ):
        path = directory / f"inputs-{index:05}.parquet"
        if index in known:
            if file_sha(path) != known[index]["sha256"]:
                raise ValueError("Previously saved token shard checksum mismatch")
            source_row += batch.num_rows
            continue
        rows = batch.to_pylist()
        texts = [f"{r['title']}\n\n{r['text']}" for r in rows]
        encoded = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=8192,
            return_attention_mask=False,
        )["input_ids"]
        candidates = [i for i, ids in enumerate(encoded) if len(ids) == 8192]
        truncated = set()
        if candidates:
            probe = tokenizer(
                [texts[i] for i in candidates],
                padding=False,
                truncation=True,
                max_length=8193,
                return_attention_mask=False,
            )["input_ids"]
            truncated = {
                i for i, ids in zip(candidates, probe, strict=True) if len(ids) > 8192
            }
        packed = [
            {
                "lang": r["lang"],
                "page_id": r["page_id"],
                "rev_id": r["rev_id"],
                "source_row": source_row + i,
                "truncated": i in truncated,
                "input_ids": encoded[i],
            }
            for i, r in enumerate(rows)
        ]
        schema = pa.schema(
            [
                ("lang", pa.string()),
                ("page_id", pa.int64()),
                ("rev_id", pa.int64()),
                ("source_row", pa.int64()),
                ("truncated", pa.bool_()),
                ("input_ids", pa.list_(pa.int32())),
            ]
        )
        temp = path.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(packed, schema=schema), temp)
        temp.replace(path)
        info = {
            "index": index,
            "file": path.name,
            "rows": len(rows),
            "first_source_row": source_row,
            "sha256": file_sha(path),
            "truncated": len(truncated),
        }
        manifest["shards"].append(info)
        source_row += len(rows)
        atomic_json(manifest_path, manifest)
        volume.commit()
        print(f"Tokenized {source_row}/{config['population']}", flush=True)
    if source_row != config["population"]:
        raise ValueError("Incomplete tokenization")
    manifest.update(
        complete=True,
        preparation_seconds=time.perf_counter() - start,
        model_path=model_path,
    )
    atomic_json(manifest_path, manifest)
    volume.commit()
    return {
        "shards": len(manifest["shards"]),
        "rows": source_row,
        "preparation_seconds": manifest["preparation_seconds"],
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=7200,
    startup_timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def embed_all(run_id, compatibility):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModel, AutoTokenizer

    from wikilens.corpus_embeddings import validate_vectors, vector_table
    from wikilens.embedding_benchmark import batches_by_tokens

    start = time.perf_counter()
    volume.reload()
    directory = Path("/data") / run_id
    inputs = json.loads((directory / "inputs.json").read_text())
    if not inputs["complete"]:
        raise ValueError("Token preparation must finish before GPU starts")
    config = inputs["config"]
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        inputs["model_path"], local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            inputs["model_path"],
            local_files_only=True,
            use_safetensors=False,
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )

    def encode(token_rows):
        tokens = tokenizer.pad(
            [{"input_ids": ids} for ids in token_rows],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            vectors = torch.nn.functional.normalize(hidden[:, 0].float(), dim=1)
        return vectors.cpu().numpy()

    checks = []
    for row in compatibility:
        ids = tokenizer(
            f"{row['title']}\n\n{row['text']}", truncation=True, max_length=8192
        )["input_ids"]
        v = encode([ids])[0]
        ref = np.asarray(row["embedding"], dtype=np.float32)
        checks.append(
            {
                "lang": row["lang"],
                "page_id": row["page_id"],
                "cosine": float(v @ ref / np.linalg.norm(ref)),
            }
        )
    if len(checks) != 9 or min(r["cosine"] for r in checks) < 0.99:
        raise ValueError("S vector compatibility failed")
    atomic_json(directory / "compatibility.json", checks)
    volume.commit()
    loaded_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    completed = []
    for spec in inputs["shards"]:
        index = spec["index"]
        source = directory / spec["file"]
        if file_sha(source) != spec["sha256"]:
            raise ValueError("Token shard checksum mismatch")
        rows = pq.read_table(source).to_pylist()
        path = directory / f"vectors-{index:05}.parquet"
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            info = json.loads(sidecar.read_text())
            if info["config_hash"] != inputs["config_hash"] or info[
                "sha256"
            ] != file_sha(path):
                raise ValueError("Cannot resume altered vector shard")
            validate_vectors(pq.read_table(path), rows)
            completed.append(info)
            continue
        if time.perf_counter() - start > 6900:
            print(
                "Checkpointed stop before hard timeout; explicit resume required",
                flush=True,
            )
            break
        t = time.perf_counter()
        matrix = np.empty((len(rows), 1024), dtype=np.float32)
        batch_seconds = 0
        for indices in batches_by_tokens([len(r["input_ids"]) for r in rows]):
            b = time.perf_counter()
            matrix[indices] = encode([rows[i]["input_ids"] for i in indices])
            batch_seconds += time.perf_counter() - b
        temp = path.with_suffix(".tmp")
        pq.write_table(vector_table(rows, matrix), temp)
        temp.replace(path)
        verified = validate_vectors(pq.read_table(path), rows)
        info = {
            "index": index,
            "file": path.name,
            "rows": len(rows),
            "config_hash": inputs["config_hash"],
            "sha256": file_sha(path),
            "bytes": path.stat().st_size,
            "truncated": spec["truncated"],
            "batch_seconds": batch_seconds,
            "shard_seconds": time.perf_counter() - t,
            "verification": verified,
        }
        atomic_json(sidecar, info)
        completed.append(info)
        atomic_json(
            directory / "progress.json",
            {
                "run_id": run_id,
                "rows": sum(s["rows"] for s in completed),
                "shards": completed,
            },
        )
        volume.commit()
        print(
            json.dumps(
                {
                    "completed": sum(s["rows"] for s in completed),
                    "total": config["population"],
                    "shard": index,
                    "elapsed_seconds": round(time.perf_counter() - start, 2),
                }
            ),
            flush=True,
        )
    total_rows = sum(s["rows"] for s in completed)
    report = {
        "run_id": run_id,
        "status": "complete" if total_rows == config["population"] else "partial",
        "config": config,
        "config_hash": inputs["config_hash"],
        "provenance": inputs["provenance"],
        "gpu": torch.cuda.get_device_name(),
        "compatibility": checks,
        "rows": total_rows,
        "shards": completed,
        "model_load_and_compatibility_seconds": loaded_seconds,
        "gpu_function_seconds": time.perf_counter() - start,
        "embedding_batch_seconds": sum(s["batch_seconds"] for s in completed),
        "vector_bytes": sum(s["bytes"] for s in completed),
        "truncated": sum(s["truncated"] for s in completed),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "notes": [
            "Pretrained inference only",
            "Transformers FP16 numerically differs from S Ollama",
            "Truncated at 8192 tokens",
            "MariaDB import/index not run",
        ],
    }
    atomic_json(directory / "report.json", report)
    volume.commit()
    return report


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=4,
    memory=16384,
    timeout=1200,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def verify_corpus(run_id):
    import pyarrow.parquet as pq

    from wikilens.corpus_embeddings import validate_vectors

    volume.reload()
    directory = Path("/data") / run_id
    report = json.loads((directory / "report.json").read_text())
    inputs = json.loads((directory / "inputs.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Full corpus is incomplete")
    seen = set()
    expected_row = 0
    for spec, vector_spec in zip(inputs["shards"], report["shards"], strict=True):
        source = directory / spec["file"]
        path = directory / vector_spec["file"]
        if (
            file_sha(source) != spec["sha256"]
            or file_sha(path) != vector_spec["sha256"]
        ):
            raise ValueError("Corpus checksum mismatch")
        rows = pq.read_table(source).to_pylist()
        if [r["source_row"] for r in rows] != list(
            range(expected_row, expected_row + len(rows))
        ):
            raise ValueError("Source rows have gaps or duplicates")
        validate_vectors(pq.read_table(path), rows)
        for row in rows:
            key = (row["lang"], row["page_id"])
            if key in seen:
                raise ValueError("Duplicate article identity")
            seen.add(key)
        expected_row += len(rows)
    if len(seen) != report["config"]["population"] or expected_row != report["rows"]:
        raise ValueError("Population coverage mismatch")
    verification = {
        "run_id": run_id,
        "verified_rows": len(seen),
        "all_shards_verified": True,
        "unique_article_ids": True,
        "contiguous_source_coverage": True,
        "finite_unit_vectors": True,
        "dimensions": 1024,
        "dtype": "float32",
        "status": "verified",
    }
    atomic_json(directory / "verification.json", verification)
    volume.commit()
    return verification


@app.local_entrypoint()
def main(resume_run_id: str = ""):
    import pyarrow.parquet as pq

    from wikilens.corpus_embeddings import digest_config

    run_id = resume_run_id or datetime.now(UTC).strftime("l-full-bge-m3-%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"l-full-bge-m3-\d{8}T\d{6}Z", run_id):
        raise ValueError("Invalid run identifier")
    provenance = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            name: file_sha(ROOT / name)
            for name in (
                "scripts/modal-embed-l.py",
                "wikilens/corpus_embeddings.py",
                "wikilens/embedding_benchmark.py",
                "uv.lock",
            )
        },
    }
    config = BASE_CONFIG | {
        "implementation_sha256": digest_config(provenance["source_sha256"])
    }
    output = ROOT / "data" / "l-embeddings" / run_id
    output.mkdir(parents=True, exist_ok=True)
    compatibility = []
    for lang in ("fi", "en", "sv"):
        source = ROOT / "data" / "s-2026-10"
        refs = {
            r["page_id"]: r
            for r in pq.read_table(
                source / f"vectors-bge-m3-{lang}.parquet"
            ).to_pylist()
        }
        for row in pq.read_table(source / f"articles-{lang}.parquet").to_pylist():
            ref = refs[row["page_id"]]
            if not ref["truncated"] and 200 <= len(row["text"]) <= 900:
                compatibility.append(row | {"embedding": ref["embedding"]})
                if sum(r["lang"] == lang for r in compatibility) == 3:
                    break
    ledger_path = ROOT / "data" / "modal-compute-budget.json"
    ledger = json.loads(ledger_path.read_text())
    if (
        sum(r["reserved_upper_bound_usd"] for r in ledger["runs"]) + 10
        > ledger["credit_allowance_usd"]
    ):
        raise ValueError("Insufficient remaining Modal credit reservation")
    record = {
        "run_id": run_id,
        "app_id": app.app_id,
        "status": "starting",
        "reserved_upper_bound_usd": 10,
        "gpu_timeout_seconds": 7200,
        "cpu_prepare_timeout_seconds": 1200,
        "cpu_verify_timeout_seconds": 1200,
        "attempts": 1,
        "gpu": "H100",
        "purpose": "User-authorized full-L pretrained embeddings",
        "provenance": provenance,
    }
    ledger["runs"].append(record)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    try:
        print("RUN", run_id, flush=True)
        print("PREPARED", prepare.remote(run_id, config, provenance), flush=True)
        record["status"] = "embedding"
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        report = embed_all.remote(run_id, compatibility)
        (ROOT / "reports" / f"{run_id}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if report["status"] != "complete":
            record["status"] = "partial_checkpointed"
            print("PARTIAL", run_id, flush=True)
            return
        verification = verify_corpus.remote(run_id)
        (ROOT / "reports" / f"{run_id}-verification.json").write_text(
            json.dumps(verification, indent=2) + "\n"
        )
        record["status"] = "verified_downloading"
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        for name in (
            "inputs.json",
            "report.json",
            "compatibility.json",
            "verification.json",
        ):
            with (output / name).open("wb") as out:
                for chunk in volume.read_file(f"/{run_id}/{name}"):
                    out.write(chunk)
        for shard in report["shards"]:
            destination = output / shard["file"]
            if destination.exists() and file_sha(destination) == shard["sha256"]:
                continue
            temp = destination.with_suffix(".part")
            with temp.open("wb") as out:
                for chunk in volume.read_file(f"/{run_id}/{shard['file']}"):
                    out.write(chunk)
            if file_sha(temp) != shard["sha256"]:
                raise ValueError("Downloaded vector shard checksum mismatch")
            temp.replace(destination)
            if pq.ParquetFile(destination).metadata.num_rows != shard["rows"]:
                raise ValueError("Downloaded shard row count mismatch")
            print("DOWNLOADED", shard["file"], flush=True)
        record.update(
            status="completed_downloaded_verified",
            local_output=str(output),
            verified_rows=verification["verified_rows"],
            vector_bytes=report["vector_bytes"],
        )
        print("RESULT", str(output), flush=True)
    except BaseException as exc:
        record.update(
            status="failed_checkpoint_preserved", error_type=type(exc).__name__
        )
        raise
    finally:
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
