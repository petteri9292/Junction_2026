"""Bounded pretrained BGE-M3 benchmark. Run with the project's Modal CLI.

    .venv/bin/modal run scripts/modal-embedding-benchmark.py

Only this subset runs. It never launches the full corpus automatically.
"""

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RELEASE = "l-2026-10"
SEED = 20261114
app = modal.App("wikilens-l-embedding-benchmark")
volume = modal.Volume.from_name(
    "wikilens-l-embedding-benchmark", create_if_missing=True
)
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
def prepare(run_id, sample_size):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import requests
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from wikilens.embedding_benchmark import sample_indices

    start = time.perf_counter()
    directory = Path("/data") / run_id
    directory.mkdir()
    source = Path("/data") / RELEASE
    source.mkdir(exist_ok=True)
    base = f"https://github.com/MariaDB/mariadb-wikipedia-datasets/releases/download/{RELEASE}"
    response = requests.get(f"{base}/SHA256SUMS", timeout=60)
    response.raise_for_status()
    (directory / "SHA256SUMS").write_text(response.text)
    sums = {
        line.split()[1].lstrip("*"): line.split()[0]
        for line in response.text.splitlines()
        if line.strip()
    }
    path = source / "articles-fi.parquet"
    if not path.exists():
        temp = path.with_suffix(".part")
        with requests.get(f"{base}/{path.name}", stream=True, timeout=120) as remote:
            remote.raise_for_status()
            with temp.open("wb") as out:
                for chunk in remote.iter_content(1024 * 1024):
                    out.write(chunk)
        temp.replace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != sums[path.name]:
        raise ValueError("L checksum mismatch")
    print("L checksum verified", flush=True)
    table = pq.read_table(path, columns=["lang", "page_id", "rev_id", "title", "text"])
    lengths = (
        np.asarray(pc.utf8_length(table["title"]))
        + 2
        + np.asarray(pc.utf8_length(table["text"]))
    )
    selected, strata = sample_indices(lengths, sample_size, SEED)
    sample = table.take(pa.array([i for i, _, _ in selected])).to_pylist()
    del table
    model_path = snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",
            "tokenizer*",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for row, (index, stratum, stress) in zip(sample, selected, strict=True):
        row.update(source_row=index, stratum=stratum, stress=stress)
        row["tokens"] = len(
            tokenizer(
                f"{row['title']}\n\n{row['text']}", truncation=False, verbose=False
            )["input_ids"]
        )
    pq.write_table(pa.Table.from_pylist(sample), directory / "sample.parquet")
    metadata = {
        "run_id": run_id,
        "release": RELEASE,
        "sha256": digest,
        "seed": SEED,
        "population": len(lengths),
        "sample_size": sample_size,
        "strata": strata,
        "model": MODEL,
        "model_revision": REVISION,
        "model_path": model_path,
        "sample_total": len(sample),
        "preparation_seconds": time.perf_counter() - start,
        "length_percentiles_chars": {
            str(p): float(np.percentile(lengths, p)) for p in (0, 50, 90, 95, 99, 100)
        },
    }
    (directory / "preparation.json").write_text(json.dumps(metadata, indent=2))
    volume.commit()
    print(json.dumps(metadata), flush=True)
    return metadata


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def benchmark(run_id, compatibility):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModel, AutoTokenizer

    from wikilens.embedding_benchmark import batches_by_tokens

    start = time.perf_counter()
    volume.reload()
    directory = Path("/data") / run_id
    metadata = json.loads((directory / "preparation.json").read_text())
    rows = pq.read_table(directory / "sample.parquet").to_pylist()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["model_path"], local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            metadata["model_path"],
            local_files_only=True,
            use_safetensors=False,
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    load_seconds = time.perf_counter() - start

    def embed(texts):
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            vectors = torch.nn.functional.normalize(hidden[:, 0].float(), dim=1)
        result = vectors.cpu().numpy()
        if result.shape != (len(texts), 1024) or not np.isfinite(result).all():
            raise ValueError("Invalid embedding")
        return result

    checks = []
    for row in compatibility:
        vector = embed([f"{row['title']}\n\n{row['text']}"])[0]
        reference = np.asarray(row["embedding"], dtype=np.float32)
        cosine = float(vector @ reference / np.linalg.norm(reference))
        checks.append(
            {
                "lang": row["lang"],
                "page_id": row["page_id"],
                "title": row["title"],
                "cosine": cosine,
            }
        )
    (directory / "compatibility.json").write_text(json.dumps(checks, indent=2))
    volume.commit()
    if min(c["cosine"] for c in checks) < 0.99:
        raise ValueError("S vector compatibility cosine below 0.99; stopping")
    # Compatibility calls warm the model; exclude them from steady-state time.
    print("Compatibility passed", json.dumps(checks), flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    embedding_start = time.perf_counter()
    records, batch_metrics = [], []
    for batch_index, indices in enumerate(
        batches_by_tokens([min(r["tokens"], 8192) for r in rows])
    ):
        selected = [rows[i] for i in indices]
        t = time.perf_counter()
        vectors = embed([f"{r['title']}\n\n{r['text']}" for r in selected])
        elapsed = time.perf_counter() - t
        batch_metrics.append(
            {
                "batch": batch_index,
                "size": len(selected),
                "seconds": elapsed,
                "max_tokens": max(min(r["tokens"], 8192) for r in selected),
            }
        )
        for row, vector in zip(selected, vectors, strict=True):
            records.append(
                {
                    k: row[k]
                    for k in (
                        "lang",
                        "page_id",
                        "rev_id",
                        "stratum",
                        "stress",
                        "tokens",
                    )
                }
                | {
                    "model": "bge-m3",
                    "truncated": row["tokens"] > 8192,
                    "embedding": vector.tolist(),
                    "allocated_batch_seconds": elapsed / len(selected),
                }
            )
        if batch_index % 20 == 0:
            pq.write_table(
                pa.Table.from_pylist(records), directory / "vectors-checkpoint.parquet"
            )
            volume.commit()
            print(f"Embedded {len(records)}/{len(rows)}", flush=True)
    embedding_seconds = time.perf_counter() - embedding_start
    pq.write_table(pa.Table.from_pylist(records), directory / "vectors.parquet")
    projected_seconds = 0
    estimates = []
    for stratum in metadata["strata"]:
        members = [
            r for r in records if r["stratum"] == stratum["stratum"] and not r["stress"]
        ]
        costs = [r["allocated_batch_seconds"] for r in members]
        estimate = stratum["population"] * float(np.mean(costs))
        projected_seconds += estimate
        estimates.append(
            stratum
            | {"mean_seconds": float(np.mean(costs)), "projected_seconds": estimate}
        )
    report = metadata | {
        "status": "completed",
        "gpu": torch.cuda.get_device_name(),
        "precision": "float16",
        "pooling": "normalized CLS",
        "attention": "sdpa",
        "max_tokens": 8192,
        "padded_token_budget": 16384,
        "max_batch": 32,
        "compatibility": checks,
        "model_load_seconds": load_seconds,
        "batches": batch_metrics,
        "embedding_seconds_including_checkpoints": embedding_seconds,
        "embedding_batch_seconds": sum(b["seconds"] for b in batch_metrics),
        "documents_per_second": len(records) / embedding_seconds,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "sampled_truncated": sum(r["truncated"] for r in records if not r["stress"]),
        "stratum_estimates": estimates,
        "projected_full_embedding_hours": projected_seconds / 3600,
        "gpu_function_seconds_before_final_commit": time.perf_counter() - start,
        "limitations": [
            "Projection excludes startup, full-corpus IO, checkpoints and MariaDB index",
            "Character-length stratification; 200 observations per decile",
            "Batch time allocated equally to members; not isolated article latency",
            "Transformers FP16 differs numerically from S's Ollama runtime",
            "Single-GPU measurement does not establish multi-GPU scaling",
        ],
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "documents_per_second",
                    "projected_full_embedding_hours",
                    "peak_gpu_allocated_gib",
                )
            }
        ),
        flush=True,
    )
    return report


@app.local_entrypoint()
def main(sample_size: int = 2000):
    import pyarrow.parquet as pq

    if not 100 <= sample_size <= 2000:
        raise ValueError("Subset only: 100–2000 articles")
    run_id = datetime.now(UTC).strftime("l-bge-m3-%Y%m%dT%H%M%SZ")
    provenance = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/modal-embedding-benchmark.py",
                "wikilens/embedding_benchmark.py",
                "uv.lock",
            )
        },
    }
    output = ROOT / "data" / "modal-benchmarks" / run_id
    output.mkdir(parents=True)
    compatibility = []
    for lang in ("fi", "en", "sv"):
        source = ROOT / "data" / "s-2026-10"
        vectors = {
            r["page_id"]: r
            for r in pq.read_table(
                source / f"vectors-bge-m3-{lang}.parquet"
            ).to_pylist()
        }
        for row in pq.read_table(source / f"articles-{lang}.parquet").to_pylist():
            reference = vectors[row["page_id"]]
            if not reference["truncated"] and 200 <= len(row["text"]) <= 900:
                compatibility.append(row | {"embedding": reference["embedding"]})
                if sum(r["lang"] == lang for r in compatibility) == 3:
                    break
    ledger_path = ROOT / "data" / "modal-compute-budget.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {
            "credit_allowance_usd": 100,
            "source": "User-confirmed Modal promotional credits",
            "runs": [],
        }
    )
    if (
        sum(r["reserved_upper_bound_usd"] for r in ledger["runs"]) + 3
        > ledger["credit_allowance_usd"]
    ):
        raise ValueError("Modal credit allowance exhausted")
    record = {
        "run_id": run_id,
        "status": "starting",
        "reserved_upper_bound_usd": 3,
        "cpu_timeout_seconds": 1200,
        "gpu_timeout_seconds": 1800,
        "attempts": 1,
        "gpu": "H100",
        "sample_size": sample_size,
        "app_id": app.app_id,
        "purpose": "Pretrained embedding inference benchmark; no training",
        "provenance": provenance,
    }
    ledger["runs"].append(record)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    try:
        prepare.remote(run_id, sample_size)
        report = benchmark.remote(run_id, compatibility)
        for name in (
            "preparation.json",
            "compatibility.json",
            "sample.parquet",
            "vectors.parquet",
            "report.json",
        ):
            with (output / name).open("wb") as stream:
                for chunk in volume.read_file(f"/{run_id}/{name}"):
                    stream.write(chunk)
        (ROOT / "reports" / f"{run_id}.json").write_text(
            json.dumps(report | {"provenance": provenance}, indent=2) + "\n"
        )
        record["status"] = "completed"
        record["local_output"] = str(output)
        print("RESULT", output, flush=True)
    except BaseException as exc:
        record["status"] = "failed"
        record["error_type"] = type(exc).__name__
        raise
    finally:
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
