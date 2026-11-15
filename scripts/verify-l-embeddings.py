"""Reopen and independently verify every downloaded full-L vector shard."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    directory = ROOT / "data" / "l-embeddings" / args.run_id
    report = json.loads((directory / "report.json").read_text())
    remote_verification = json.loads((directory / "verification.json").read_text())
    if report["status"] != "complete" or remote_verification["status"] != "verified":
        raise ValueError("Remote full-corpus verification did not complete")
    seen = set()
    expected_source_row = 0
    total_bytes = 0
    truncated = 0
    norm_min, norm_max = float("inf"), 0.0
    for spec in report["shards"]:
        path = directory / spec["file"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != spec["sha256"]:
                raise ValueError(f"Checksum mismatch: {path.name}")
        table = pq.read_table(path)
        if table.num_rows != spec["rows"]:
            raise ValueError("Row count mismatch")
        if table["source_row"].to_pylist() != list(
            range(expected_source_row, expected_source_row + table.num_rows)
        ):
            raise ValueError("Source rows have gaps or duplicates")
        if table["embedding"].type != pa.list_(pa.float32(), 1024):
            raise ValueError("Unexpected vector format")
        matrix = table["embedding"].combine_chunks().values.to_numpy().reshape(-1, 1024)
        norms = np.linalg.norm(matrix, axis=1)
        if not np.isfinite(matrix).all() or not np.allclose(norms, 1, atol=2e-5):
            raise ValueError("Invalid vector values")
        if set(table["model"].to_pylist()) != {"bge-m3"}:
            raise ValueError("Unexpected model")
        token_counts = table["tokens_used"].to_numpy()
        if not ((token_counts > 0) & (token_counts <= 8192)).all():
            raise ValueError("Unexpected token count")
        for lang, page_id in zip(
            table["lang"].to_pylist(), table["page_id"].to_pylist(), strict=True
        ):
            if lang != "fi" or (lang, page_id) in seen:
                raise ValueError("Unexpected language or duplicate page identity")
            seen.add((lang, page_id))
        truncated += sum(table["truncated"].to_pylist())
        expected_source_row += table.num_rows
        total_bytes += path.stat().st_size
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    if (
        expected_source_row != report["config"]["population"]
        or len(seen) != report["rows"]
    ):
        raise ValueError("Incomplete corpus")
    if total_bytes != report["vector_bytes"] or truncated != report["truncated"]:
        raise ValueError("Summary differs from downloaded artifacts")
    result = {
        "run_id": args.run_id,
        "status": "verified_locally",
        "verified_rows": len(seen),
        "shards": len(report["shards"]),
        "vector_bytes": total_bytes,
        "vector_GB": total_bytes / 1e9,
        "vector_GiB": total_bytes / 2**30,
        "dimensions": 1024,
        "dtype": "float32",
        "truncated": truncated,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "gpu_function_seconds": report["gpu_function_seconds"],
        "embedding_batch_seconds": report["embedding_batch_seconds"],
        "all_hashes_match_durable_shards": True,
    }
    (ROOT / "reports" / f"{args.run_id}-local-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
