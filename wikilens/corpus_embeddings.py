"""Integrity checks shared by the full-corpus embedding job and local tests."""

import hashlib
import json

import numpy as np
import pyarrow as pa


def digest_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def vector_table(rows, matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (len(rows), 1024):
        raise ValueError("Expected one 1024-dimensional vector per input")
    return pa.table(
        {
            "lang": [r["lang"] for r in rows],
            "page_id": [r["page_id"] for r in rows],
            "rev_id": [r["rev_id"] for r in rows],
            "source_row": [r["source_row"] for r in rows],
            "model": ["bge-m3"] * len(rows),
            "truncated": [r["truncated"] for r in rows],
            "tokens_used": [len(r["input_ids"]) for r in rows],
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.ravel()), 1024
            ),
        }
    )


def validate_vectors(table, expected_rows):
    if table.num_rows != len(expected_rows):
        raise ValueError("Checkpoint row count mismatch")
    if table["embedding"].type != pa.list_(pa.float32(), 1024):
        raise ValueError("Checkpoint must contain fixed-size float32 vectors")
    for key in ("lang", "page_id", "rev_id", "source_row", "truncated"):
        if table[key].to_pylist() != [r[key] for r in expected_rows]:
            raise ValueError(f"Checkpoint {key} differs from its source")
    if table["tokens_used"].to_pylist() != [len(r["input_ids"]) for r in expected_rows]:
        raise ValueError("Checkpoint token counts differ from source")
    if set(table["model"].to_pylist()) != {"bge-m3"}:
        raise ValueError("Unexpected checkpoint model")
    matrix = table["embedding"].combine_chunks().values.to_numpy().reshape(-1, 1024)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite vector")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1, atol=2e-5):
        raise ValueError("Non-unit vector")
    return {
        "rows": table.num_rows,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }
