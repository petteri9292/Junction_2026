"""Verify downloaded subset results and emit an S-schema float32 vector file."""

import argparse
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
    directory = ROOT / "data" / "modal-benchmarks" / args.run_id
    report = json.loads((directory / "report.json").read_text())
    sample = pq.read_table(directory / "sample.parquet").to_pylist()
    vectors = pq.read_table(directory / "vectors.parquet").to_pylist()
    sample_by_id = {(r["lang"], r["page_id"]): r for r in sample}
    vector_keys = [(r["lang"], r["page_id"]) for r in vectors]
    assert len(sample_by_id) == len(sample) == len(vectors) == report["sample_total"]
    assert len(set(vector_keys)) == len(vector_keys)
    assert set(vector_keys) == set(sample_by_id)
    assert sum(not r["stress"] for r in vectors) == report["sample_size"]
    for row in vectors:
        source = sample_by_id[(row["lang"], row["page_id"])]
        assert row["rev_id"] == source["rev_id"]
        assert row["tokens"] == source["tokens"]
        assert row["truncated"] == (source["tokens"] > report["max_tokens"])
    matrix = np.asarray([r["embedding"] for r in vectors], dtype=np.float32)
    assert matrix.shape == (len(sample), 1024)
    assert np.isfinite(matrix).all()
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1, atol=2e-5)
    minimum_cosine = min(r["cosine"] for r in report["compatibility"])
    assert len(report["compatibility"]) == 9 and minimum_cosine >= 0.99
    # Match S's native vector schema instead of Python-list inferred doubles.
    canonical = pa.table(
        {
            "lang": [r["lang"] for r in vectors],
            "page_id": [r["page_id"] for r in vectors],
            "model": [r["model"] for r in vectors],
            "truncated": [r["truncated"] for r in vectors],
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.ravel()), 1024
            ),
        }
    )
    output = directory / "vectors-bge-m3-fi.parquet"
    pq.write_table(canonical, output)
    reopened = pq.read_table(output)
    assert reopened["embedding"].type == pa.list_(pa.float32(), 1024)
    assert np.array_equal(
        np.asarray(reopened["embedding"].to_pylist(), dtype=np.float32), matrix
    )
    verification = {
        "run_id": args.run_id,
        "verified_rows": len(vectors),
        "representative_rows": report["sample_size"],
        "extra_stress_rows": len(vectors) - report["sample_size"],
        "dimensions": 1024,
        "finite": True,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "minimum_s_compatibility_cosine": minimum_cosine,
        "canonical_vector_file": str(output.relative_to(ROOT)),
        "canonical_vector_dtype": "fixed_size_list<float32>[1024]",
        "canonical_file_bytes": output.stat().st_size,
        "projected_full_embedding_hours": report["projected_full_embedding_hours"],
    }
    (ROOT / "reports" / f"{args.run_id}-verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
