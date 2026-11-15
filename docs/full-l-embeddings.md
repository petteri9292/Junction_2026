# Data and reproduction

## Verified corpus

Dataset L (`l-2026-10`) contains **568,416 Finnish Wikipedia articles**. The source file is published in the MariaDB Wikipedia datasets release:
https://github.com/MariaDB/mariadb-wikipedia-datasets/releases/tag/l-2026-10

Expected article Parquet SHA-256:
`05d5520fc272b0f879bdf62ea67e8bf853c55ce5975eb5393f20c78c086158ec`.

The completed embedding run produced 57 fixed-size 1024-dimensional float32 Parquet shards, 2,374,491,697 bytes. Original page/revision identities and source-row positions are preserved. The importer verifies input hashes, schema and source alignment before inserting into MariaDB's native vector index. The live demo uses the completed L import, rather than the earlier small S prototype.

BGE-M3 revision: `5617a9f61b028005a4858fdac845db406aefb181`. Inputs are title, two newlines, then source text. Embeddings use normalized CLS pooling, FP16 inference and an 8192-token input limit. **2,441** long articles were marked truncated. Full source text remains available for passage extraction. Independent remote and downloaded-shard verification checked hashes, finite values, vector norms, unique identities and contiguous full coverage.

## Reproduction components

Use Python 3.12 and `uv sync --locked`. Credentials belong in ignored environment files; `.env.example` documents local DB settings.

- `scripts/fetch-l-articles.py`: fetch and checksum the official article file.
- `scripts/modal-embedding-benchmark.py`: bounded embedding compatibility/throughput benchmark.
- `scripts/modal-embed-l.py`: checkpointed full pretrained embedding inference; not training.
- `scripts/modal-l-control.py`: status, explicit checkpoint resume, verify and collect.
- `scripts/verify-l-embeddings.py`: reopen downloaded shards and verify them.
- `scripts/setup-remote-mariadb.py`: provision the native MariaDB runtime used in the experiments.
- `scripts/import-l-embeddings.py`: validate and import articles/vectors.
- `scripts/modal-full-l-search.py`: bounded cloud database import and retrieval checks.

Inspect the command's `--help` and configuration before running cloud entrypoints. Historical control scripts expect the operator's ignored ledgers and prepared artifact paths. They are reproducible experiment code, not automatic installation for an arbitrary new Modal account. A fresh account must provision its own Volume and set the source snapshot path in the live controller.

To validate or import locally, after obtaining the source file and verified vector shards:

```bash
.venv/bin/python scripts/import-l-embeddings.py \
  --articles data/l-2026-10/articles-fi.parquet \
  --vectors data/l-embeddings/l-full-bge-m3-20261114T123013Z \
  --verify-only
```

Remove `--verify-only` only when the intended MariaDB database is configured. The importer supports `--resume` and writes a report; preserve the corpus revision, model revision, code hashes and run configuration with each result.

## Environment separation

The root `pyproject.toml` and `uv.lock` support notebooks and legacy Transformers inference. `inference/vllm/pyproject.toml` and its own `uv.lock` pin the H100 inference stack. `scripts/modal-vllm-h100.py` builds that stack with CUDA and precompiles the sampling kernel; `scripts/modal-paper-live.py` adds the persistent database-backed live worker.

Model revisions used in the presentation:

| Component | Checkpoint | Revision |
| --- | --- | --- |
| Generator | Qwen/Qwen3-4B-Instruct-2507-FP8 | `8591804019c8b22094c3b5b4454e0edc05dffc98` |
| Embedder | BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` |
| Reranker | BAAI/bge-reranker-v2-m3 | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

No model weights or full article/vector files are committed. Wikipedia and derived vectors retain CC BY-SA 4.0 attribution. The public fallback contains selected source excerpts with revision links.
