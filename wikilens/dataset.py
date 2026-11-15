"""Download and inspect the challenge's versioned, checksum-verified release."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import requests

RELEASE = "s-2026-10"
BASE_URL = (
    "https://github.com/MariaDB/mariadb-wikipedia-datasets/releases/download/" + RELEASE
)
LANGUAGES = ("fi", "en", "sv")
FILES = (
    *(f"articles-{lang}.parquet" for lang in LANGUAGES),
    *(f"vectors-bge-m3-{lang}.parquet" for lang in LANGUAGES),
    "article-categories-fi.tsv",
    "category-links-fi.tsv",
    "README.md",
)
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / RELEASE


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_dataset(destination=DEFAULT_DATA):
    """Download S without rebuilding Wikipedia or computing any embeddings."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{BASE_URL}/SHA256SUMS", timeout=60)
    response.raise_for_status()
    checksums = {}
    for line in response.text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            checksums[name.lstrip(" *")] = digest
    manifest = {"release": RELEASE, "source": BASE_URL, "files": {}}
    for name in FILES:
        expected = checksums[name]
        path = destination / name
        if not path.exists() or sha256(path) != expected:
            temporary = path.with_suffix(path.suffix + ".part")
            with requests.get(f"{BASE_URL}/{name}", stream=True, timeout=120) as remote:
                remote.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in remote.iter_content(1024 * 1024):
                        output.write(chunk)
            if sha256(temporary) != expected:
                temporary.unlink()
                raise ValueError(f"Checksum mismatch: {name}")
            temporary.replace(path)
        print(f"Verified {name} ({path.stat().st_size:,} bytes)", flush=True)
        manifest["files"][name] = {"sha256": expected, "bytes": path.stat().st_size}
    (destination / "SHA256SUMS").write_text(response.text)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def profile_dataset(directory=DEFAULT_DATA):
    """Read metadata and identifiers without loading all article text or vectors."""
    directory = Path(directory)
    per_language = {}
    topics = {}
    for lang in LANGUAGES:
        article_path = directory / f"articles-{lang}.parquet"
        vector_path = directory / f"vectors-bge-m3-{lang}.parquet"
        articles = pq.read_table(article_path, columns=["page_id", "wikidata_id"])
        vectors = pq.read_table(vector_path, columns=["page_id", "truncated", "model"])
        article_ids = articles.column("page_id").to_pylist()
        vector_ids = vectors.column("page_id").to_pylist()
        topic_ids = articles.column("wikidata_id").to_pylist()
        topics[lang] = {topic for topic in topic_ids if topic}
        per_language[lang] = {
            "articles": len(article_ids),
            "vectors": len(vector_ids),
            "duplicate_article_ids": len(article_ids) - len(set(article_ids)),
            "duplicate_vector_ids": len(vector_ids) - len(set(vector_ids)),
            "articles_without_vectors": len(set(article_ids) - set(vector_ids)),
            "orphan_vectors": len(set(vector_ids) - set(article_ids)),
            "missing_wikidata_id": sum(not topic for topic in topic_ids),
            "truncated_vectors": sum(vectors.column("truncated").to_pylist()),
            "models": sorted(set(vectors.column("model").to_pylist())),
            "article_columns": pq.ParquetFile(article_path).schema_arrow.names,
            "vector_schema": str(pq.ParquetFile(vector_path).schema_arrow),
        }
    overlap = {}
    for langs in (("fi", "en"), ("fi", "sv"), ("en", "sv"), LANGUAGES):
        overlap["+".join(langs)] = len(
            set.intersection(*(topics[lang] for lang in langs))
        )
    return {"release": RELEASE, "languages": per_language, "shared_topics": overlap}
