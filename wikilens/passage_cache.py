"""Revision/model/content-keyed CPU passage vectors; no network or inference."""

import hashlib
import json
import sqlite3
import struct
from pathlib import Path

from .evidence import pack_vector


class PassageCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS passage_vector (cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL CHECK(length(vector)=4096))"
            )

    def _connect(self):
        return sqlite3.connect(self.path, timeout=20)

    @staticmethod
    def key(passage, revision):
        if not isinstance(revision, str) or not revision:
            raise ValueError("A resolved embedding model revision is required")
        identity = {
            k: passage[k]
            for k in ("lang", "page_id", "rev_id", "start_char", "end_char")
        }
        identity.update(
            model="BAAI/bge-m3",
            model_revision=revision,
            pooling="cls_l2_v1",
            text_sha256=hashlib.sha256(passage["text"].encode()).hexdigest(),
        )
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def get_many(self, passages, revision):
        with self._connect() as db:
            result = []
            for passage in passages:
                row = db.execute(
                    "SELECT vector FROM passage_vector WHERE cache_key=?",
                    (self.key(passage, revision),),
                ).fetchone()
                if row:
                    vector = list(struct.unpack("<1024f", row[0]))
                    pack_vector(
                        vector
                    )  # Reject corrupt/nonfinite values, not just byte size.
                    result.append(vector)
                else:
                    result.append(None)
        return result

    def put_many(self, passages, revision, vectors):
        if len(passages) != len(vectors):
            raise ValueError("Passage and vector counts differ")
        rows = [
            (self.key(p, revision), pack_vector(v)) for p, v in zip(passages, vectors)
        ]
        with self._connect() as db:
            db.executemany(
                "INSERT OR IGNORE INTO passage_vector(cache_key,vector) VALUES (?,?)",
                rows,
            )
