"""Download the exact L article snapshot matching the supplied embedding run."""

import hashlib, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "data/l-2026-10/articles-fi.parquet"
p.parent.mkdir(parents=True, exist_ok=True)
expected = "05d5520fc272b0f879bdf62ea67e8bf853c55ce5975eb5393f20c78c086158ec"


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


if not p.exists():
    tmp = p.with_suffix(".part")
    size = 0
    with (
        urllib.request.urlopen(
            "https://github.com/MariaDB/mariadb-wikipedia-datasets/releases/download/l-2026-10/articles-fi.parquet",
            timeout=120,
        ) as response,
        tmp.open("wb") as f,
    ):
        while block := response.read(8 * 1024 * 1024):
            f.write(block)
            size += len(block)
            if size % (64 * 1024 * 1024) == 0:
                print("L articles bytes", size, flush=True)
    if digest(tmp) != expected:
        raise ValueError("L article SHA256 mismatch")
    tmp.replace(p)
if digest(p) != expected:
    raise ValueError("Existing L article SHA256 mismatch")
print("Verified L articles", p.stat().st_size, flush=True)
