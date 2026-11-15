"""Inspectable coverage signals, not judgements of article quality or bias."""

import math
import re
import struct
from urllib.parse import urlsplit

DIMENSIONS = 1024


def fuse_rankings(*rankings, limit=12, rank_constant=60):
    """Reciprocal rank fusion combines keyword and semantic rankings, not scores."""
    combined = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, 1):
            key = (row["lang"], row["page_id"])
            entry = combined.setdefault(key, {"fusion_score": 0.0})
            score = entry["fusion_score"] + 1 / (rank_constant + rank)
            entry.update(row)
            entry["fusion_score"] = score
    return sorted(
        combined.values(),
        key=lambda row: (-row["fusion_score"], row["lang"], row["page_id"]),
    )[:limit]


def pack_vector(vector):
    if len(vector) != DIMENSIONS or not all(math.isfinite(float(x)) for x in vector):
        raise ValueError("Expected 1024 finite BGE-M3 embedding values")
    return struct.pack(f"<{DIMENSIONS}f", *vector)


def external_links(text):
    """Distinct HTTP(S) links in Markdown; a heuristic, not a reference count."""
    found = set()
    for match in re.finditer(r"https?://[^\s<>\]\)\"']+", text):
        url = match.group(0).rstrip(".,;")
        try:
            host = (urlsplit(url).hostname or "").lower()
        except ValueError:
            # Full Wikipedia text can contain broken URLs. This optional
            # coverage counter must not prevent importing the original article.
            continue
        if not host or any(
            host == domain or host.endswith("." + domain)
            for domain in ("wikipedia.org", "wikimedia.org", "wikidata.org")
        ):
            continue
        found.add(url)
    return sorted(found)


def coverage_signals(text):
    headings = re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, flags=re.MULTILINE)
    links = external_links(text)
    return {
        "characters": len(text),
        "sections": len(headings),
        "external_links": len(links),
        "headings": headings,
        "links": links,
    }


def excerpt(text, max_characters=800):
    # FineWiki often has single newlines after headings and between paragraphs.
    # Removing entire blank-line-delimited blocks would discard the article lead.
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "|", "!["))
    ]
    content = "\n\n".join(lines[:3])
    return content[:max_characters] + ("…" if len(content) > max_characters else "")


def revision_url(lang, revision_id):
    if lang not in ("fi", "en", "sv"):
        raise ValueError("Unsupported Wikipedia language")
    return f"https://{lang}.wikipedia.org/w/index.php?oldid={int(revision_id)}"
