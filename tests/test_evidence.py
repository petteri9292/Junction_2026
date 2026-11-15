import struct

import pytest

from wikilens.evidence import (
    coverage_signals,
    excerpt,
    external_links,
    fuse_rankings,
    pack_vector,
    revision_url,
)


def test_coverage_counts_distinct_external_links_and_preserves_finnish_headings():
    text = """# Suomen historia
## Lähteet
[Archive](https://archives.example.org/finland)
[Same archive again](https://archives.example.org/finland)
[Wikipedia](https://fi.wikipedia.org/wiki/Suomi)
[Image](https://commons.wikimedia.org/example)
[Other](http://research.example.net/paper)
"""
    signals = coverage_signals(text)
    assert signals["sections"] == 2
    assert signals["headings"] == ["Suomen historia", "Lähteet"]
    assert signals["external_links"] == 2
    assert "https://archives.example.org/finland" in external_links(text)


@pytest.mark.parametrize(
    "vector", [[0.0] * 1023, [float("nan")] * 1024, [float("inf")] * 1024]
)
def test_rejects_vectors_that_cannot_match_the_supplied_index(vector):
    with pytest.raises(ValueError, match="1024 finite"):
        pack_vector(vector)


def test_vector_binary_round_trip_uses_little_endian_float32():
    vector = [0.0] * 1024
    vector[0], vector[-1] = 0.5, -0.25
    binary = pack_vector(vector)
    assert len(binary) == 4096
    assert struct.unpack("<1024f", binary) == tuple(vector)


def test_citation_identifies_snapshot_revision_instead_of_live_article():
    assert (
        revision_url("fi", 12345) == "https://fi.wikipedia.org/w/index.php?oldid=12345"
    )
    with pytest.raises(ValueError):
        revision_url("invalid.example", 12345)


def test_finewiki_single_newline_after_heading_preserves_source_evidence():
    text = "# Talvisota\nTalvisota alkoi 30. marraskuuta 1939.\n## Tausta\nNeuvottelut."
    assert excerpt(text).startswith("Talvisota alkoi 30. marraskuuta 1939.")
    assert "# Talvisota" not in excerpt(text)


def test_fusion_promotes_topic_supported_by_both_retrieval_methods():
    unrelated = {"lang": "en", "page_id": 1}
    topic = {"lang": "en", "page_id": 2}
    related = {"lang": "fi", "page_id": 3}
    results = fuse_rankings([unrelated, topic], [related, topic])
    assert results[0]["page_id"] == 2
    assert len(results) == 3


def test_malformed_source_url_does_not_block_coverage_metadata():
    text = "Broken https://[example and valid https://example.org/article."
    original = text
    assert external_links(text) == ["https://example.org/article"]
    assert coverage_signals(text)["characters"] == len(original)
    assert text == original
