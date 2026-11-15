import json

import pytest

from wikilens.grounding import (
    GroundingError,
    chunk_articles,
    validate_quotations,
    validate_synthesis,
)


def article(text, **updates):
    return {"lang": "fi", "page_id": 12, "rev_id": 34, "text": text, **updates}


def test_chunking_covers_full_unicode_article_with_literal_overlapping_offsets():
    text = "# Åäö\n" + "Alku. " * 20 + "Lopun tärkeä vastaus."
    passages = chunk_articles([article(text)], max_chars=40, overlap_chars=10)
    covered = set()
    for p in passages:
        assert p["text"] == text[p["start_char"] : p["end_char"]]
        assert p["revision_url"] == "https://fi.wikipedia.org/w/index.php?oldid=34"
        covered.update(range(p["start_char"], p["end_char"]))
    assert covered == set(range(len(text)))
    assert passages[-1]["text"].endswith("Lopun tärkeä vastaus.")
    assert passages[0]["text"][-10:] == passages[1]["text"][:10]


def test_ids_do_not_depend_on_retrieval_order_and_revisions_stay_distinct():
    a, b = article("First article."), article("Second article.", page_id=22)
    one, two = chunk_articles([a, b]), chunk_articles([b, a])
    assert {p["id"] for p in one} == {p["id"] for p in two}
    assert len(chunk_articles([a, a])) == 1
    assert len(chunk_articles([a, {**a, "rev_id": 35}])) == 2
    with pytest.raises(GroundingError, match="conflicting"):
        chunk_articles([a, {**a, "text": "Different text under same revision."}])


@pytest.mark.parametrize("max_chars,overlap", [(0, 0), (10, 10), (10, -1), (True, 0)])
def test_invalid_chunk_parameters_fail_before_work(max_chars, overlap):
    with pytest.raises(GroundingError):
        chunk_articles([], max_chars=max_chars, overlap_chars=overlap)


def test_empty_article_and_untrusted_revision_metadata_cannot_create_fake_links():
    assert chunk_articles([article("   ")]) == []
    with pytest.raises(GroundingError):
        chunk_articles([article("Text", lang="example.com")])
    with pytest.raises(GroundingError):
        chunk_articles([article("Text", rev_id="34&action=edit")])


QUOTE = "The fortress construction began in 1748 under the Swedish Crown."
PASSAGES = [{"id": "S1", "text": "Prefix.\n" + QUOTE + "\nEnd.", "start_char": 100}]


def test_quote_whitespace_is_canonicalized_to_exact_source_and_offsets_are_recoverable():
    source = [
        {
            "id": "S1",
            "text": "Prefix.\n" + QUOTE.replace("began in", "began\n  in"),
            "start_char": 200,
        }
    ]
    checked = validate_quotations([{"source_id": "S1", "quote": QUOTE}], source)
    result = checked[0]
    assert result["quote"] == QUOTE.replace("began in", "began\n  in")
    assert (
        result["quote"]
        == source[0]["text"][result["quote_start_char"] : result["quote_end_char"]]
    )
    assert result["quote_article_start_char"] == 200 + result["quote_start_char"]
    assert result["provenance_checked"] is True


@pytest.mark.parametrize(
    "selection",
    [
        {"source_id": "S99", "quote": QUOTE},
        {"source_id": [], "quote": QUOTE},
        {
            "source_id": "S1",
            "quote": "The fortress construction began in 1900 under the Swedish Crown.",
        },
        {"source_id": "S1", "quote": "The fortress"},
        {"source_id": "S1", "quote": QUOTE, "url": "https://evil.example"},
    ],
)
def test_one_bad_quotation_rejects_whole_response(selection):
    valid = {"source_id": "S1", "quote": QUOTE}
    with pytest.raises(GroundingError):
        validate_quotations([valid, selection], PASSAGES)


def test_duplicate_ids_are_ambiguous_and_no_evidence_is_explicitly_empty():
    with pytest.raises(GroundingError, match="Duplicate passage ID"):
        validate_quotations([], PASSAGES + PASSAGES)
    assert validate_quotations([], PASSAGES) == []


SOURCES = [{"id": "S1", "quote": QUOTE}]


def synthesis(text="Construction began in 1748.", sources=None):
    return {
        "insufficient_evidence": False,
        "sentences": [
            {"text": text, "sources": ["S1"] if sources is None else sources}
        ],
    }


def test_synthesis_validates_ids_but_never_asserts_claim_entailment():
    # Deliberately unsupported text passes ID checks: the API must not label it verified.
    result = validate_synthesis(synthesis("Construction began in 1900."), SOURCES)
    assert result["status"] == "draft"
    assert result["claim_support_reviewed"] is False
    assert "do not establish claim support" in result["limitation"]


@pytest.mark.parametrize(
    "payload",
    [
        synthesis(sources=[]),
        synthesis(sources=["S2"]),
        synthesis(sources=[["S1"]]),
        synthesis(sources=["S1", "S1"]),
        synthesis(text="See [S1]."),
        synthesis(text='<script>alert("x")</script>'),
        synthesis(text="https://evil.example"),
        {"insufficient_evidence": 1, "sentences": []},
        {"insufficient_evidence": False, "sentences": []},
        {
            "insufficient_evidence": True,
            "sentences": [{"text": "Answer", "sources": ["S1"]}],
        },
        {**synthesis(), "confidence": 1.0},
    ],
)
def test_malformed_or_untrusted_synthesis_is_rejected(payload):
    with pytest.raises(GroundingError):
        validate_synthesis(payload, SOURCES)


def test_empty_evidence_allows_only_abstention_and_json_fences_are_supported():
    payload = {"insufficient_evidence": True, "sentences": []}
    assert validate_synthesis(payload, [])["status"] == "insufficient_evidence"
    with pytest.raises(GroundingError):
        validate_synthesis(synthesis(), [])
    raw = "```json\n" + json.dumps(synthesis()) + "\n```"
    assert validate_synthesis(raw, SOURCES)["status"] == "draft"


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        '{"insufficient_evidence": true, "sentences": [], "sentences": []}',
        '{"insufficient_evidence": NaN, "sentences": []}',
    ],
)
def test_json_parser_rejects_ambiguous_keys_and_nonstandard_constants(raw):
    with pytest.raises(GroundingError):
        validate_synthesis(raw, SOURCES)
