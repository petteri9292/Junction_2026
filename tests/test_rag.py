from wikilens.rag import materialize_claims, validate_claims

SOURCES = [
    {
        "id": "S1",
        "excerpt": "The Winter War began on 30 November 1939. It ended in March 1940.",
    }
]


def test_quote_comes_from_server_owned_passage_not_model_text():
    generated = [
        {
            "text": "It began in November 1939.",
            "evidence_id": "S1-P1",
            "quote": "made up",
        }
    ]
    result = materialize_claims(generated, SOURCES)
    assert result[0]["quote"] == "The Winter War began on 30 November 1939."
    assert result[0]["source_id"] == "S1"
    assert materialize_claims([{"text": "x", "evidence_id": "invented"}], SOURCES) == []


def test_valid_quote_is_retained_but_fabricated_and_unknown_citations_are_removed():
    valid = {
        "source_id": "S1",
        "text": "The war began in November 1939.",
        "quote": "The Winter War began on 30 November 1939.",
    }
    invented = {
        "source_id": "S1",
        "text": "Invented claim",
        "quote": "The Winter War began on 1 January 1900.",
    }
    unknown = dict(valid, source_id="S99")
    assert validate_claims([valid, invented, unknown], SOURCES) == [valid]


def test_malformed_output_cannot_bypass_citation_validation():
    assert validate_claims("pretend claims", SOURCES) == []
    assert (
        validate_claims([None, {"source_id": [], "text": "x", "quote": 42}], SOURCES)
        == []
    )
