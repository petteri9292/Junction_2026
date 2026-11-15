import json
import pytest
from lmformatenforcer import JsonSchemaParser
from pydantic import ValidationError
from wikilens.output_schemas import Answer, Compression, answer_schema


def accepted_by_decoder(text, schema):
    parser = JsonSchemaParser(schema)
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def test_schema_decoder_requires_nested_evidence_and_only_available_ids():
    schema = answer_schema(["S2"])
    valid = {
        "insufficient_evidence": False,
        "claims": [
            {
                "text": "A statement.",
                "evidence": [
                    {
                        "source_id": "S2",
                        "quote": "An exact source quotation with enough words.",
                    }
                ],
            }
        ],
    }
    assert accepted_by_decoder(json.dumps(valid), schema)
    missing = {"insufficient_evidence": False, "claims": [{"text": "A statement."}]}
    assert not accepted_by_decoder(json.dumps(missing), schema)
    valid["claims"][0]["evidence"][0]["source_id"] = "S99"
    assert not accepted_by_decoder(json.dumps(valid), schema)
    assert accepted_by_decoder('{"insufficient_evidence":true,"claims":[]}', schema)


def test_pydantic_rejects_missing_evidence_extra_fields_and_invalid_abstention():
    for raw in [
        {"insufficient_evidence": False, "claims": [{"text": "Missing evidence"}]},
        {"insufficient_evidence": False, "claims": []},
        {"insufficient_evidence": True, "claims": [], "extra": 1},
    ]:
        with pytest.raises(ValidationError):
            Answer.model_validate(raw)
    with pytest.raises(ValidationError):
        Compression.model_validate({"score": True, "excerpts": []})


def test_runtime_passes_schema_into_backend_without_silently_falling_back():
    from wikilens.torch_runtime import TorchRuntime

    class Backend:
        def load(self, *a):
            return object(), object(), "revision"

        def cleanup(self):
            pass

        def reset_peak(self):
            pass

        def memory(self):
            return {}

        def generate(self, *a, **kwargs):
            assert kwargs["output_schema"] == answer_schema(["S2"])
            return '{"insufficient_evidence":true,"claims":[]}'

    with TorchRuntime(_backend=Backend()) as runtime:
        value = runtime.generate(
            [{"role": "user", "content": "Question"}],
            output_schema=answer_schema(["S2"]),
        )
        assert Answer.model_validate_json(value).insufficient_evidence
