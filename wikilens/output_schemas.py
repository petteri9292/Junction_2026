"""Pydantic output contracts used both for constrained decoding and validation."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(StrictOutput):
    source_id: Annotated[str, Field(min_length=1)]
    quote: Annotated[str, Field(min_length=1, max_length=1600)]


class Claim(StrictOutput):
    text: Annotated[str, Field(min_length=1, max_length=1800)]
    heading: str = Field(default="", max_length=120)
    evidence: Annotated[list[Evidence], Field(min_length=1, max_length=3)]


class Answer(StrictOutput):
    insufficient_evidence: bool
    claims: Annotated[list[Claim], Field(max_length=24)]

    @model_validator(mode="after")
    def check_abstention(self):
        if self.insufficient_evidence != (len(self.claims) == 0):
            raise ValueError("Abstention must have no claims; an answer needs claims")
        return self


class Compression(StrictOutput):
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    excerpts: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=700)]], Field(max_length=3)
    ]


def answer_schema(source_ids):
    """Limit the decoder's citation choices to the supplied evidence IDs."""
    ids = list(dict.fromkeys(source_ids))
    if not ids or any(not isinstance(s, str) or not s for s in ids):
        raise ValueError("Nonempty source IDs required")
    schema = Answer.model_json_schema()
    schema["$defs"]["Evidence"]["properties"]["source_id"]["enum"] = ids
    return schema
