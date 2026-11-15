import pytest
from pydantic import ValidationError
from wikilens.notebook_settings import PipelineSettings

BASE = {"question": "When?", "search_focus": "history", "answer_language": "English"}


@pytest.mark.parametrize(
    "change",
    [
        {"search_top_k": 0},
        {"search_language": "de"},
        {"passage_chars": 100, "passage_overlap": 100},
        {"answer_max_new_tokens": 8192},
        {"answer_max_new_tokens": True},
        {"hyde_max_time": 61},
        {"postprocessing_seconds": float("nan")},
        {"model_context_tokens": 512},
        {"min_relevance": 1.1},
        {"question": "  "},
    ],
)
def test_invalid_settings_fail_before_model_work(change):
    with pytest.raises(ValidationError):
        PipelineSettings.model_validate({**BASE, **change})


def test_default_controls_match_pipeline_defaults_and_roundtrip():
    c = PipelineSettings.model_validate(BASE)
    assert c.search_top_k == 10 and c.rerank_top_k == 10
    assert c.answer_max_new_tokens == 4000 and c.model_context_tokens == 8192
    assert PipelineSettings.model_validate_json(c.model_dump_json()) == c


def test_lengths_above_700_are_configurable_but_must_leave_prompt_room():
    c = PipelineSettings.model_validate(
        {
            **BASE,
            "answer_max_new_tokens": 1500,
            "hyde_max_new_tokens": 900,
            "compression_max_new_tokens": 1000,
        }
    )
    assert c.answer_max_new_tokens == 1500
    with pytest.raises(ValidationError):
        PipelineSettings.model_validate({**BASE, "answer_max_new_tokens": 8192})
