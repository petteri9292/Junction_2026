"""Validated user controls for the interactive personalized RAG notebook."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PipelineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=1, max_length=2000)
    search_focus: str = Field(max_length=200)
    answer_language: str = Field(min_length=1, max_length=100)
    input_mode: Literal["hyde", "saved"] = "hyde"
    saved_retrieval_file: str = "reports/winter-war-top10.json"
    hyde_max_new_tokens: int = Field(default=192, ge=1)
    hyde_max_time: float = Field(default=45, gt=0, le=60, allow_inf_nan=False)
    search_top_k: int = Field(default=10, ge=1, le=100)
    search_language: Literal["fi", "en", "sv"] | None = None
    rerank_top_k: int = Field(default=10, ge=1, le=100)
    min_relevance: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    passage_chars: int = Field(default=1600, ge=100, le=4000)
    passage_overlap: int = Field(default=250, ge=0)
    compression_max_new_tokens: int = Field(default=384, ge=1)
    compression_max_time: float = Field(default=15, gt=0, le=60, allow_inf_nan=False)
    postprocessing_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    max_evidence_nodes: int = Field(default=10, ge=1, le=50)
    max_per_article: int = Field(default=3, ge=1, le=10)
    context_chars: int = Field(default=7000, ge=1, le=20000)
    model_context_tokens: int = Field(default=8192, ge=256, le=8192)
    embedding_max_batch_tokens: int = Field(default=1024, ge=1, le=8192)
    strict_generation: bool = False
    answer_target_words: int = Field(default=500, ge=1)
    answer_max_new_tokens: int = Field(default=4000, ge=1)
    answer_max_time: float = Field(default=300, gt=0, le=600, allow_inf_nan=False)
    citation_margin_chars: int = Field(default=300, ge=0, le=5000)

    @model_validator(mode="after")
    def check_budgets(self):
        if not self.question.strip() or not self.answer_language.strip():
            raise ValueError("Question and answer language must not be blank")
        if self.passage_overlap >= self.passage_chars:
            raise ValueError("passage_overlap must be smaller than passage_chars")
        if (
            max(
                self.hyde_max_new_tokens,
                self.compression_max_new_tokens,
                self.answer_max_new_tokens,
            )
            >= self.model_context_tokens
        ):
            raise ValueError("Output tokens must leave room for the input prompt")
        return self
