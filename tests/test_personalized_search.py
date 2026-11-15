import json
from unittest.mock import MagicMock, patch
import pytest
from wikilens.hyde_search import generate_focused_hyde, retrieve_hyde
from wikilens.qwen_postprocessing import qwen_postprocess
from wikilens.node_pipeline import generate_answer
from wikilens.personalization import normalize_focus


class Runtime:
    def __init__(self):
        self.calls = []
        self.embeds = []

    def generate(self, messages, **kw):
        self.calls.append((messages, kw))
        data = json.loads(messages[1]["content"])
        if "text" in data:
            return '{"score":0,"excerpts":[]}'
        if "passages" in data:
            return '{"insufficient_evidence":true,"claims":[]}'
        return "HYPOTHETICAL passage for " + data["search_focus"]

    def embed(self, texts, **kw):
        self.embeds.append(texts)
        return [[1.0] + [0.0] * 1023]


@pytest.mark.parametrize("focus", ["history", "video games", ""])
def test_focus_reaches_hyde_reranking_and_answer_as_preference(focus):
    r = Runtime()
    question = "What was the Winter War?"
    hyde = generate_focused_hyde(r, question, focus)
    assert hyde["role"] == "retrieval_only_never_evidence"
    assert json.loads(r.calls[-1][0][1]["content"]) == {
        "question": question,
        "search_focus": focus,
    }
    article = dict(
        lang="en",
        page_id=1,
        rev_id=2,
        title="Topic",
        text="Actual source text about a real event with sufficient words.",
    )
    post = qwen_postprocess(r, question, [article], search_focus=focus)
    data = json.loads(r.calls[-1][0][1]["content"])
    assert data["query"] == question and data["search_focus"] == focus
    assert data["text"] == article["text"] and "HYPOTHETICAL" not in data["text"]
    answer = generate_answer(
        r,
        question,
        [{"id": "S1", "title": "Topic", "lang": "en", "text": article["text"]}],
        search_focus=focus,
    )
    data = json.loads(r.calls[-1][0][1]["content"])
    assert data["search_focus"] == focus and "HYPOTHETICAL" not in str(data["passages"])
    assert post["search_focus"] == focus and answer["search_focus"] == focus


def test_retrieval_embeds_only_hyde_and_preserves_global_top_ten_order():
    from wikilens.evidence import revision_url

    r = Runtime()
    h = generate_focused_hyde(r, "Query", "history")
    candidate = {"lang": "fi", "page_id": 1, "distance": 0.2}
    article = dict(
        lang="fi",
        page_id=1,
        rev_id=2,
        wikidata_id="Q1",
        title="Topic",
        url="https://fi.wikipedia.org/wiki/Topic",
        text="Source",
    )
    cursor = MagicMock()
    cursor.fetchall.side_effect = [[{"key": "article_semantics"}], [article]]
    db = MagicMock()
    db.__enter__.return_value = db
    db.cursor.return_value.__enter__.return_value = cursor
    with (
        patch("wikilens.database.connect", return_value=db),
        patch("wikilens.database._vector_search", return_value=[candidate]) as search,
    ):
        result = retrieve_hyde(r, "Query", h)
    assert r.embeds == [[h["text"]]]
    assert search.call_args.kwargs == {"limit": 10}
    assert result["articles"][0]["retrieval_rank"] == 1
    assert result["articles"][0]["revision_url"] == revision_url("fi", 2)
    assert result["search_focus"] == "history" and result["method"] == "hyde_only"


@pytest.mark.parametrize("value", [None, 42, "x" * 201])
def test_bad_focus_rejected(value):
    with pytest.raises(ValueError):
        normalize_focus(value)
