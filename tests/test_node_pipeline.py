"""Literal passage lineage and cited generation, using mocked inference only."""

import copy
import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wikilens.grounding import GroundingError
from wikilens.node_pipeline import (
    generate_answer,
    load_retrieved,
    rerank_nodes,
    resolve_topics,
    topic_options,
    trim_context,
    validate_answer,
)

QUOTE = "Construction of the fortress began in 1748 under the Swedish Crown."


def article(lang="en", page=1, topic="Q1", text=QUOTE):
    return dict(
        lang=lang,
        page_id=page,
        rev_id=page + 100,
        wikidata_id=topic,
        title=f"Fortress {lang}",
        text=text,
        url=f"https://{lang}.wikipedia.org/wiki/Fortress",
        retrieval_rank=page,
    )


class Runtime:
    """Give relevant text a higher cosine without importing Torch."""

    def __init__(self, output=None):
        self.embeddings = []
        self.generations = []
        self.output = output

    def embed(self, texts, **kwargs):
        self.embeddings.append((texts, kwargs))
        vectors = []
        for i, text in enumerate(texts):
            relevant = i == 0 or "1748" in text or "TARGET" in text
            vectors.append(([1.0, 0.0] if relevant else [0.0, 1.0]) + [0.0] * 1022)
        return vectors

    def generate(self, messages, **kwargs):
        self.generations.append((messages, kwargs))
        return self.output


def nodes():
    ranked = rerank_nodes(Runtime(), "When did construction begin?", [article()])[
        "ranked"
    ]
    return trim_context(Runtime(), "When did construction begin?", ranked)["nodes"]


def answer(source_id="S1", quote=QUOTE):
    return {
        "insufficient_evidence": False,
        "claims": [
            {
                "text": "Construction began in 1748.",
                "evidence": [{"source_id": source_id, "quote": quote}],
            }
        ],
    }


def test_topic_disambiguation_keeps_language_counterparts_and_requires_selection():
    rows = [article("fi", 1), article("en", 2), article("sv", 3, "Q2")]
    before = copy.deepcopy(rows)
    options = topic_options(rows)
    assert [o["topic_id"] for o in options] == ["Q1", "Q2"]
    assert options[0]["languages"] == ["fi", "en"]
    selected = resolve_topics(rows, ["Q1"])
    assert {a["lang"] for a in selected} == {"fi", "en"}
    assert rows == before
    with pytest.raises(ValueError, match="Select topic IDs"):
        resolve_topics(rows, [])
    with pytest.raises(ValueError, match="not in the retrieved"):
        resolve_topics(rows, ["Q999"])


def test_missing_wikidata_does_not_merge_unrelated_articles():
    rows = [article("fi", 1, None), article("en", 2, None)]
    assert {o["topic_id"] for o in topic_options(rows)} == {"fi:1", "en:2"}
    assert resolve_topics(rows, ["en:2"])[0]["page_id"] == 2


def test_aligning_editions_preserves_distinct_source_wording(tmp_path):
    rows = [
        article("fi", 1, text="Rakentaminen alkoi vuonna 1748."),
        article(
            "en", 2, text="Construction started in 1749, according to this passage."
        ),
        article("sv", 3, text="Bygget inleddes 1748."),
    ]
    for row in rows:
        pq.write_table(
            pa.Table.from_pylist([row]), tmp_path / f"articles-{row['lang']}.parquet"
        )
    selected = resolve_topics([rows[0]], ["Q1"], tmp_path)
    assert len(selected) == 3
    assert {a["text"] for a in selected} == {a["text"] for a in rows}
    assert [a["origin"] for a in selected if a["lang"] != "fi"] == [
        "wikidata_counterpart"
    ] * 2


def test_duplicate_revision_conflicts_are_rejected():
    a = article()
    assert len(resolve_topics([a, dict(a)], ["Q1"])) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_topics([a, {**a, "text": "Conflicting content"}], ["Q1"])


def test_load_existing_report_recovers_revision_only_from_exact_text(tmp_path):
    a = article()
    pq.write_table(pa.Table.from_pylist([a]), tmp_path / "articles-en.parquet")
    short = {k: a[k] for k in ("title", "lang", "url", "text")}
    report = tmp_path / "retrieval.json"
    report.write_text(json.dumps({"results": [short]}))
    output = load_retrieved(report, tmp_path)
    assert output[0]["page_id"] == a["page_id"]
    assert output[0]["rev_id"] == a["rev_id"]
    assert output[0]["text"] == QUOTE
    assert json.loads(report.read_text())["results"][0] == short
    short["text"] = "A changed excerpt, not the original full article."
    report.write_text(json.dumps({"results": [short]}))
    with pytest.raises(ValueError, match="exact source text"):
        load_retrieved(report, tmp_path)


def test_load_supports_saved_hyde_and_full_run_reports(tmp_path):
    a = article()
    for payload in (
        {"results": {"hyde": [a]}},
        {"stages": {"retrieval": {"articles": [a]}}},
        [a],
    ):
        report = tmp_path / "retrieval.json"
        report.write_text(json.dumps(payload))
        assert load_retrieved(report, tmp_path)[0]["rev_id"] == a["rev_id"]


def test_reranking_searches_full_article_and_keeps_literal_source_offsets():
    text = (
        ("Unrelated introductory history. " * 130) + QUOTE + (" Later background." * 50)
    )
    a = article(text=text)
    runtime = Runtime()
    result = rerank_nodes(runtime, "When did construction begin?", [a])
    assert result["ranked"][0]["start_char"] > 1200
    assert "1748" in result["ranked"][0]["text"]
    for n in result["ranked"]:
        assert a["text"][n["start_char"] : n["end_char"]] == n["text"]
        assert n["revision_url"] == "https://en.wikipedia.org/w/index.php?oldid=101"
    assert runtime.embeddings[0][1] == {"batch_size": 1, "max_tokens": 1024}
    with pytest.raises(ValueError, match="explicit limit"):
        rerank_nodes(Runtime(), "When?", [a], max_candidates=1)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_embedding_is_rejected(bad_value):
    runtime = Runtime()
    runtime.embed = lambda texts, **kw: [[bad_value] + [0.0] * 1023 for _ in texts]
    with pytest.raises(ValueError, match="finite 1024"):
        rerank_nodes(runtime, "When?", [article()])


def test_trimmed_context_preserves_citation_lineage_and_budgets():
    text = (
        ("General background is not the requested answer. " * 30)
        + QUOTE
        + (" More background." * 30)
    )
    rows = [
        article("fi", 1, text=text),
        article("en", 2, text=text),
        article("sv", 3, text=text),
    ]
    original = {(a["lang"], a["page_id"], a["rev_id"]): a["text"] for a in rows}
    ranking = rerank_nodes(Runtime(), "When did construction begin?", rows)["ranked"]
    context = trim_context(
        Runtime(),
        "When did construction begin?",
        ranking,
        max_nodes=4,
        node_chars=300,
        context_chars=850,
        max_per_article=2,
    )
    selected = context["nodes"]
    assert 1 <= len(selected) <= 4
    assert sum(len(n["text"]) for n in selected) <= 850
    assert all(len(n["text"]) <= 300 for n in selected)
    assert max(Counter((n["lang"], n["page_id"]) for n in selected).values()) <= 2
    assert [n["id"] for n in selected] == [f"S{i}" for i in range(1, len(selected) + 1)]
    for n in selected:
        source = original[(n["lang"], n["page_id"], n["rev_id"])]
        assert source[n["start_char"] : n["end_char"]] == n["text"]
        assert n["node_id"].endswith(f"-{n['start_char']}-{n['end_char']}")
        assert (
            n["parent_start_char"]
            <= n["start_char"]
            < n["end_char"]
            <= n["parent_end_char"]
        )
    for i, n in enumerate(selected):
        for m in selected[i + 1 :]:
            if (n["lang"], n["page_id"]) == (m["lang"], m["page_id"]):
                overlap = max(
                    0,
                    min(n["end_char"], m["end_char"])
                    - max(n["start_char"], m["start_char"]),
                )
                assert overlap / min(len(n["text"]), len(m["text"])) <= 0.5


def test_trim_can_find_relevant_tail_of_long_unpunctuated_span():
    a = article(
        text=("background " * 85) + "TARGET evidence matters here " + ("tail " * 20)
    )
    ranking = rerank_nodes(Runtime(), "TARGET question", [a])["ranked"]
    context = trim_context(
        Runtime(), "TARGET question", ranking, node_chars=300, inspect_top=1
    )
    assert any("TARGET" in n["text"] for n in context["nodes"])


def test_valid_answer_links_each_claim_to_literal_quotation():
    selected = nodes()
    checked = validate_answer(answer(), selected)
    assert checked["status"] == "draft"
    assert checked["claim_support_reviewed"] is False
    support = checked["claims"][0]["evidence"][0]
    assert support["quote"] == QUOTE
    assert support["revision_url"] == selected[0]["revision_url"]
    assert support["quote_article_start_char"] >= selected[0]["start_char"]


@pytest.mark.parametrize(
    "payload",
    [
        answer("invented"),
        answer(quote="This quotation was invented and is absent from every passage."),
        {
            "insufficient_evidence": False,
            "claims": [{"text": "Uncited answer.", "evidence": []}],
        },
        {"insufficient_evidence": True, "claims": answer()["claims"]},
        {
            "insufficient_evidence": False,
            "claims": [
                {
                    "text": "<script>bad</script>",
                    "evidence": answer()["claims"][0]["evidence"],
                }
            ],
        },
    ],
)
def test_invalid_generated_citations_or_quotes_reject_the_entire_answer(payload):
    with pytest.raises((GroundingError, ValueError)):
        validate_answer(payload, nodes())


def test_one_invalid_claim_does_not_display_a_valid_prefix():
    payload = answer()
    payload["claims"].append(
        {
            "text": "A second claim.",
            "evidence": [{"source_id": "unknown", "quote": QUOTE}],
        }
    )
    with pytest.raises(GroundingError):
        validate_answer(payload, nodes())


def test_generation_receives_only_selected_evidence_and_preserves_raw_output():
    selected = nodes()
    raw = json.dumps(answer())
    runtime = Runtime(raw)
    result = generate_answer(runtime, "When was construction begun?", selected)
    assert result["status"] == "draft"
    assert result["raw_output"] == raw
    messages, config = runtime.generations[0]
    context = json.loads(messages[1]["content"])
    assert context["question"] == "When was construction begun?"
    assert context["passages"] == [
        {
            "source_id": n["id"],
            "title": n["title"],
            "lang": n["lang"],
            "text": n["text"],
        }
        for n in selected
    ]
    assert "hyde" not in context and "hypothetical" not in context
    assert config["max_new_tokens"] == 4000 and config["max_time"] == 300
    assert config["output_schema"]["$defs"]["Claim"]["required"] == ["text", "evidence"]
    assert config["output_schema"]["$defs"]["Evidence"]["properties"]["source_id"][
        "enum"
    ] == [n["id"] for n in selected]
    bad = Runtime(json.dumps(answer("invented")))
    rejected = generate_answer(bad, "When?", selected)
    assert rejected["status"] == "draft_unverified" and rejected["claims"] == []
    assert rejected["citation_validation"] == "failed"
    assert rejected["draft_text"] == rejected["raw_output"]
    assert "invented" in rejected["raw_output"]


def test_empty_context_abstains_without_model_call_and_empty_nodes_need_no_embedding():
    runtime = Runtime("unused")
    assert generate_answer(runtime, "When?", [])["status"] == "insufficient_evidence"
    assert rerank_nodes(runtime, "When?", [])["ranked"] == []
    assert trim_context(runtime, "When?", [])["nodes"] == []
    assert runtime.generations == [] and runtime.embeddings == []
    assert (
        validate_answer({"insufficient_evidence": True, "claims": []}, [])["status"]
        == "insufficient_evidence"
    )


def test_trimmed_citation_reopens_original_context_and_rejects_tampering():
    from wikilens.node_pipeline import citation_context

    article = {
        "lang": "fi",
        "page_id": 1,
        "rev_id": 2,
        "title": "Topic",
        "text": "Before. Selected literal passage. After.",
    }
    start = article["text"].index("Selected")
    end = article["text"].index(" After.")
    node = {
        **article,
        "id": "S1",
        "start_char": start,
        "end_char": end,
        "text": article["text"][start:end],
        "revision_url": "https://fi.wikipedia.org/w/index.php?oldid=2",
    }
    context = citation_context([node], [article], "S1", margin=100)
    assert context["before"] + context["passage"] + context["after"] == article["text"]
    assert context["revision_url"].endswith("oldid=2")
    with pytest.raises(ValueError, match="does not match"):
        citation_context([{**node, "text": "Invented replacement"}], [article], "S1")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        citation_context([node], [article], "unknown")
