"""Scaling selection, query isolation and reported-statistic checks."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "measure_scaling",
    Path(__file__).resolve().parents[1] / "scripts/measure-scaling.py",
)
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)


def test_subset_is_nested_seeded_language_stratified_and_keeps_every_target_edition():
    articles = [
        dict(lang=lang, page_id=i, wikidata_id="Q1" if i < 3 else f"Q{i}")
        for lang in ("fi", "en", "sv")
        for i in range(1, 31)
    ]
    selected = scaling.subset_keys(articles, {"Q1"}, sizes=(12, 45, 90))
    assert selected == scaling.subset_keys(articles, {"Q1"}, sizes=(12, 45, 90))
    assert set(selected[12]) <= set(selected[45]) <= set(selected[90])
    targets = {(lang, i) for lang in ("fi", "en", "sv") for i in (1, 2)}
    for size, keys in selected.items():
        assert len(keys) == size == len(set(keys))
        assert targets <= set(keys)
        assert {lang for lang, _ in keys} == {"fi", "en", "sv"}
    assert selected != scaling.subset_keys(
        articles, {"Q1"}, sizes=(12, 45, 90), seed=43
    )


def test_sampling_cannot_hide_missing_targets_by_duplicating_articles():
    rows = [dict(lang="fi", page_id=1, wikidata_id="Q1")]
    with pytest.raises(ValueError, match="Duplicate"):
        scaling.subset_keys(rows * 2, {"Q1"}, sizes=(1,))
    with pytest.raises(ValueError, match="fit the original"):
        scaling.subset_keys(rows, {"Q1"}, sizes=(2,))


@pytest.mark.parametrize(
    "name",
    [
        "article_vector",
        "article",
        "_wl_scale_1_1000;DROP TABLE article",
        "_wl_scale_bad_1000",
        "_wl_scale_1_1000_extra",
    ],
)
def test_query_and_cleanup_identifiers_cannot_target_production_tables(name):
    with pytest.raises(ValueError):
        scaling.query_sql(name)
    with pytest.raises(ValueError):
        scaling.table_identifier(name)


def test_benchmark_query_has_own_native_index_and_production_metadata_join_only():
    sql = scaling.query_sql("_wl_scale_123_1000")
    assert "FROM `_wl_scale_123_1000`" in sql
    assert "FROM article_vector" not in sql
    assert "JOIN article a" in sql
    assert "ORDER BY distance LIMIT %s" in sql


def test_summary_uses_nearest_rank_p95_and_keeps_latency_distinct_from_throughput():
    rows = [
        {"sql_ms": i, "expected_rank": 1 if i == 1 else (3 if i < 20 else None)}
        for i in range(1, 21)
    ]
    summary = scaling.summarize(rows, 2)
    assert summary["median_sql_ms"] == 10.5
    assert summary["p95_sql_ms_nearest_rank"] == 19
    assert summary["throughput_queries_per_second"] == 10
    assert summary["hit_at_1_topic"] == 0.05
    assert summary["hit_at_9_articles"] == 0.95
