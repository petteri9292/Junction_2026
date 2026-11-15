import numpy as np

from wikilens.embedding_benchmark import batches_by_tokens, sample_indices


def test_sampling_preserves_stratum_counts_and_excludes_duplicate_stress_rows():
    lengths = np.arange(10000)
    selected, strata = sample_indices(lengths, sample_size=200)
    assert selected == sample_indices(lengths, sample_size=200)[0]
    assert len({i for i, _, _ in selected}) == len(selected)
    assert sum(s["population"] for s in strata) == len(lengths)
    assert sum(not stress for _, _, stress in selected) == 200
    for s in strata:
        assert (
            sum(group == s["stratum"] and not stress for _, group, stress in selected)
            == s["sample"]
        )
    assert max(i for i, _, _ in selected) == 9999


def test_batches_cover_each_item_and_respect_padded_memory_budget():
    lengths = [8192, 20, 8192, 4000, 6000, 10, 1024] * 20
    batches = list(batches_by_tokens(lengths))
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    assert all(
        max(lengths[i] for i in batch) * len(batch) <= 16384 for batch in batches
    )
    assert all(len(batch) <= 32 for batch in batches)
