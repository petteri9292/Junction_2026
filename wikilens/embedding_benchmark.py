"""Pure sampling and batching helpers for the L embedding benchmark."""

import numpy as np


def sample_indices(lengths, sample_size=2000, seed=20261114):
    """Sample equal-count character-length deciles; retain population weights."""
    lengths = np.asarray(lengths)
    if sample_size < 10 or sample_size > len(lengths):
        raise ValueError("Sample must contain 10 to population-size articles")
    rng = np.random.default_rng(seed)
    selected, strata = [], []
    for stratum, indices in enumerate(
        np.array_split(np.argsort(lengths, kind="stable"), 10)
    ):
        count = sample_size // 10 + (stratum < sample_size % 10)
        chosen = rng.choice(indices, count, replace=False)
        selected.extend((int(i), stratum, False) for i in chosen)
        strata.append(
            {
                "stratum": stratum,
                "population": len(indices),
                "sample": count,
                "min_chars": int(lengths[indices].min()),
                "max_chars": int(lengths[indices].max()),
            }
        )
    already = {i for i, _, _ in selected}
    selected.extend(
        (int(i), 9, True) for i in np.argsort(lengths)[-8:] if i not in already
    )
    return selected, strata


def batches_by_tokens(lengths, token_budget=16384, max_batch=32):
    """Sort by length and cap padded tokens; caller supplies capped lengths."""
    if any(n < 1 or n > token_budget for n in lengths):
        raise ValueError("Each item must fit the positive token budget")
    batch = []
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        if batch and (
            len(batch) >= max_batch or lengths[i] * (len(batch) + 1) > token_budget
        ):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch
