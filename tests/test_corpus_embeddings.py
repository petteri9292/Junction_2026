import numpy as np
import pytest

from wikilens.corpus_embeddings import digest_config, validate_vectors, vector_table


def rows():
    return [
        {
            "lang": "fi",
            "page_id": 42,
            "rev_id": 84,
            "source_row": 0,
            "truncated": False,
            "input_ids": [0, 20, 2],
        }
    ]


def test_resume_config_hash_is_order_independent_and_sensitive_to_model_revision():
    assert digest_config({"a": 1, "b": 2}) == digest_config({"b": 2, "a": 1})
    assert digest_config({"revision": "one"}) != digest_config({"revision": "two"})


def test_checkpoint_rejects_wrong_source_and_nonfinite_vectors():
    matrix = np.zeros((1, 1024), dtype=np.float32)
    matrix[0, 0] = 1
    table = vector_table(rows(), matrix)
    assert validate_vectors(table, rows())["rows"] == 1
    with pytest.raises(ValueError, match="page_id"):
        validate_vectors(table, [rows()[0] | {"page_id": 43}])
    matrix[0, 1] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        validate_vectors(vector_table(rows(), matrix), rows())


def test_checkpoint_rejects_unnormalized_vectors():
    with pytest.raises(ValueError, match="Non-unit"):
        validate_vectors(vector_table(rows(), np.ones((1, 1024))), rows())
