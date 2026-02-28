"""Tests for sst_navigator.embedder — sanitization and normalisation helpers.

mlx and transformers are Apple-Silicon-only libraries not available in the
CI environment.  We stub them in sys.modules before importing the embedder
so that the numpy-only helpers can be tested without any hardware dependency.
"""

import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub out Apple-Silicon-only (and other heavy) dependencies so the module
# can be imported in any environment.
# ---------------------------------------------------------------------------

def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]


# mlx — Apple Silicon ML framework
_mlx_core = _stub("mlx.core")
_mlx = _stub("mlx")
_mlx.core = _mlx_core  # type: ignore[attr-defined]

# mlx_lm — needs load_model and _download importable
_mlx_lm_utils = _stub("mlx_lm.utils")
_mlx_lm_utils.load_model = None  # type: ignore[attr-defined]
_mlx_lm_utils._download = None  # type: ignore[attr-defined]
_stub("mlx_lm")

# huggingface_hub — needs hf_hub_download importable
_hf_hub = _stub("huggingface_hub")
_hf_hub.hf_hub_download = None  # type: ignore[attr-defined]

# transformers — needs AutoTokenizer importable
_transformers = _stub("transformers")
_transformers.AutoTokenizer = None  # type: ignore[attr-defined]

from sst_navigator.embedder import (  # noqa: E402
    _sanitize_embedding_batch,
    _l2_normalize_rows,
    _sanitize_and_normalize_rows,
)


# ---------------------------------------------------------------------------
# _sanitize_embedding_batch
# ---------------------------------------------------------------------------

class TestSanitizeEmbeddingBatch:
    def test_all_finite_rows_unchanged(self):
        data = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        original = data.copy()
        _sanitize_embedding_batch(data)
        np.testing.assert_array_equal(data, original)

    def test_nan_row_replaced_with_zeros(self):
        data = np.array([[1.0, 0.0], [np.nan, 0.5], [0.0, 1.0]], dtype=np.float32)
        _sanitize_embedding_batch(data)
        np.testing.assert_array_equal(data[1], [0.0, 0.0])
        # Adjacent valid rows are untouched
        np.testing.assert_array_equal(data[0], [1.0, 0.0])
        np.testing.assert_array_equal(data[2], [0.0, 1.0])

    def test_pos_inf_row_replaced_with_zeros(self):
        data = np.array([[0.6, 0.8], [np.inf, 1.0], [1.0, 0.0]], dtype=np.float32)
        _sanitize_embedding_batch(data)
        np.testing.assert_array_equal(data[1], [0.0, 0.0])

    def test_neg_inf_row_replaced_with_zeros(self):
        data = np.array([[0.6, 0.8], [-np.inf, 1.0]], dtype=np.float32)
        row0_before = data[0].copy()
        _sanitize_embedding_batch(data)
        np.testing.assert_array_equal(data[0], row0_before)  # valid row unchanged
        np.testing.assert_array_equal(data[1], [0.0, 0.0])  # -inf row zeroed

    def test_multiple_bad_rows_all_zeroed(self):
        data = np.array(
            [[np.nan, 1.0], [0.5, 0.5], [np.inf, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        _sanitize_embedding_batch(data)
        np.testing.assert_array_equal(data[0], [0.0, 0.0])
        np.testing.assert_array_equal(data[2], [0.0, 0.0])
        # Valid rows survive
        np.testing.assert_array_equal(data[1], [0.5, 0.5])
        np.testing.assert_array_equal(data[3], [0.0, 1.0])

    def test_single_row_nan(self):
        data = np.array([[np.nan, np.nan]], dtype=np.float32)
        _sanitize_embedding_batch(data)
        np.testing.assert_array_equal(data[0], [0.0, 0.0])

    def test_modifies_in_place(self):
        data = np.array([[np.inf, 0.0], [0.0, 1.0]], dtype=np.float32)
        original_id = id(data)
        _sanitize_embedding_batch(data)
        assert id(data) == original_id
        np.testing.assert_array_equal(data[0], [0.0, 0.0])

    def test_warning_emitted_for_bad_rows(self, caplog):
        import logging
        data = np.array([[np.nan, 1.0], [0.0, 1.0]], dtype=np.float32)
        with caplog.at_level(logging.WARNING, logger="sst_navigator.embedder"):
            _sanitize_embedding_batch(data)
        assert any("NaN/Inf" in r.message for r in caplog.records)

    def test_no_warning_for_clean_data(self, caplog):
        import logging
        data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with caplog.at_level(logging.WARNING, logger="sst_navigator.embedder"):
            _sanitize_embedding_batch(data)
        assert not any("NaN/Inf" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _l2_normalize_rows
# ---------------------------------------------------------------------------

class TestL2NormalizeRows:
    def test_unit_vectors_unchanged(self):
        """Rows that are already unit-length should stay the same."""
        data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = _l2_normalize_rows(data)
        np.testing.assert_allclose(result, data, atol=1e-7)

    def test_non_unit_vectors_normalised(self):
        """Rows with norms != 1 should become unit-length."""
        data = np.array([[3.0, 4.0], [0.0, 10.0]], dtype=np.float32)
        result = _l2_normalize_rows(data)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
        # Direction preserved
        np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)
        np.testing.assert_allclose(result[1], [0.0, 1.0], atol=1e-6)

    def test_zero_row_stays_zero(self):
        """A zero vector (e.g. sanitised NaN row) must remain zeros, not NaN."""
        data = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        result = _l2_normalize_rows(data)
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
        np.testing.assert_allclose(np.linalg.norm(result[1]), 1.0, atol=1e-6)

    def test_very_large_vectors_normalised(self):
        """Vectors with very large magnitudes should normalise without overflow."""
        data = np.array([[1e20, 1e20], [1e30, 0.0]], dtype=np.float32)
        result = _l2_normalize_rows(data)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
        assert np.all(np.isfinite(result))

    def test_output_is_float32(self):
        """Result must be float32 regardless of input dtype."""
        data = np.array([[3.0, 4.0]], dtype=np.float64)
        result = _l2_normalize_rows(data)
        assert result.dtype == np.float32

    def test_returns_new_array(self):
        """Should return a new array, not modify in-place."""
        data = np.array([[3.0, 4.0]], dtype=np.float32)
        result = _l2_normalize_rows(data)
        # Original unchanged
        np.testing.assert_array_equal(data[0], [3.0, 4.0])
        np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)

    def test_normalised_matmul_no_overflow(self):
        """After normalisation, matmul of two sets of vectors must not overflow."""
        import warnings
        # Simulate the exact scenario: large doc embeddings × unit query
        docs = np.array(
            [[1e18, 1e18, 1e18], [1e20, -1e20, 1e20]],
            dtype=np.float32,
        )
        query = np.array([[0.6, 0.0, 0.8]], dtype=np.float32)
        docs_normed = _l2_normalize_rows(docs)
        query_normed = _l2_normalize_rows(query)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scores = docs_normed @ query_normed.T
        assert np.all(np.isfinite(scores))
        assert np.all(np.abs(scores) <= 1.0 + 1e-6)


class TestSanitizeAndNormalizeRows:
    def test_replaces_invalid_values_then_normalizes(self):
        rows = np.array(
            [[np.inf, 2.0, 0.0], [np.nan, -1.0, 1.0], [3.0, 4.0, 0.0]],
            dtype=np.float32,
        )

        result = _sanitize_and_normalize_rows(rows)

        assert np.all(np.isfinite(result))
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-6)

    def test_zero_vectors_stay_zero(self):
        rows = np.array([[0.0, 0.0], [np.nan, np.nan]], dtype=np.float32)

        result = _sanitize_and_normalize_rows(rows)

        np.testing.assert_array_equal(result, np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32))

    def test_warning_emitted_when_sanitization_changes_values(self, caplog):
        import logging

        rows = np.array([[1.0, np.inf], [np.nan, 0.0]], dtype=np.float32)

        with caplog.at_level(logging.WARNING, logger="sst_navigator.embedder"):
            _sanitize_and_normalize_rows(rows)

        assert any("Sanitizing" in r.message for r in caplog.records)

    def test_no_warning_when_rows_are_finite(self, caplog):
        import logging

        rows = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)

        with caplog.at_level(logging.WARNING, logger="sst_navigator.embedder"):
            _sanitize_and_normalize_rows(rows)

        assert not any("Sanitizing" in r.message for r in caplog.records)
