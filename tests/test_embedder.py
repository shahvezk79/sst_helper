"""Tests for sst_navigator.embedder — sanitization logic only.

mlx and transformers are Apple-Silicon-only libraries not available in the
CI environment.  We stub them in sys.modules before importing the embedder
so that the numpy-only helper _sanitize_embedding_batch can be tested
without any hardware dependency.
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

from sst_navigator.embedder import _sanitize_embedding_batch  # noqa: E402


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
