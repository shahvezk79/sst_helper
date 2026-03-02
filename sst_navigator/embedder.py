"""
Stage 1 — Semantic search via bi-encoder embeddings.

Supports two backends:
  1. Local MLX model (default) — runs on Apple Silicon via mlx-lm with
     manual hidden-state extraction and last-token pooling (the official
     Qwen3 embedding strategy).
  2. DeepInfra API — set DEEPINFRA_API_KEY in env; uses the OpenAI-
     compatible embeddings endpoint with Qwen3-Embedding-8B.
"""

import logging
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from . import config

logger = logging.getLogger(__name__)


def _sanitize_embedding_batch(result: np.ndarray) -> None:
    """Zero out rows containing NaN/Inf in *result* in-place.

    Called after converting MLX tensors to numpy so that downstream
    matmul operations never encounter invalid float values.  Rows
    replaced with zeros will produce a cosine similarity of 0 and
    effectively rank last in search results.
    """
    bad_rows = ~np.isfinite(result).all(axis=1)
    if bad_rows.any():
        n_bad = int(bad_rows.sum())
        logger.warning(
            "%d of %d embedding vectors contain NaN/Inf — replacing with zeros.",
            n_bad,
            result.shape[0],
        )
        result[bad_rows] = 0.0


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a copy of *arr* with every row L2-normalised.

    Zero-norm rows (e.g. previously sanitised bad vectors) are left as
    zeros rather than producing NaN from 0/0.  This guarantees that the
    dot-product of any two rows stays in [-1, 1] and cannot overflow
    float32 during matmul accumulation.

    The norm is computed in float64 to avoid overflow when squaring
    large float32 values (e.g. 1e20² exceeds float32 max).
    """
    arr64 = arr.astype(np.float64)
    norms = np.linalg.norm(arr64, axis=1, keepdims=True)
    # Clamp to avoid 0/0; zero-norm rows stay as zeros.
    norms = np.maximum(norms, 1e-12)
    return (arr64 / norms).astype(np.float32)


def _sanitize_and_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a sanitized + unit-normalized copy of *arr*.

    This is the one-time preparation step for embedding matrices loaded
    from cache or produced by MLX.  Keeping this off the query hot path
    avoids repeated O(n·d) work on every search.
    """
    non_finite_mask = ~np.isfinite(arr)
    n_non_finite = int(non_finite_mask.sum())
    if n_non_finite:
        n_bad_rows = int(non_finite_mask.any(axis=1).sum())
        logger.warning(
            "Sanitizing %d non-finite embedding values across %d rows.",
            n_non_finite,
            n_bad_rows,
        )

    clean = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    return _l2_normalize_rows(clean)


# ---------------------------------------------------------------------------
# Qwen3 embedding model wrapper — strips lm_head so we get hidden states
# (MLX backend only)
# ---------------------------------------------------------------------------

def _get_embedding_model_classes(model_config: dict):
    """Return a (ModelClass, ModelArgs) tuple that strips the lm_head."""
    import importlib
    import mlx.core as mx  # noqa: F401 — used by the generated class

    model_type = model_config.get("model_type", "qwen3").lower()
    # mlx-lm stores arch implementations by model_type
    try:
        arch_module = importlib.import_module(f"mlx_lm.models.{model_type}")
    except ModuleNotFoundError:
        # Fall back to qwen2 which shares the same architecture
        arch_module = importlib.import_module("mlx_lm.models.qwen2")

    class Qwen3EmbeddingModel(arch_module.Model):
        """Thin wrapper that removes lm_head and returns hidden states."""

        def __init__(self, args):
            super().__init__(args)
            # Embedding models don't ship lm_head weights — delete so
            # load_model doesn't complain about missing tensors.
            if hasattr(self, "lm_head"):
                delattr(self, "lm_head")

        def __call__(
            self,
            inputs,
            cache=None,
            input_embeddings=None,
        ):
            # self.model is the inner transformer (Qwen2Model / Qwen3Model)
            return self.model(inputs, cache=cache, input_embeddings=input_embeddings)

    return Qwen3EmbeddingModel, arch_module.ModelArgs


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SemanticSearcher:
    """Embeds documents and queries, returns top-K by cosine similarity.

    Args:
        model_name: HuggingFace model ID (used by the MLX backend only).
        backend: ``"mlx"`` (default, local Apple Silicon inference) or
            ``"deepinfra"`` (cloud API, requires ``DEEPINFRA_API_KEY``).
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        backend: str = "mlx",
    ):
        self.model_name = model_name
        self.backend = backend
        self.model = None
        self.tokenizer = None
        # Stored document embeddings (numpy for fast cosine on CPU)
        self._doc_embeddings: np.ndarray | None = None
        # URL list from the cache, used for alignment then discarded
        self._cache_urls: list[str] | None = None

  # -- Persistent cache --------------------------------------------------

    def load_embeddings_cache(
        self,
        cache_dir: str = config.EMBEDDING_CACHE_DIR,
        repo_id: str = config.EMBEDDING_CACHE_REPO_ID,
        repo_type: str = config.EMBEDDING_CACHE_REPO_TYPE,
        embeddings_file: str = config.EMBEDDING_CACHE_FILE,
        metadata_file: str = config.EMBEDDING_METADATA_FILE,
    ) -> bool:
        """Load cached embeddings from local disk or Hugging Face dataset cache.

        Downloads the embedding vectors and metadata from the HuggingFace
        dataset repo if they are not already present locally.  The metadata
        URL list is stored internally so that :meth:`align_to_urls` can
        later reorder the vectors to match the active dataframe.
        """
        base = Path(cache_dir)
        base.mkdir(parents=True, exist_ok=True)
        emb_path = base / embeddings_file
        meta_path = base / metadata_file

        if not emb_path.exists() or not meta_path.exists():
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=embeddings_file,
                    repo_type=repo_type,
                    local_dir=str(base),
                    local_dir_use_symlinks=False,
                )
                hf_hub_download(
                    repo_id=repo_id,
                    filename=metadata_file,
                    repo_type=repo_type,
                    local_dir=str(base),
                    local_dir_use_symlinks=False,
                )
                logger.info("Downloaded embedding cache from %s", repo_id)
            except Exception as exc:
                logger.warning(
                    "Could not download embedding cache from %s: %s", repo_id, exc
                )

        if not emb_path.exists() or not meta_path.exists():
            return False

        # Parse metadata to get the URL ordering of the cached embeddings.
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Embedding metadata file is invalid JSON: %s", exc)
            return False

        if isinstance(raw, list):
            self._cache_urls = [str(u) for u in raw]
        elif isinstance(raw, dict) and "url_en" in raw:
            self._cache_urls = [str(u) for u in raw["url_en"]]
        else:
            logger.error("Unexpected metadata format — expected list or {url_en: [...]}")
            return False

        self._doc_embeddings = np.load(emb_path, allow_pickle=False).astype(
            np.float32, copy=False,
        )

        # Validate that the two files are consistent.
        if self._doc_embeddings.shape[0] != len(self._cache_urls):
            logger.error(
                "Embedding/metadata mismatch: %d vectors vs %d URLs. "
                "Cache may be corrupted — delete %s and retry.",
                self._doc_embeddings.shape[0],
                len(self._cache_urls),
                base,
            )
            self._doc_embeddings = None
            self._cache_urls = None
            return False

        n_unique = len(set(self._cache_urls))
        if n_unique < len(self._cache_urls):
            n_dups = len(self._cache_urls) - n_unique
            logger.warning(
                "Metadata contains %d duplicate URLs (total %d). "
                "Deduplicating — keeping first occurrence of each URL.",
                n_dups,
                len(self._cache_urls),
            )
            seen: set[str] = set()
            keep: list[int] = []
            for i, url in enumerate(self._cache_urls):
                if url not in seen:
                    seen.add(url)
                    keep.append(i)
            self._doc_embeddings = self._doc_embeddings[np.array(keep)]
            self._cache_urls = [self._cache_urls[i] for i in keep]

        logger.info(
            "Loaded embedding cache from %s — %d vectors, %d metadata URLs",
            emb_path,
            self._doc_embeddings.shape[0],
            len(self._cache_urls),
        )
        return True

    def align_to_urls(self, target_urls: list[str]) -> list[int]:
        """Reorder cached embeddings to match *target_urls*.

        Only URLs present in both the cache and *target_urls* are kept.
        The returned list may be shorter than *target_urls* if some URLs
        have no cached embedding (e.g. newly added decisions not yet in
        the cache).  The caller should use these indices to trim its own
        data structures so that row *i* in the dataframe corresponds to
        row *i* in the embedding matrix.

        The cached embedding vectors may be in a different order than
        *target_urls* (``scripts/update_index.py`` sorts by text length
        for efficient batching).  This method handles the reordering.
        """
        if self._doc_embeddings is None or self._cache_urls is None:
            raise RuntimeError("Call load_embeddings_cache() first.")

        if not target_urls:
            raise ValueError("target_urls is empty — nothing to align to.")

        cache_lookup = {url: i for i, url in enumerate(self._cache_urls)}

        matched_target_indices: list[int] = []
        matched_cache_indices: list[int] = []
        for target_idx, url in enumerate(target_urls):
            cache_idx = cache_lookup.get(url)
            if cache_idx is not None:
                matched_target_indices.append(target_idx)
                matched_cache_indices.append(cache_idx)

        if not matched_cache_indices:
            raise RuntimeError(
                "No cached embeddings match the target URL list. "
                "The cache may be stale — run scripts/update_index.py to refresh."
            )

        self._doc_embeddings = self._doc_embeddings[np.array(matched_cache_indices)]
        self._cache_urls = None  # Free memory; no longer needed

        # One-time sanitisation + re-normalisation on load/alignment.
        # Cached fp8 vectors can contain invalid values or norm drift.
        self._doc_embeddings = _sanitize_and_normalize_rows(self._doc_embeddings)

        logger.info(
            "Aligned embeddings: %d of %d target URLs matched.",
            len(matched_target_indices),
            len(target_urls),
        )
        return matched_target_indices

    # -- Model lifecycle ---------------------------------------------------

    def load_model(self) -> None:
        """Download (if needed) and load the embedding model into MLX.

        No-op when ``backend == "deepinfra"`` — the API needs no local model.
        """
        if self.backend != "mlx":
            return

        import mlx.core as mx  # noqa: F401 — imported for side-effects
        from mlx_lm.utils import load_model as mlx_load_model, _download
        from transformers import AutoTokenizer

        logger.info("Loading embedding model %s …", self.model_name)
        try:
            model_path = _download(self.model_name)
            self.model, _ = mlx_load_model(
                model_path=model_path,
                get_model_classes=_get_embedding_model_classes,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            raise

    def unload_model(self) -> None:
        """Free the model from memory (useful before loading the reranker).

        No-op when ``backend == "deepinfra"``.
        """
        if self.backend != "mlx":
            return

        import mlx.core as mx

        self.model = None
        self.tokenizer = None
        mx.clear_cache()
        logger.info("Embedding model unloaded.")

    # -- Embedding logic ---------------------------------------------------

    def _embed_batch(self, texts: list[str], max_tokens: int) -> np.ndarray:
        """Embed a list of texts and return L2-normalised numpy vectors."""
        if self.backend == "deepinfra":
            return self._embed_batch_deepinfra(texts)
        return self._embed_batch_mlx(texts, max_tokens)

    def _embed_batch_deepinfra(self, texts: list[str]) -> np.ndarray:
        """Embed texts via the DeepInfra OpenAI-compatible embeddings API."""
        from openai import OpenAI

        api_key = os.environ.get("DEEPINFRA_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPINFRA_API_KEY in your environment.")

        client = OpenAI(api_key=api_key, base_url=config.DEEPINFRA_BASE_URL)
        response = client.embeddings.create(
            input=texts,
            model=config.DEEPINFRA_EMBEDDING_MODEL,
            encoding_format="float",
        )
        vectors = np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )
        _sanitize_embedding_batch(vectors)
        return vectors

    def _embed_batch_mlx(self, texts: list[str], max_tokens: int) -> np.ndarray:
        """Embed texts using the local MLX model."""
        import mlx.core as mx
        from transformers import AutoTokenizer  # noqa: F401 — already loaded

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        hidden_states = self.model(input_ids)  # (batch, seq, hidden)

        # Last-token pooling (official Qwen3 strategy)
        seq_lengths = mx.sum(attention_mask, axis=1) - 1
        seq_lengths = mx.maximum(seq_lengths, 0)
        batch_idx = mx.arange(hidden_states.shape[0])
        pooled = hidden_states[batch_idx, seq_lengths]  # (batch, hidden)

        # ---- Upcast to float32 BEFORE normalisation ----
        # The mxfp8 model computes activations in float16 (max 65 504).
        # L2-norm involves squaring elements: any value > ~256 causes
        # x² > 65 504 → Inf in float16, cascading to NaN through the
        # entire vector.  Upcasting first keeps the norm in float32
        # (max ~3.4e38) where overflow is effectively impossible.
        pooled = pooled.astype(mx.float32)

        # L2 normalise (now in float32 — stable for any float16-range input)
        norms = mx.linalg.norm(pooled, axis=-1, keepdims=True)
        normed = pooled / mx.maximum(norms, mx.array(1e-9, dtype=mx.float32))
        mx.eval(normed)

        result = np.array(normed, dtype=np.float32)

        # Safety net: the float32 upcast above prevents normalization
        # overflow, but the fp8 forward pass itself can still produce
        # NaN/Inf hidden states in rare cases.  Zero those rows so the
        # downstream matmul never sees non-finite values.
        _sanitize_embedding_batch(result)

        # Eagerly free large MLX tensors so Metal memory is reclaimable
        # *before* the next batch allocates.  Without this, hidden_states
        # (~235 MB at batch_size=4 / 8192 tokens) lingers until Python's
        # reference counter catches up, fragmenting the Metal heap.
        del encoded, input_ids, attention_mask, hidden_states
        del seq_lengths, batch_idx, pooled, norms, normed

        return result

    # -- Public API --------------------------------------------------------

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 4,
        max_tokens: int = config.EMBEDDING_MAX_TOKENS,
        progress_callback=None,
        progress_start: int = 0,
        progress_total: int | None = None,
    ) -> np.ndarray:
        """Compute embeddings for a list of texts and return vectors."""
        all_embeddings: list[np.ndarray] = []
        total = len(texts)
        progress_total = progress_total if progress_total is not None else total

        if total == 0:
            return np.empty((0, 0), dtype=np.float32)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]
            emb = self._embed_batch(batch, max_tokens)
            all_embeddings.append(emb)

            if self.backend == "mlx":
                import mlx.core as mx
                mx.clear_cache()

            if progress_callback:
                progress_callback(progress_start + end, progress_total)
            logger.debug("Embedded %d / %d documents", end, total)

        return np.concatenate(all_embeddings, axis=0)

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 4,
        max_tokens: int = config.EMBEDDING_MAX_TOKENS,
        progress_callback=None,
    ) -> None:
        """Compute and cache embeddings for all decision texts.

        Args:
            texts: The full decision texts to embed.
            batch_size: How many texts to embed per forward pass.
            max_tokens: Truncation length per text.
            progress_callback: Optional callable(current, total) for UI updates.
        """
        total = len(texts)

        if total == 0:
            logger.warning("No documents were provided for embedding.")
            self._doc_embeddings = np.empty((0, 0), dtype=np.float32)
            return

        self._doc_embeddings = self.embed_texts(
            texts,
            batch_size=batch_size,
            max_tokens=max_tokens,
            progress_callback=progress_callback,
        )
        # Re-normalise to ensure unit length (MLX fp8 normalisation can
        # be imprecise), matching what align_to_urls does for cached vectors.
        self._doc_embeddings = _sanitize_and_normalize_rows(self._doc_embeddings)
        logger.info(
            "Document embeddings cached — shape %s", self._doc_embeddings.shape
        )

    def search(
        self,
        query: str,
        top_k: int = config.STAGE1_TOP_K,
        max_tokens: int = config.EMBEDDING_MAX_TOKENS,
    ) -> list[tuple[int, float]]:
        """Return the top-K (index, score) pairs for a query.

        The query is automatically wrapped with the retrieval instruction.
        """
        if self._doc_embeddings is None:
            raise RuntimeError("Call embed_documents() before search().")
        if self._doc_embeddings.shape[0] == 0:
            return []

        formatted_query = (
            f"Instruct: {config.EMBEDDING_INSTRUCTION}\n"
            f"Query:{query}"
        )
        q_vec = self._embed_batch([formatted_query], max_tokens)
        # Ensure finite + unit-length query vector.  _embed_batch already
        # sanitises obvious NaN/Inf rows, but we do a final defensive pass
        # here because query encoding can still drift on fp8 models.
        q_vec = _sanitize_and_normalize_rows(q_vec)

        # Cached vectors are sanitised during align_to_urls(), but keep a
        # guardrail in case callers inject embeddings manually in tests or
        # legacy code paths bypass alignment.
        if not np.isfinite(self._doc_embeddings).all():
            logger.warning(
                "Detected non-finite values in cached document embeddings at search time; "
                "re-sanitizing in-memory matrix."
            )
            self._doc_embeddings = _sanitize_and_normalize_rows(self._doc_embeddings)

        # Cosine similarity (vectors are unit-normalised → dot product).
        # np.errstate suppresses spurious RuntimeWarnings that Apple's
        # Accelerate BLAS (AMX-backed SGEMM) can emit during float32
        # matmul even when both operands are finite unit vectors.  The
        # post-matmul nan_to_num below handles any genuinely non-finite
        # scores that may result on affected platforms.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = self._doc_embeddings @ q_vec.T  # (n_docs, 1)
        scores = scores.squeeze(-1)

        # Replace any residual NaN (e.g. 0-vec · 0-vec, or platform-
        # specific BLAS artifacts) with -1 so they sort last instead of
        # polluting argsort.
        n_bad_scores = int((~np.isfinite(scores)).sum())
        if n_bad_scores:
            logger.warning(
                "matmul produced %d non-finite similarity scores out of %d "
                "(likely a platform BLAS artifact); clamping to -1.",
                n_bad_scores,
                scores.shape[0],
            )
        np.nan_to_num(scores, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    @property
    def has_embeddings(self) -> bool:
        return self._doc_embeddings is not None and self._doc_embeddings.shape[0] > 0
