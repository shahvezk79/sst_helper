"""
Stage 1 — Semantic search via bi-encoder embeddings.

Uses the Qwen3-Embedding model through mlx-lm with manual hidden-state
extraction and last-token pooling (the official Qwen3 embedding strategy).
"""

import logging
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from mlx_lm.utils import load_model, _download
from transformers import AutoTokenizer

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen3 embedding model wrapper — strips lm_head so we get hidden states
# ---------------------------------------------------------------------------

def _get_embedding_model_classes(config: dict):
    """Return a (ModelClass, ModelArgs) tuple that strips the lm_head."""
    import importlib

    model_type = config.get("model_type", "qwen3").lower()
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
            inputs: mx.array,
            cache=None,
            input_embeddings: mx.array | None = None,
        ) -> mx.array:
            # self.model is the inner transformer (Qwen2Model / Qwen3Model)
            return self.model(inputs, cache=cache, input_embeddings=input_embeddings)

    return Qwen3EmbeddingModel, arch_module.ModelArgs


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SemanticSearcher:
    """Embeds documents and queries, returns top-K by cosine similarity."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        # Stored document embeddings (numpy for fast cosine on CPU)
        self._doc_embeddings: np.ndarray | None = None

  # -- Persistent cache --------------------------------------------------

    def load_embeddings_cache(
        self,
        cache_dir: str = config.EMBEDDING_CACHE_DIR,
        repo_id: str = config.EMBEDDING_CACHE_REPO_ID,
        repo_type: str = config.EMBEDDING_CACHE_REPO_TYPE,
        embeddings_file: str = config.EMBEDDING_CACHE_FILE,
        metadata_file: str = config.EMBEDDING_METADATA_FILE,
    ) -> bool:
        """Load cached embeddings from local disk or Hugging Face dataset cache."""
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

        try:
            json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Embedding metadata file is invalid JSON: %s", exc)
            return False

        self._doc_embeddings = np.load(emb_path, allow_pickle=False)
        logger.info("Loaded embedding cache from %s", emb_path)
        return True

    # -- Model lifecycle ---------------------------------------------------

    def load_model(self) -> None:
        """Download (if needed) and load the embedding model into MLX."""
        logger.info("Loading embedding model %s …", self.model_name)
        try:
            model_path = _download(self.model_name)
            self.model, _ = load_model(
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
        """Free the model from memory (useful before loading the reranker)."""
        self.model = None
        self.tokenizer = None
        mx.clear_cache()
        logger.info("Embedding model unloaded.")

    # -- Embedding logic ---------------------------------------------------

    def _embed_batch(self, texts: list[str], max_tokens: int) -> np.ndarray:
        """Embed a list of texts and return L2-normalised numpy vectors."""
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

        # L2 normalise
        norms = mx.linalg.norm(pooled, axis=-1, keepdims=True)
        normed = pooled / mx.maximum(norms, mx.array(1e-9))
        mx.eval(normed)

        result = np.array(normed, dtype=np.float32)

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

        # Cosine similarity (vectors are already normalised → dot product)
        scores = self._doc_embeddings @ q_vec.T  # (n_docs, 1)
        scores = scores.squeeze(-1)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    @property
    def has_embeddings(self) -> bool:
        return self._doc_embeddings is not None and self._doc_embeddings.shape[0] > 0
