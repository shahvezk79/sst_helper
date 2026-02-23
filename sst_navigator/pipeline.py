"""
Pipeline orchestrator — ties the data loader, embedder, reranker, and
generator into a single search-and-summarise workflow.

Because the 4B embedding, reranker, and generation models are each ~4 GB,
we load and unload them sequentially to keep peak memory manageable on
machines with ≤32 GB unified memory.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .data_loader import load_sst_decisions
from .embedder import SemanticSearcher
from .reranker import Reranker
from .generator import CaseCardGenerator

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single ranked decision returned to the UI."""
    rank: int
    name: str
    date: str
    url: str
    reranker_score: float
    snippet: str
    full_text: str


@dataclass
class PipelineOutput:
    """Everything the UI needs after a search."""
    case_card: str
    results: list[SearchResult] = field(default_factory=list)


class SSTNavigatorPipeline:
    """End-to-end two-stage RAG pipeline for SST decisions."""

    def __init__(
        self,
        dev_mode: bool = True,
        generation_backend: str = "mlx",
        fast_mode: bool = False,
    ):
        self.dev_mode = dev_mode
        self.generation_backend = generation_backend
        self.fast_mode = fast_mode

        self._df: pd.DataFrame | None = None
        self._searcher = SemanticSearcher()
        self._reranker = Reranker()
        self._generator = CaseCardGenerator(backend=generation_backend)

        # Track which heavy model is currently loaded
        self._loaded: str | None = None

    def set_generation_backend(self, backend: str) -> None:
        """Switch generation backend safely and reset generator state."""
        if backend == self.generation_backend:
            return

        # If generator is currently active, unload any local model first.
        if self._loaded == "generator":
            self._generator.unload_model()

        self.generation_backend = backend
        self._generator = CaseCardGenerator(backend=backend)

        # Force a fresh component load on next search.
        if self._loaded == "generator":
            self._loaded = None


    def _active_params(self) -> dict:
        """Return runtime parameters based on current quality/speed mode."""
        if self.fast_mode:
            return {
                "embedding_max_tokens": config.FAST_EMBEDDING_MAX_TOKENS,
                "stage1_top_k": config.FAST_STAGE1_TOP_K,
                "stage2_top_k": config.FAST_STAGE2_TOP_K,
                "reranker_max_tokens": config.FAST_RERANKER_MAX_TOKENS,
                "generation_max_tokens": config.FAST_GENERATION_MAX_TOKENS,
                "generation_max_chars": config.FAST_GENERATION_MAX_CHARS,
            }
        return {
            "embedding_max_tokens": config.EMBEDDING_MAX_TOKENS,
            "stage1_top_k": config.STAGE1_TOP_K,
            "stage2_top_k": config.STAGE2_TOP_K,
            "reranker_max_tokens": config.RERANKER_MAX_TOKENS,
            "generation_max_tokens": config.GENERATION_MAX_TOKENS,
            "generation_max_chars": config.GENERATION_MAX_CHARS,
        }

    # -- Data loading & indexing -------------------------------------------

    def load_data(self, progress_callback=None) -> int:
        """Load the SST dataset and return the row count."""
        max_rows = config.DEV_ROW_LIMIT if self.dev_mode else None
        self._df = load_sst_decisions(max_rows=max_rows)
        return len(self._df)

    def build_index(self, progress_callback=None) -> None:
        """Embed all documents (Stage 1 prep)."""
        if self._df is None:
            raise RuntimeError("Call load_data() first.")

        texts = self._df["unofficial_text_en"].tolist()
        if not texts:
            raise RuntimeError("No decision texts available to index after preprocessing.")

        # Keep the batch size configuration we saved earlier
        batch_size = (
            config.EMBEDDING_BATCH_SIZE_DEV if self.dev_mode else config.EMBEDDING_BATCH_SIZE_PROD
        )
        params = self._active_params()

        # Use the clean, dynamic cache loading from the main branch
        if self._searcher.load_embeddings_cache(
            texts=texts,
            cache_dir=config.EMBEDDING_CACHE_DIR,
            max_tokens=params["embedding_max_tokens"],
        ):
            logger.info("Using cached embeddings; skipping rebuild.")
            return

        self._load_component("embedder")
        self._searcher.embed_documents(
            texts,
            batch_size=batch_size,
            max_tokens=params["embedding_max_tokens"],
            progress_callback=progress_callback,
        )
        
        # Use the clean cache saving method from the main branch
        self._searcher.save_embeddings_cache(
            texts=texts,
            cache_dir=config.EMBEDDING_CACHE_DIR,
            max_tokens=params["embedding_max_tokens"],
        )
        
    @property
    def is_ready(self) -> bool:
        return self._df is not None and self._searcher.has_embeddings

    # -- Memory management -------------------------------------------------

    def _load_component(self, name: str) -> None:
        """Ensure only one heavy model is in memory at a time."""
        if self._loaded == name:
            return

        # Unload current model
        if self._loaded == "embedder":
            self._searcher.unload_model()
        elif self._loaded == "reranker":
            self._reranker.unload_model()
        elif self._loaded == "generator":
            self._generator.unload_model()

        # Load requested model
        if name == "embedder":
            self._searcher.load_model()
        elif name == "reranker":
            self._reranker.load_model()
        elif name == "generator":
            self._generator.load_model()

        self._loaded = name
        logger.info("Active component: %s", name)

    # -- Search ------------------------------------------------------------

    def search(self, query: str) -> PipelineOutput:
        """Run the full two-stage search + generation pipeline.

        1. Semantic search (bi-encoder) → top 20 candidates
        2. Rerank (cross-encoder)       → top 3
        3. Generate case card for #1

        Returns a PipelineOutput with the case card and result list.
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline not initialised. Call load_data() + build_index().")

        params = self._active_params()

        # ---- Stage 1: Semantic search ------------------------------------
        # Embedder must already be loaded (index is cached), but we need
        # the model in memory to embed the query.
        self._load_component("embedder")
        hits = self._searcher.search(
            query,
            top_k=params["stage1_top_k"],
            max_tokens=params["embedding_max_tokens"],
        )
        logger.info("Stage 1 returned %d candidates.", len(hits))
        if not hits:
            return PipelineOutput(
                case_card="No similar cases were found for this query.",
                results=[],
            )

        # Build candidate dicts for the reranker
        candidates = []
        for idx, score in hits:
            row = self._df.iloc[idx]
            candidates.append({
                "index": idx,
                "name": row.get("name_en", "Unnamed"),
                "date": str(row.get("document_date_en", "")),
                "url": row.get("url_en", ""),
                "text": row["unofficial_text_en"],
                "stage1_score": score,
            })

        # ---- Stage 2: Reranking -----------------------------------------
        self._load_component("reranker")
        top_results = self._reranker.rerank(
            query,
            candidates,
            top_k=params["stage2_top_k"],
            max_tokens=params["reranker_max_tokens"],
        )
        logger.info("Stage 2 returned %d results.", len(top_results))
        if not top_results:
            return PipelineOutput(
                case_card="No sufficiently relevant cases were found after reranking.",
                results=[],
            )

        # ---- Stage 3: Case card generation -------------------------------
        self._load_component("generator")
        case_card = self._generator.generate_case_card(
            top_results[0]["text"],
            max_tokens=params["generation_max_tokens"],
            max_chars=params["generation_max_chars"],
        )

        # ---- Assemble output ---------------------------------------------
        results = []
        for rank, r in enumerate(top_results, 1):
            results.append(SearchResult(
                rank=rank,
                name=r["name"],
                date=r["date"],
                url=r["url"],
                reranker_score=r["reranker_score"],
                snippet=r["text"][:config.SNIPPET_LENGTH],
                full_text=r["text"],
            ))

        return PipelineOutput(case_card=case_card, results=results)
