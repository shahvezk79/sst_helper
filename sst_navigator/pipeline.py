"""
Pipeline orchestrator — ties the data loader, embedder, reranker, and
generator into a single search-and-summarise workflow.

Because the 8B embedding, reranker, and generation models are each ~8 GB,
we load and unload them sequentially to keep peak memory manageable on
machines with ≤32 GB unified memory.  Cloud mode (DeepInfra) bypasses
local model loading for embedding and reranking entirely.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .authority import assess as _authority_assess, check_outcome_diversity
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
    # --- Layer 2: authority-aware fields ---
    authority_score: float = 0.0   # reranker_score × recency × grounding
    relevance_label: str = "Medium"  # "High" / "Medium" / "Low"
    recency_label: str = "Unknown"   # "Recent" / "Older" / "Historical"
    authority_risk: str = "Medium"   # "Low" / "Medium" / "Verify"
    outcome: str = "Unknown"         # "Allowed" / "Dismissed" / "Unknown"
    # --- Layer 3: fast-mode landscape role ---
    landscape_role: str = ""         # "supportive" / "unsupportive" / "context" / ""


@dataclass
class PipelineOutput:
    """Everything the UI needs after a search."""
    case_card: str | None = None
    results: list[SearchResult] = field(default_factory=list)
    outcome_diversity: bool = True  # False → all top results share the same outcome
    landscape_mode: bool = False    # True when fast mode returns labeled landscape pairs


class SSTNavigatorPipeline:
    """End-to-end two-stage RAG pipeline for SST decisions."""

    def __init__(
        self,
        dev_mode: bool = True,
        generation_backend: str = "mlx",
        compute_mode: str = "local",
        fast_mode: bool = False,
    ):
        self.dev_mode = dev_mode
        self.compute_mode = compute_mode
        self.generation_backend = generation_backend
        self.fast_mode = fast_mode

        embedder_backend = "deepinfra" if compute_mode == "cloud" else "mlx"
        reranker_backend = "deepinfra" if compute_mode == "cloud" else "mlx"

        self._df: pd.DataFrame | None = None
        self._searcher = SemanticSearcher(backend=embedder_backend)
        self._reranker = Reranker(backend=reranker_backend)
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

    def set_compute_mode(self, mode: str) -> None:
        """Switch between local (MLX) and cloud (DeepInfra) compute.

        When cloud is selected, both reranker and generator use
        DeepInfra APIs — no local models need to be loaded for those
        stages.
        """
        if mode == self.compute_mode:
            return

        # Tear down any locally-loaded component before switching backend.
        if self._loaded == "embedder":
            self._searcher.unload_model()
            self._loaded = None
        elif self._loaded == "reranker":
            self._reranker.unload_model()
            self._loaded = None
        elif self._loaded == "generator":
            self._generator.unload_model()
            self._loaded = None

        self.compute_mode = mode
        embedder_backend = "deepinfra" if mode == "cloud" else "mlx"
        reranker_backend = "deepinfra" if mode == "cloud" else "mlx"

        # Carry over already-loaded document embeddings to the new searcher
        # so we don't need to re-download/re-align the cache.
        old_doc_embeddings = self._searcher._doc_embeddings
        self._searcher = SemanticSearcher(backend=embedder_backend)
        self._searcher._doc_embeddings = old_doc_embeddings

        self._reranker = Reranker(backend=reranker_backend)


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
        """Download precomputed embeddings and align with the loaded dataframe.

        The cached embedding vectors may be in a different order than the
        dataframe rows (``scripts/update_index.py`` sorts by text length for
        efficient batching).  This method downloads the cache from
        HuggingFace, then reorders the vectors so that embedding row *i*
        corresponds to ``self._df.iloc[i]``.  In dev mode only the subset
        of vectors matching the loaded rows is kept.
        """
        if self._df is None:
            raise RuntimeError("Call load_data() first.")

        if not self._searcher.load_embeddings_cache(cache_dir=config.EMBEDDING_CACHE_DIR):
            raise RuntimeError(
                "Could not load precomputed embedding cache. "
                "Check your internet connection or run scripts/update_index.py "
                "to refresh the local cache before starting the app."
            )

        # Align cached embeddings with the dataframe row ordering.
        target_urls = self._df["url_en"].astype(str).tolist()
        original_count = len(target_urls)
        matched_indices = self._searcher.align_to_urls(target_urls)

        if len(matched_indices) < original_count:
            self._df = self._df.iloc[matched_indices].reset_index(drop=True)
            logger.warning(
                "Only %d of %d decisions had cached embeddings; "
                "dataframe trimmed to match.",
                len(matched_indices),
                original_count,
            )

        logger.info(
            "Index ready — %d decisions with aligned embeddings.",
            len(self._df),
        )
        
    @property
    def is_ready(self) -> bool:
        return self._df is not None and self._searcher.has_embeddings

    # -- Memory management -------------------------------------------------

    def _load_component(self, name: str) -> None:
        """Ensure only one heavy model is in memory at a time.

        Cloud-backed components (DeepInfra) don't load local models, so
        we skip the memory swap to keep the previous local model
        resident (typically the embedder for query encoding).
        """
        if self._loaded == name:
            return

        # Cloud-backed stages have no local model — skip the swap.
        if name == "embedder" and self._searcher.backend != "mlx":
            return
        if name == "reranker" and self._reranker.backend != "mlx":
            return
        if name == "generator" and self._generator.backend != "mlx":
            return

        # Unload current local model
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

    def generate_case_card(self, decision_text: str, grounded: bool = False) -> str:
        """Generate a case card for a selected decision text on demand.

        When *grounded* is True, claims in the summary include paragraph
        references and explicit/inferred confidence markers.
        """
        params = self._active_params()
        self._load_component("generator")
        return self._generator.generate_case_card(
            decision_text,
            max_tokens=params["generation_max_tokens"],
            max_chars=params["generation_max_chars"],
            grounded=grounded,
        )

    def search(self, query: str, include_case_card: bool = True) -> PipelineOutput:
        """Run the full two-stage search + generation pipeline.

        1. Semantic search (bi-encoder) → top 40 candidates
        2. Rerank (cross-encoder)       → top 5
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

        # ---- Stage 2.5: Authority scoring --------------------------------
        # Compute authority profiles and re-rank so that the final order
        # reflects both semantic relevance and how safe each decision is
        # to rely on (recency × reasoning depth).
        for r in top_results:
            profile = _authority_assess(r["reranker_score"], r["date"], r["text"])
            r["authority_score"]  = profile.authority_score
            r["relevance_label"]  = profile.relevance_label
            r["recency_label"]    = profile.recency_label
            r["authority_risk"]   = profile.authority_risk
            r["outcome"]          = profile.outcome

        top_results.sort(key=lambda x: x["authority_score"], reverse=True)
        logger.info(
            "Authority re-rank complete. Top result: %s (auth=%.3f)",
            top_results[0]["name"],
            top_results[0]["authority_score"],
        )

        # Outcome diversity check — flag if every resolved case went the same way
        diversity_ok = check_outcome_diversity([r["outcome"] for r in top_results])

        # ---- Stage 3: Optional case-card generation ----------------------
        case_card: str | None = None
        if include_case_card:
            case_card = self.generate_case_card(top_results[0]["text"])

        # ---- Fast-mode landscape pairing ------------------------------------
        # In fast mode, instead of returning the top 3 by score, we label
        # results by their role in a balanced "legal landscape":
        #   - supportive:   could support the user's position (Allowed)
        #   - unsupportive: could weaken the user's position (Dismissed)
        #   - context:      procedural/nuance case (Unknown or extra)
        landscape_mode = False
        if self.fast_mode and len(top_results) >= 2:
            landscape_mode = True
            labeled: list[tuple[dict, str]] = []

            supportive = [r for r in top_results if r["outcome"] == "Allowed"]
            unsupportive = [r for r in top_results if r["outcome"] == "Dismissed"]
            context = [r for r in top_results if r["outcome"] == "Unknown"]

            if supportive:
                labeled.append((supportive[0], "supportive"))
            if unsupportive:
                labeled.append((unsupportive[0], "unsupportive"))

            # Fill remaining slots (up to 3 total) with context cases
            used = {id(t[0]) for t in labeled}
            for r in context:
                if len(labeled) < 3 and id(r) not in used:
                    labeled.append((r, "context"))

            # If we still have fewer than 3 and there are unused results,
            # add the next best regardless of outcome
            for r in top_results:
                if len(labeled) >= 3:
                    break
                if id(r) not in used:
                    role = ("supportive" if r["outcome"] == "Allowed"
                            else "unsupportive" if r["outcome"] == "Dismissed"
                            else "context")
                    labeled.append((r, role))
                    used.add(id(r))

            top_results_labeled = labeled
        else:
            top_results_labeled = [(r, "") for r in top_results]

        # ---- Assemble output ---------------------------------------------
        results = []
        for rank, (r, role) in enumerate(top_results_labeled, 1):
            results.append(SearchResult(
                rank=rank,
                name=r["name"],
                date=r["date"],
                url=r["url"],
                reranker_score=r["reranker_score"],
                snippet=r["text"][:config.SNIPPET_LENGTH],
                full_text=r["text"],
                authority_score=r["authority_score"],
                relevance_label=r["relevance_label"],
                recency_label=r["recency_label"],
                authority_risk=r["authority_risk"],
                outcome=r["outcome"],
                landscape_role=role,
            ))

        return PipelineOutput(
            case_card=case_card,
            results=results,
            outcome_diversity=diversity_ok,
            landscape_mode=landscape_mode,
        )
