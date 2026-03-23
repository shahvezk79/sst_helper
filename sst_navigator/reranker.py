"""
Stage 2 — Cross-encoder reranking via Qwen3-Reranker.

Supports two backends:
  1. Local MLX model (default) — runs on Apple Silicon via mlx-lm.
  2. DeepInfra API — set DEEPINFRA_API_KEY in env.
"""

import logging
import math

from . import config
from .auth import deepinfra_auth_error_hint, get_deepinfra_api_key
from .section_parser import pack_for_reranker

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only "
    'be "yes" or "no".'
)


class Reranker:
    """Cross-encoder reranker with MLX or DeepInfra backend."""

    def __init__(
        self,
        model_name: str = config.RERANKER_MODEL,
        backend: str = "mlx",
    ):
        self.model_name = model_name
        self.backend = backend
        self.model = None
        self.tokenizer = None
        self._token_yes: int | None = None
        self._token_no: int | None = None

    # -- Model lifecycle ---------------------------------------------------

    def load_model(self) -> None:
        if self.backend != "mlx":
            return

        import mlx.core as mx          # noqa: F811 — lazy import
        from mlx_lm import load as mlx_lm_load

        logger.info("Loading reranker model %s …", self.model_name)
        try:
            self.model, self.tokenizer = mlx_lm_load(self.model_name)
            self._token_yes = self.tokenizer.convert_tokens_to_ids("yes")
            self._token_no = self.tokenizer.convert_tokens_to_ids("no")
            logger.info("Reranker model loaded (yes=%d, no=%d).",
                        self._token_yes, self._token_no)
        except Exception as e:
            logger.error("Failed to load reranker model: %s", e)
            raise

    def unload_model(self) -> None:
        if self.backend != "mlx":
            return

        import mlx.core as mx

        self.model = None
        self.tokenizer = None
        mx.metal.clear_cache()
        logger.info("Reranker model unloaded.")

    # -- MLX scoring logic -------------------------------------------------

    def _build_prompt(self, query: str, document: str) -> str:
        """Build the chat-template prompt expected by Qwen3-Reranker."""
        user_content = (
            f"<Instruct>: {config.RERANKER_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Append the think block that the model expects
        prompt += "<think>\n\n</think>\n\n"
        return prompt

    def _score_one(self, query: str, document: str) -> float:
        """Return a relevance probability in [0, 1] for a single pair."""
        import mlx.core as mx

        prompt = self._build_prompt(query, document)

        input_ids = mx.array(
            [self.tokenizer.encode(prompt, add_special_tokens=False)]
        )

        # Forward pass → logits over vocab at every position
        logits = self.model(input_ids)  # (1, seq_len, vocab_size)
        last_logits = logits[:, -1, :]  # (1, vocab_size)

        yes_logit = last_logits[:, self._token_yes]
        no_logit = last_logits[:, self._token_no]

        # Stack [no, yes] and apply temperature-scaled softmax.
        # Temperature > 1 spreads the distribution so scores are not
        # compressed into the 99–100% range.
        stacked = mx.concatenate([no_logit, yes_logit], axis=0)  # (2,)
        temperature = config.RERANKER_TEMPERATURE
        probs = mx.softmax(stacked / temperature)
        score = probs[1].item()
        return float(score)

    # -- DeepInfra scoring logic -------------------------------------------

    def _score_batch_deepinfra(
        self, query: str, documents: list[str],
    ) -> list[float]:
        """Score all documents in one API call via DeepInfra."""
        import requests

        api_key = get_deepinfra_api_key()

        # Prepend the task instruction to the query so the model can
        # leverage it for relevance judgement (same info as the local
        # <Instruct> field).
        formatted_query = f"{config.RERANKER_INSTRUCTION}\n{query}"

        resp = requests.post(
            config.DEEPINFRA_RERANKER_ENDPOINT,
            headers={
                "Authorization": f"bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "queries": [formatted_query],
                "documents": documents,
            },
            timeout=120,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            if resp.status_code in {401, 403}:
                raise RuntimeError(deepinfra_auth_error_hint()) from exc
            raise
        data = resp.json()

        scores = data["scores"]
        # The API may return [[s1, s2, …]] (one list per query) or [s1, s2, …]
        if scores and isinstance(scores[0], list):
            scores = scores[0]
        return [float(s) for s in scores]

    # -- Score calibration -------------------------------------------------

    @staticmethod
    def _calibrate_score(raw_score: float) -> float:
        """Apply temperature scaling to a raw reranker probability.

        Converts P(yes) back to log-odds, divides by the configured
        temperature, and converts back to a probability.  This spreads
        scores that cluster near 1.0 into a more interpretable range.
        Returns the raw score unchanged when temperature is 1.0.
        """
        temperature = config.RERANKER_TEMPERATURE
        if temperature == 1.0:
            return raw_score
        # Clamp to avoid log(0) / division-by-zero
        p = max(min(raw_score, 1.0 - 1e-9), 1e-9)
        log_odds = math.log(p / (1.0 - p))
        return 1.0 / (1.0 + math.exp(-log_odds / temperature))

    # -- Score diagnostics -------------------------------------------------

    @staticmethod
    def _log_score_distribution(candidates: list[dict]) -> None:
        """Log summary statistics for reranker scores to aid calibration."""
        scores = [c["reranker_score"] for c in candidates if "reranker_score" in c]
        if not scores:
            return
        scores.sort(reverse=True)
        n = len(scores)
        mean = sum(scores) / n
        top = scores[0]
        bottom = scores[-1]
        median = scores[n // 2]
        spread = top - bottom
        logger.info(
            "Reranker score distribution (n=%d): "
            "max=%.4f  median=%.4f  min=%.4f  mean=%.4f  spread=%.4f",
            n, top, median, bottom, mean, spread,
        )
        # Flag compressed distributions where ranking signal may be weak
        if spread < 0.05 and n > 1:
            logger.warning(
                "Score spread is very narrow (%.4f) — ranking signal may be weak. "
                "Consider inspecting query specificity or candidate diversity.",
                spread,
            )

    # -- Public API --------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = config.STAGE2_TOP_K,
        max_tokens: int = config.RERANKER_MAX_TOKENS,
    ) -> list[dict]:
        """Score and re-sort candidates by cross-encoder relevance.

        Args:
            query: The user's plain-language query.
            candidates: List of dicts, each must contain at least
                        'text' and 'index' keys.
            top_k: How many top results to return.

        Returns:
            The top_k candidates sorted by descending reranker score,
            each augmented with a 'reranker_score' key.
        """
        logger.info("Reranking %d candidates (backend=%s) …",
                     len(candidates), self.backend)

        if self.backend == "deepinfra":
            return self._rerank_deepinfra(query, candidates, top_k, max_tokens)

        # --- MLX path (original) ------------------------------------------
        for cand in candidates:
            # Pack legally salient sections into the context window
            doc_text = pack_for_reranker(cand["text"], char_budget=max_tokens * 4)
            cand["reranker_score"] = self._score_one(query, doc_text)
            logger.debug(
                "  candidate %s → %.4f", cand.get("name", "?"), cand["reranker_score"]
            )

        ranked = sorted(candidates, key=lambda c: c["reranker_score"], reverse=True)
        self._log_score_distribution(candidates)
        return ranked[:top_k]

    def _rerank_deepinfra(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
        max_tokens: int,
    ) -> list[dict]:
        """Rerank candidates via DeepInfra API in a single batch call."""
        documents = []
        for cand in candidates:
            doc_text = pack_for_reranker(cand["text"], char_budget=max_tokens * 4)
            documents.append(doc_text)

        scores = self._score_batch_deepinfra(query, documents)

        for cand, score in zip(candidates, scores):
            cand["reranker_score"] = self._calibrate_score(score)
            logger.debug(
                "  candidate %s → %.4f (raw %.4f)",
                cand.get("name", "?"), cand["reranker_score"], score,
            )

        ranked = sorted(candidates, key=lambda c: c["reranker_score"], reverse=True)
        self._log_score_distribution(candidates)
        return ranked[:top_k]
