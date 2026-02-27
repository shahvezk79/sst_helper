"""
Stage 2 — Cross-encoder reranking via Qwen3-Reranker.

Loads the reranker model through mlx-lm and scores each (query, document)
pair by extracting the log-probability of the "yes" token vs "no" token.
"""

import logging

import mlx.core as mx
from mlx_lm import load as mlx_lm_load
from transformers import AutoTokenizer

from . import config
from .section_parser import pack_for_reranker

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only "
    'be "yes" or "no".'
)


class Reranker:
    """Cross-encoder reranker using Qwen3-Reranker on MLX."""

    def __init__(self, model_name: str = config.RERANKER_MODEL):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._token_yes: int | None = None
        self._token_no: int | None = None

    # -- Model lifecycle ---------------------------------------------------

    def load_model(self) -> None:
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
        self.model = None
        self.tokenizer = None
        mx.metal.clear_cache()
        logger.info("Reranker model unloaded.")

    # -- Scoring logic -----------------------------------------------------

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
        prompt = self._build_prompt(query, document)

        input_ids = mx.array(
            [self.tokenizer.encode(prompt, add_special_tokens=False)]
        )

        # Forward pass → logits over vocab at every position
        logits = self.model(input_ids)  # (1, seq_len, vocab_size)
        last_logits = logits[:, -1, :]  # (1, vocab_size)

        yes_logit = last_logits[:, self._token_yes]
        no_logit = last_logits[:, self._token_no]

        # Stack [no, yes] and softmax → P(yes) is the relevance score
        stacked = mx.concatenate([no_logit, yes_logit], axis=0)  # (2,)
        probs = mx.softmax(stacked)
        score = probs[1].item()
        return float(score)

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
        logger.info("Reranking %d candidates …", len(candidates))
        for cand in candidates:
            # Pack legally salient sections into the context window
            doc_text = pack_for_reranker(cand["text"], char_budget=max_tokens * 4)
            cand["reranker_score"] = self._score_one(query, doc_text)
            logger.debug(
                "  candidate %s → %.4f", cand.get("name", "?"), cand["reranker_score"]
            )

        ranked = sorted(candidates, key=lambda c: c["reranker_score"], reverse=True)
        return ranked[:top_k]
