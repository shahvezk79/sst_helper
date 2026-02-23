"""
Case-card generator — produces a plain-language four-section summary
of a top-ranked SST decision.

Supports three backends:
  1. Local MLX model (default) — runs entirely on Apple Silicon.
  2. OpenAI API — set OPENAI_API_KEY in env.
  3. Google Gemini API — set GOOGLE_API_KEY in env.
"""

import logging
import os

import mlx.core as mx
from mlx_lm import load as mlx_lm_load, generate as mlx_generate

from . import config

logger = logging.getLogger(__name__)

_CASE_CARD_SYSTEM = (
    "You are a legal-aid assistant. Read the tribunal decision below and "
    "produce a plain-language summary a self-represented person can understand. "
    "Output EXACTLY four sections with these headings:\n\n"
    "**Issue:** (What legal question was the tribunal deciding?)\n"
    "**Key Facts:** (What were the most important facts?)\n"
    "**Test Applied:** (What legal test or criteria did the tribunal use?)\n"
    "**Outcome:** (What did the tribunal decide, and why?)\n\n"
    "Be concise. Use simple language. Do not invent facts."
)


def _build_prompt(
    decision_text: str,
    max_chars: int = config.GENERATION_MAX_CHARS,
) -> str:
    """Wrap the decision text into a chat prompt."""
    # Truncate to fit context — leave room for system prompt + output
    truncated = decision_text[:max_chars]
    return (
        f"<|im_start|>system\n{_CASE_CARD_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Here is the tribunal decision:\n\n{truncated}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class CaseCardGenerator:
    """Generate a structured summary of an SST decision."""

    def __init__(self, backend: str = "mlx"):
        """
        Args:
            backend: One of "mlx", "openai", or "gemini".
        """
        self.backend = backend
        self._model = None
        self._tokenizer = None

    # -- MLX local model ---------------------------------------------------

    def load_model(self) -> None:
        """Load the local MLX generation model."""
        if self.backend != "mlx":
            return
        logger.info("Loading generation model %s …", config.GENERATION_MODEL)
        try:
            self._model, self._tokenizer = mlx_lm_load(config.GENERATION_MODEL)
            logger.info("Generation model loaded.")
        except Exception as e:
            logger.error("Failed to load generation model: %s", e)
            raise

    def unload_model(self) -> None:
        self._model = None
        self._tokenizer = None
        mx.metal.clear_cache()
        logger.info("Generation model unloaded.")

    # -- Backend dispatch --------------------------------------------------

    def generate_case_card(
        self,
        decision_text: str,
        max_tokens: int = config.GENERATION_MAX_TOKENS,
        max_chars: int = config.GENERATION_MAX_CHARS,
    ) -> str:
        """Return a formatted case-card string for the given decision."""
        if self.backend == "mlx":
            return self._generate_mlx(decision_text, max_tokens=max_tokens, max_chars=max_chars)
        if self.backend == "openai":
            return self._generate_openai(decision_text, max_tokens=max_tokens, max_chars=max_chars)
        if self.backend == "gemini":
            return self._generate_gemini(decision_text, max_tokens=max_tokens, max_chars=max_chars)
        raise ValueError(f"Unknown backend: {self.backend}")

    # -- MLX ---------------------------------------------------------------

    def _generate_mlx(
        self,
        decision_text: str,
        max_tokens: int = config.GENERATION_MAX_TOKENS,
        max_chars: int = config.GENERATION_MAX_CHARS,
    ) -> str:
        if self._model is None:
            raise RuntimeError("Call load_model() first.")
        prompt = _build_prompt(decision_text, max_chars=max_chars)
        response = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response.strip()

    # -- OpenAI ------------------------------------------------------------

    def _generate_openai(
        self,
        decision_text: str,
        max_tokens: int = config.GENERATION_MAX_TOKENS,
        max_chars: int = config.GENERATION_MAX_CHARS,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai  to use the OpenAI backend.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")

        client = OpenAI(api_key=api_key)
        truncated = decision_text[:max_chars]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _CASE_CARD_SYSTEM},
                {"role": "user", "content": f"Here is the tribunal decision:\n\n{truncated}"},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    # -- Gemini ------------------------------------------------------------

    def _generate_gemini(
        self,
        decision_text: str,
        max_tokens: int = config.GENERATION_MAX_TOKENS,
        max_chars: int = config.GENERATION_MAX_CHARS,
    ) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "pip install google-generativeai  to use the Gemini backend."
            )

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY in your environment.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        truncated = decision_text[:max_chars]

        resp = model.generate_content(
            f"{_CASE_CARD_SYSTEM}\n\nHere is the tribunal decision:\n\n{truncated}",
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
            ),
        )
        return resp.text.strip()
