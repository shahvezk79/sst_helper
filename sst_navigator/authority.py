"""
Authority-Aware Ranking Policy
==============================

Implements a two-layer scoring framework on top of the existing
cross-encoder relevance scores:

    FinalScore = RelevanceScore × RecencyFactor × GroundingFactor

Layer 1 (Relevance)  — handled upstream by bi-encoder + cross-encoder.
Layer 2 (Authority)  — computed here from metadata and text signals:
  - Recency factor      : exponential decay favouring recent decisions
  - Grounding factor    : boost for decisions with rich reasoning sections
  - Authority risk flag : user-facing safety signal derived from age + score
  - Outcome detection   : inferred "Allowed" / "Dismissed" label

Why this matters for self-represented litigants
------------------------------------------------
Semantic similarity alone can surface factually-similar but legally-outdated
decisions.  In Canadian administrative law, legislative amendments and
evolving policy interpretations mean a case from 10+ years ago may no longer
reflect the current standard.  The authority-adjusted score nudges the ranking
to surface safer, more reliable precedents without hiding older ones entirely.

What the system *cannot* guarantee
------------------------------------
- Actual precedential status (binding vs. persuasive)
- Whether the decision was appealed or overturned
- Compliance with any post-decision regulatory changes

Users should always verify with current sources and consult a legal professional.

This module is deliberately free of I/O and heavy dependencies so it can
be tested without loading any ML models.
"""

import math
import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_RECENCY_HALF_LIFE_YEARS = 7.0  # Score halves every 7 years (ln2/7 ≈ 0.099/yr)

# Age thresholds (years from today)
_RECENT_MAX_YEARS = 5           # ≤ 5 years  → "Recent"
_OLDER_MAX_YEARS  = 12          # 5–12 years → "Older"; > 12 → "Historical"

# Relevance thresholds (cross-encoder score ∈ [0, 1])
_HIGH_REL   = 0.65
_MEDIUM_REL = 0.35


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _today_year() -> int:
    """Return the current calendar year (called lazily — no module-level side effects)."""
    from datetime import date
    return date.today().year


def _parse_year(date_str: str) -> int | None:
    """Extract a 4-digit year from any common date-string format."""
    if not date_str:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", str(date_str))
    return int(m.group(0)) if m else None


# ---------------------------------------------------------------------------
# Individual scoring factors
# ---------------------------------------------------------------------------

def compute_recency_factor(date_str: str) -> float:
    """
    Exponential-decay recency weight in (0, 1].

    A decision from this year scores 1.0; one from 7 years ago scores ~0.5;
    one from 14 years ago scores ~0.25.  Unknown dates get a neutral 0.5.

    This uses a soft decay rather than a hard cutoff so older-but-highly-
    relevant decisions are demoted, not hidden.
    """
    year = _parse_year(date_str)
    if year is None:
        return 0.5
    age = max(0, _today_year() - year)
    return math.exp(-math.log(2) * age / _RECENCY_HALF_LIFE_YEARS)


def compute_grounding_factor(text: str) -> float:
    """
    Estimate decision quality / reasoning depth from text signals.

    Returns a multiplier in [0.75, 1.0]:
      • 0.75 baseline — never penalises a decision just for being short
      • +0.05 for a structured Analysis section heading
      • +0.00–0.10 for legal citation density (acts, sections, regulations)
      • +0.00–0.10 for reasoning-quality language markers

    The ceiling at 1.0 means the grounding factor can only help, never push
    the authority score above the raw relevance score.
    """
    if not text:
        return 0.75

    tl = text.lower()
    score = 0.75

    # Structured Analysis section is the strongest grounding signal in SST decisions
    if re.search(r"\banalysis\b", tl):
        score += 0.05

    # Legal citation density: statutory references and key actors
    citations = len(re.findall(
        r"\b(section \d+|s\.\s*\d+|the act|regulation|canada pension plan"
        r"|employment insurance act|minister|tribunal)\b",
        tl,
    ))
    score += min(0.10, citations * 0.01)

    # Reasoning-quality markers that indicate explicit fact-finding
    reasoning_hits = sum(
        1
        for pat in (
            r"\bi find\b",
            r"\btherefore\b",
            r"\bthe evidence\b",
            r"\bi am (satisfied|not satisfied)\b",
            r"\bthe (law|test|criteria)\b",
            r"\bi (conclude|am persuaded)\b",
        )
        if re.search(pat, tl)
    )
    score += min(0.10, reasoning_hits * 0.02)

    return min(1.0, score)


def compute_outcome(text: str) -> str:
    """
    Infer the decision outcome: 'Allowed', 'Dismissed', or 'Unknown'.

    Scans the first 800 characters — the Decision/Conclusion sections appear
    near the top of SST decisions — and applies conservative keyword matching.
    Used to detect outcome diversity across the top-K results.
    """
    if not text:
        return "Unknown"
    head = text[:800].lower()
    allowed   = bool(re.search(r"\bappeal is allowed\b", head))
    dismissed = bool(re.search(r"\bappeal is dismissed\b", head))
    if allowed and not dismissed:
        return "Allowed"
    if dismissed and not allowed:
        return "Dismissed"
    return "Unknown"


# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------

def recency_label(date_str: str) -> str:
    """Return 'Recent', 'Older', 'Historical', or 'Unknown'."""
    year = _parse_year(date_str)
    if year is None:
        return "Unknown"
    age = _today_year() - year
    if age <= _RECENT_MAX_YEARS:
        return "Recent"
    if age <= _OLDER_MAX_YEARS:
        return "Older"
    return "Historical"


def relevance_label(reranker_score: float) -> str:
    """Return 'High', 'Medium', or 'Low' based on cross-encoder score."""
    if reranker_score >= _HIGH_REL:
        return "High"
    if reranker_score >= _MEDIUM_REL:
        return "Medium"
    return "Low"


def authority_risk_label(date_str: str, reranker_score: float) -> str:
    """
    Three-level authority-risk signal for user-facing display:

      Low    — recent + high relevance; relatively safer to rely on
      Medium — moderately old or medium relevance; use with care
      Verify — historical or low relevance; verify with current sources

    "Verify" does not mean the case is wrong — it means the user should
    independently confirm it reflects current law before relying on it.
    """
    year = _parse_year(date_str)
    age  = (_today_year() - year) if year is not None else 999
    if age > _OLDER_MAX_YEARS or reranker_score < _MEDIUM_REL:
        return "Verify"
    if age > _RECENT_MAX_YEARS or reranker_score < _HIGH_REL:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Combined assessment
# ---------------------------------------------------------------------------

class AuthorityProfile(NamedTuple):
    """Full authority assessment for a single search result."""
    authority_score:  float  # FinalScore = relevance × recency × grounding
    recency_factor:   float  # Raw recency weight ∈ (0, 1]
    grounding_factor: float  # Raw grounding weight ∈ [0.75, 1.0]
    relevance_label:  str    # "High" / "Medium" / "Low"
    recency_label:    str    # "Recent" / "Older" / "Historical" / "Unknown"
    authority_risk:   str    # "Low" / "Medium" / "Verify"
    outcome:          str    # "Allowed" / "Dismissed" / "Unknown"


def assess(reranker_score: float, date_str: str, text: str) -> AuthorityProfile:
    """
    Compute the full authority profile for one search result.

    Parameters
    ----------
    reranker_score : float
        Cross-encoder relevance probability ∈ [0, 1].
    date_str : str
        Decision date string in any common format (e.g. "2021-03-15").
    text : str
        Full decision text used for grounding-factor estimation.

    Returns
    -------
    AuthorityProfile
        Named tuple with authority_score and all human-readable labels.
    """
    rf    = compute_recency_factor(date_str)
    gf    = compute_grounding_factor(text)
    final = reranker_score * rf * gf

    return AuthorityProfile(
        authority_score=final,
        recency_factor=rf,
        grounding_factor=gf,
        relevance_label=relevance_label(reranker_score),
        recency_label=recency_label(date_str),
        authority_risk=authority_risk_label(date_str, reranker_score),
        outcome=compute_outcome(text),
    )


def check_outcome_diversity(outcomes: list[str]) -> bool:
    """
    Return True if the result set has meaningful outcome diversity.

    A result set where every resolved case went the same way may give
    a user a misleadingly one-sided picture.  The caller can use this
    to surface a diversity notice in the UI.

    'Unknown' outcomes are ignored when assessing uniformity.
    """
    resolved = [o for o in outcomes if o != "Unknown"]
    if len(resolved) < 2:
        return True   # not enough information to judge
    return len(set(resolved)) > 1
