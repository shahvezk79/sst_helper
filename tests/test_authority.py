"""Tests for sst_navigator.authority."""

import math
import pytest

from sst_navigator.authority import (
    AuthorityProfile,
    assess,
    authority_risk_label,
    check_outcome_diversity,
    compute_grounding_factor,
    compute_outcome,
    compute_recency_factor,
    recency_label,
    relevance_label,
    _parse_year,
    _today_year,
)


# ---------------------------------------------------------------------------
# _parse_year
# ---------------------------------------------------------------------------

class TestParseYear:
    def test_iso_date(self):
        assert _parse_year("2021-03-15") == 2021

    def test_year_only(self):
        assert _parse_year("2019") == 2019

    def test_partial_text(self):
        assert _parse_year("Decided on 2017-09-22") == 2017

    def test_empty_returns_none(self):
        assert _parse_year("") is None

    def test_none_returns_none(self):
        assert _parse_year(None) is None

    def test_no_year_returns_none(self):
        assert _parse_year("no date here") is None


# ---------------------------------------------------------------------------
# compute_recency_factor
# ---------------------------------------------------------------------------

class TestComputeRecencyFactor:
    def test_current_year_is_one(self):
        today = str(_today_year())
        factor = compute_recency_factor(today)
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_half_life_at_seven_years(self):
        year_seven_ago = str(_today_year() - 7)
        factor = compute_recency_factor(year_seven_ago)
        assert factor == pytest.approx(0.5, abs=0.02)

    def test_fourteen_years_ago_is_quarter(self):
        year_fourteen_ago = str(_today_year() - 14)
        factor = compute_recency_factor(year_fourteen_ago)
        assert factor == pytest.approx(0.25, abs=0.02)

    def test_unknown_date_neutral(self):
        assert compute_recency_factor("") == pytest.approx(0.5)
        assert compute_recency_factor(None) == pytest.approx(0.5)

    def test_factor_in_unit_interval(self):
        for year in ["1990", "2000", "2010", "2020", "2025"]:
            f = compute_recency_factor(year)
            assert 0.0 < f <= 1.0

    def test_older_scores_lower_than_newer(self):
        f_old  = compute_recency_factor(str(_today_year() - 15))
        f_new  = compute_recency_factor(str(_today_year() - 2))
        assert f_old < f_new


# ---------------------------------------------------------------------------
# compute_grounding_factor
# ---------------------------------------------------------------------------

class TestComputeGroundingFactor:
    def test_empty_returns_baseline(self):
        assert compute_grounding_factor("") == pytest.approx(0.75)

    def test_analysis_section_boosts(self):
        text_with = "Analysis [1] The test requires a severe and prolonged disability."
        text_without = "[1] The test requires a severe and prolonged disability."
        assert compute_grounding_factor(text_with) > compute_grounding_factor(text_without)

    def test_legal_citations_boost(self):
        dense = " ".join(["section 42"] * 20)
        sparse = "some text with no citations"
        assert compute_grounding_factor(dense) > compute_grounding_factor(sparse)

    def test_reasoning_markers_boost(self):
        rich = "I find that the evidence shows therefore I am satisfied the law applies."
        plain = "The case was decided."
        assert compute_grounding_factor(rich) > compute_grounding_factor(plain)

    def test_result_always_in_range(self):
        texts = [
            "",
            "short text",
            "Analysis " + "section 42 " * 50 + "I find therefore I conclude " * 20,
        ]
        for t in texts:
            g = compute_grounding_factor(t)
            assert 0.75 <= g <= 1.0


# ---------------------------------------------------------------------------
# compute_outcome
# ---------------------------------------------------------------------------

class TestComputeOutcome:
    def test_allowed(self):
        text = "Decision [1] The appeal is allowed. The Appellant succeeds."
        assert compute_outcome(text) == "Allowed"

    def test_dismissed(self):
        text = "Decision [1] The appeal is dismissed. The Appellant cannot receive EI."
        assert compute_outcome(text) == "Dismissed"

    def test_ambiguous_returns_unknown(self):
        text = "The Tribunal considered all submissions carefully."
        assert compute_outcome(text) == "Unknown"

    def test_empty_returns_unknown(self):
        assert compute_outcome("") == "Unknown"

    def test_both_keywords_returns_unknown(self):
        # Rare edge case: both words appear (e.g. "the appeal is allowed... the cross-appeal is dismissed")
        text = "The appeal is allowed. The cross-appeal is dismissed."
        assert compute_outcome(text) == "Unknown"


# ---------------------------------------------------------------------------
# recency_label
# ---------------------------------------------------------------------------

class TestRecencyLabel:
    def test_very_recent(self):
        assert recency_label(str(_today_year() - 2)) == "Recent"

    def test_boundary_recent(self):
        assert recency_label(str(_today_year() - 5)) == "Recent"

    def test_older(self):
        assert recency_label(str(_today_year() - 8)) == "Older"

    def test_historical(self):
        assert recency_label(str(_today_year() - 15)) == "Historical"

    def test_unknown_date(self):
        assert recency_label("") == "Unknown"
        assert recency_label("not a date") == "Unknown"


# ---------------------------------------------------------------------------
# relevance_label
# ---------------------------------------------------------------------------

class TestRelevanceLabel:
    def test_high(self):
        assert relevance_label(0.75) == "High"
        assert relevance_label(0.65) == "High"

    def test_medium(self):
        assert relevance_label(0.50) == "Medium"
        assert relevance_label(0.35) == "Medium"

    def test_low(self):
        assert relevance_label(0.20) == "Low"
        assert relevance_label(0.0)  == "Low"


# ---------------------------------------------------------------------------
# authority_risk_label
# ---------------------------------------------------------------------------

class TestAuthorityRiskLabel:
    def test_low_risk_recent_and_high_relevance(self):
        recent = str(_today_year() - 1)
        assert authority_risk_label(recent, 0.80) == "Low"

    def test_medium_risk_older(self):
        older = str(_today_year() - 8)
        assert authority_risk_label(older, 0.70) == "Medium"

    def test_medium_risk_medium_relevance_recent(self):
        recent = str(_today_year() - 2)
        assert authority_risk_label(recent, 0.50) == "Medium"

    def test_verify_historical(self):
        old = str(_today_year() - 15)
        assert authority_risk_label(old, 0.80) == "Verify"

    def test_verify_low_relevance(self):
        recent = str(_today_year() - 1)
        assert authority_risk_label(recent, 0.20) == "Verify"

    def test_verify_unknown_date(self):
        assert authority_risk_label("", 0.90) == "Verify"


# ---------------------------------------------------------------------------
# assess (combined)
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Decision [1] The appeal is allowed.\n"
    "Analysis [2] I find the evidence shows the Appellant meets the criteria "
    "under section 42 of the Canada Pension Plan.\n"
    "[3] Therefore the appeal succeeds.\n"
)


class TestAssess:
    def test_returns_authority_profile(self):
        recent = str(_today_year() - 1)
        result = assess(0.80, recent, SAMPLE_TEXT)
        assert isinstance(result, AuthorityProfile)

    def test_authority_score_below_reranker_score(self):
        # recency × grounding ≤ 1, so authority_score ≤ reranker_score
        recent = str(_today_year() - 1)
        result = assess(0.80, recent, SAMPLE_TEXT)
        assert result.authority_score <= 0.80 + 1e-9

    def test_older_decision_scores_lower_than_recent(self):
        text = SAMPLE_TEXT
        old_result    = assess(0.80, str(_today_year() - 14), text)
        recent_result = assess(0.80, str(_today_year() - 1),  text)
        assert old_result.authority_score < recent_result.authority_score

    def test_labels_populated(self):
        result = assess(0.75, str(_today_year() - 3), SAMPLE_TEXT)
        assert result.relevance_label  in ("High", "Medium", "Low")
        assert result.recency_label    in ("Recent", "Older", "Historical", "Unknown")
        assert result.authority_risk   in ("Low", "Medium", "Verify")
        assert result.outcome          in ("Allowed", "Dismissed", "Unknown")

    def test_outcome_detected(self):
        result = assess(0.70, str(_today_year() - 2), SAMPLE_TEXT)
        assert result.outcome == "Allowed"

    def test_all_factors_in_valid_range(self):
        result = assess(0.60, str(_today_year() - 5), SAMPLE_TEXT)
        assert 0.0 < result.recency_factor   <= 1.0
        assert 0.75 <= result.grounding_factor <= 1.0
        assert 0.0 < result.authority_score  <= 1.0


# ---------------------------------------------------------------------------
# check_outcome_diversity
# ---------------------------------------------------------------------------

class TestCheckOutcomeDiversity:
    def test_mixed_outcomes_diverse(self):
        assert check_outcome_diversity(["Allowed", "Dismissed", "Allowed"]) is True

    def test_all_dismissed_not_diverse(self):
        assert check_outcome_diversity(["Dismissed", "Dismissed", "Dismissed"]) is False

    def test_all_allowed_not_diverse(self):
        assert check_outcome_diversity(["Allowed", "Allowed"]) is False

    def test_unknowns_ignored(self):
        # All resolved cases go one way, but unknowns shouldn't count as diversity
        assert check_outcome_diversity(["Dismissed", "Unknown", "Dismissed"]) is False

    def test_all_unknown_returns_true(self):
        # Not enough info — report diverse (conservative default)
        assert check_outcome_diversity(["Unknown", "Unknown"]) is True

    def test_single_resolved_returns_true(self):
        assert check_outcome_diversity(["Allowed"]) is True

    def test_empty_list_returns_true(self):
        assert check_outcome_diversity([]) is True
