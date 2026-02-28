"""Tests for sst_navigator.generator."""

from sst_navigator.generator import _sanitize_case_card_output


def test_sanitize_removes_think_block_and_keeps_summary():
    raw = (
        "<think>Internal reasoning that should not be shown.</think>\n\n"
        "**Issue:** Whether benefits were denied.\n"
        "**Key Facts:** The claimant left work voluntarily.\n"
        "**Test Applied:** Voluntary leaving without just cause.\n"
        "**Outcome:** Appeal dismissed."
    )

    cleaned = _sanitize_case_card_output(raw)

    assert "<think>" not in cleaned
    assert "Internal reasoning" not in cleaned
    assert cleaned.startswith("**Issue:**")
    assert "**Outcome:** Appeal dismissed." in cleaned


def test_sanitize_drops_preamble_before_first_expected_section():
    raw = (
        "Reasoning trace:\nStep 1...\n\n"
        "**Issue:** Whether there was misconduct.\n"
        "**Key Facts:** Employer policy was breached.\n"
        "**Test Applied:** Wilful misconduct standard.\n"
        "**Outcome:** Appeal dismissed."
    )

    cleaned = _sanitize_case_card_output(raw)

    assert "Reasoning trace" not in cleaned
    assert cleaned.startswith("**Issue:**")


def test_sanitize_returns_trimmed_text_when_no_sections_present():
    raw = "  Plain answer without explicit headings.  "

    cleaned = _sanitize_case_card_output(raw)

    assert cleaned == "Plain answer without explicit headings."
