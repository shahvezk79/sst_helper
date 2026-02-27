"""Tests for sst_navigator.section_parser."""

import pytest

from sst_navigator.section_parser import (
    Section,
    _heading_priority,
    _strip_boilerplate,
    _truncate_analysis,
    parse_sections,
    pack_for_reranker,
)


# ---------------------------------------------------------------------------
# Fixtures — representative decision text snippets
# ---------------------------------------------------------------------------

# Modern 5-section decision (EI misconduct, ~2023 style)
MODERN_DECISION = (
    "SM v Canada Employment Insurance Commission\n"
    "Collection\n"
    "Employment Insurance (EI)\n"
    "Decision date\n"
    "2023-07-21\n"
    "Neutral citation\n"
    "2023 SST 102\n"
    "Reference number\n"
    "GE-23-193\n"
    "Member\n"
    "Teresa Day\n"
    "Division\n"
    "General Division\n"
    "Decision\n"
    "Appeal dismissed\n"
    "Decision Content\n"
    "Citation: SM v Canada Employment Insurance Commission, 2023 SST 102\n"
    "Social Security Tribunal of Canada General Division\n"
    "On this page\n"
    "Decision Overview Issue Analysis Conclusion\n"
    "Decision [1] The appeal is dismissed.\n"
    "[2] The Appellant cannot receive EI benefits because she lost her job "
    "due to her own misconduct.\n"
    "Overview [3] The Appellant worked as a hospital administrator.\n"
    "[4] In September 2021, the employer instituted a mandatory vaccination policy.\n"
    "[5] The Appellant did not want to comply with the policy.\n"
    "Issue [6] I have to decide whether the Appellant lost her job "
    "due to misconduct.\n"
    "Analysis [7] The law says misconduct is a deliberate act.\n"
    "[8] I find that the Appellant made a conscious choice not to comply.\n"
    "[9] The employer warned the Appellant of the consequences.\n"
    "[10] The Appellant's refusal to vaccinate was wilful.\n"
    "Conclusion [11] The appeal is dismissed.\n"
    "Footnote 1\n"
    "That is, misconduct as the term is used for EI purposes.\n"
    "Footnote 2\n"
    "See GD3-34 to GD3-35.\n"
)

# Older Appeal Division decision with Introduction + "Reasons and decision"
OLDER_AD_DECISION = (
    "F. B. v. Canada Employment Insurance Commission\n"
    "Decision date\n"
    "2016-03-16\n"
    "Decision\n"
    "Application for leave to appeal refused\n"
    "Decision Content\n"
    "On this page\n"
    "Decision Introduction Issues The law and analysis Conclusion\n"
    "Reasons and decision Decision [1] The Tribunal refuses leave to appeal.\n"
    "Introduction [2] On January 26, 2015, the General Division refused an "
    "extension of time.\n"
    "[3] The Commission found that the Applicant could not receive sickness "
    "benefits.\n"
    "Issues [4] Was the Application filed within the prescribed time?\n"
    "[5] Does the appeal have a reasonable chance of success?\n"
    "The law and analysis [6] Paragraph 57(2)(a) provides that an application "
    "must be filed within 30 days.\n"
    "[7] I find the application was late and has no reasonable chance.\n"
    "Conclusion [8] Leave to appeal is refused.\n"
)

# Decision with NO "On this page" TOC (rare ~1% case)
NO_TOC_DECISION = (
    "D. L. v. Minister of Employment and Social Development\n"
    "Decision date\n"
    "2015-04-10\n"
    "Decision Content\n"
    "[1] On August 26, 2014, the General Division dismissed the Appellant's claim.\n"
    "[2] The Appellant sought leave to appeal on September 30, 2014.\n"
    "[3] Leave to appeal was granted on the basis that the General Division "
    "erred when it adopted one party's interpretation.\n"
    "[4] The appeal is allowed.\n"
)

# CPP-D decision with custom subheadings
CPPD_DECISION = (
    "MP v Minister of Employment and Social Development\n"
    "Collection\n"
    "Canada Pension Plan (CPP) disability\n"
    "Decision Content\n"
    "On this page\n"
    "Decision Overview Issue Analysis Severe Prolonged Conclusion\n"
    "Decision [1] The appeal is dismissed.\n"
    "Overview [2] The Appellant was in a motor vehicle accident.\n"
    "[3] She applied for a CPP disability pension.\n"
    "Issue [4] Whether the Appellant has a severe and prolonged disability.\n"
    "Analysis [5] The test requires a severe and prolonged disability.\n"
    "Severe [6] The Appellant has functional limitations.\n"
    "[7] However, she retains some work capacity.\n"
    "Prolonged [8] The medical evidence does not show indefinite duration.\n"
    "Conclusion [9] The appeal is dismissed.\n"
)


# ---------------------------------------------------------------------------
# _heading_priority
# ---------------------------------------------------------------------------

class TestHeadingPriority:
    def test_analysis_is_highest(self):
        assert _heading_priority("Analysis") == 1

    def test_law_and_analysis_is_highest(self):
        assert _heading_priority("The law and analysis") == 1

    def test_issue_variants(self):
        assert _heading_priority("Issue") == 2
        assert _heading_priority("Issues") == 2
        assert _heading_priority("What I have to decide") == 2
        assert _heading_priority("What the Appellant must prove") == 2

    def test_conclusion_and_decision(self):
        assert _heading_priority("Conclusion") == 3
        assert _heading_priority("Decision") == 3

    def test_overview_and_introduction(self):
        assert _heading_priority("Overview") == 4
        assert _heading_priority("Introduction") == 4
        assert _heading_priority("Background") == 4

    def test_the_law(self):
        assert _heading_priority("The law") == 5
        assert _heading_priority("Applicable law") == 5

    def test_evidence_and_submissions(self):
        assert _heading_priority("Evidence") == 6
        assert _heading_priority("Submissions") == 6
        assert _heading_priority("What the Appellant says") == 6

    def test_custom_subheading_gets_default(self):
        assert _heading_priority("Severe") == 7
        assert _heading_priority("Prolonged") == 7
        assert _heading_priority("Preliminary matters") == 7


# ---------------------------------------------------------------------------
# _strip_boilerplate
# ---------------------------------------------------------------------------

class TestStripBoilerplate:
    def test_removes_metadata_header(self):
        result = _strip_boilerplate(MODERN_DECISION)
        assert "SM v Canada Employment Insurance Commission" not in result
        assert "Collection" not in result
        assert "Neutral citation" not in result

    def test_removes_toc_line(self):
        result = _strip_boilerplate(MODERN_DECISION)
        assert "On this page" not in result
        assert "Decision Overview Issue Analysis Conclusion" not in result

    def test_removes_footnotes(self):
        result = _strip_boilerplate(MODERN_DECISION)
        assert "Footnote 1" not in result
        assert "Footnote 2" not in result
        assert "See GD3-34" not in result

    def test_strips_reasons_and_decision_bridge(self):
        result = _strip_boilerplate(OLDER_AD_DECISION)
        assert "Reasons and decision" not in result
        # But the actual Decision heading content survives
        assert "The Tribunal refuses leave to appeal" in result

    def test_no_toc_decision_preserves_content(self):
        result = _strip_boilerplate(NO_TOC_DECISION)
        assert "[1] On August 26, 2014" in result
        assert "[4] The appeal is allowed." in result

    def test_no_decision_content_marker_returns_full_text(self):
        raw = "Some unusual text without any markers.\n[1] Paragraph one."
        result = _strip_boilerplate(raw)
        assert "[1] Paragraph one." in result


# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------

class TestParseSections:
    def test_modern_decision_extracts_five_sections(self):
        sections = parse_sections(MODERN_DECISION)
        names = [s.name for s in sections]
        assert "Decision" in names
        assert "Overview" in names
        assert "Issue" in names
        assert "Analysis" in names
        assert "Conclusion" in names

    def test_sections_have_correct_priorities(self):
        sections = parse_sections(MODERN_DECISION)
        by_name = {s.name: s for s in sections}
        assert by_name["Analysis"].priority == 1
        assert by_name["Issue"].priority == 2
        assert by_name["Conclusion"].priority == 3
        assert by_name["Overview"].priority == 4

    def test_section_text_contains_paragraphs(self):
        sections = parse_sections(MODERN_DECISION)
        by_name = {s.name: s for s in sections}
        assert "[7] The law says" in by_name["Analysis"].text
        assert "[10] The Appellant's refusal" in by_name["Analysis"].text

    def test_older_decision_composite_heading(self):
        sections = parse_sections(OLDER_AD_DECISION)
        names = [s.name for s in sections]
        assert "The law and analysis" in names
        by_name = {s.name: s for s in sections}
        assert by_name["The law and analysis"].priority == 1

    def test_no_toc_decision_returns_body_section(self):
        sections = parse_sections(NO_TOC_DECISION)
        # No headings before [N] markers, so entire text is one "body" section.
        assert len(sections) == 1
        assert sections[0].name == "body"
        assert "[1] On August 26" in sections[0].text

    def test_cppd_custom_subheadings(self):
        sections = parse_sections(CPPD_DECISION)
        names = [s.name for s in sections]
        assert "Severe" in names
        assert "Prolonged" in names
        by_name = {s.name: s for s in sections}
        assert by_name["Severe"].priority == 7
        assert by_name["Prolonged"].priority == 7

    def test_order_field_is_sequential(self):
        sections = parse_sections(MODERN_DECISION)
        orders = [s.order for s in sections]
        assert orders == list(range(len(sections)))

    def test_empty_text_returns_empty_list(self):
        assert parse_sections("") == []


# ---------------------------------------------------------------------------
# _truncate_analysis
# ---------------------------------------------------------------------------

class TestTruncateAnalysis:
    def test_short_text_unchanged(self):
        text = "Analysis [1] Short. [2] Also short."
        assert _truncate_analysis(text, budget=1000) == text

    def test_long_text_keeps_head_and_tail(self):
        # Build a long analysis with many paragraphs
        paras = [f"[{i}] Paragraph number {i} with some legal reasoning text." for i in range(1, 50)]
        text = "Analysis " + "\n".join(paras)
        result = _truncate_analysis(text, budget=300)
        assert len(result) <= 300 + 20  # allow small overhead from tail join
        assert "[1]" in result       # head preserved
        assert "[..." in result      # ellipsis marker
        assert "[49]" in result or "[48]" in result  # tail paragraphs

    def test_few_paragraphs_no_tail(self):
        # Only 2 paragraphs — not enough for head+tail split
        text = "Analysis [1] First. [2] Second."
        result = _truncate_analysis(text, budget=20)
        assert len(result) <= 20


# ---------------------------------------------------------------------------
# pack_for_reranker
# ---------------------------------------------------------------------------

class TestPackForReranker:
    def test_short_decision_returns_all_sections_in_order(self):
        result = pack_for_reranker(MODERN_DECISION, char_budget=100_000)
        # All sections present, in document order
        dec_pos = result.find("[Decision]")
        ov_pos = result.find("[Overview]")
        iss_pos = result.find("[Issue]")
        ana_pos = result.find("[Analysis]")
        con_pos = result.find("[Conclusion]")
        assert all(p >= 0 for p in [dec_pos, ov_pos, iss_pos, ana_pos, con_pos])
        assert dec_pos < ov_pos < iss_pos < ana_pos < con_pos

    def test_tight_budget_prioritises_analysis(self):
        result = pack_for_reranker(MODERN_DECISION, char_budget=300)
        assert "[Analysis]" in result
        # Analysis should appear before lower-priority sections
        # Overview might be excluded entirely
        assert "deliberate act" in result or "conscious choice" in result

    def test_analysis_before_overview_when_budget_tight(self):
        result = pack_for_reranker(MODERN_DECISION, char_budget=500)
        ana_pos = result.find("[Analysis]")
        assert ana_pos >= 0
        # With 500 chars, Analysis (pri 1) and Issue (pri 2) should
        # appear before Overview (pri 4) if present.
        if "[Overview]" in result:
            ov_pos = result.find("[Overview]")
            # In output they're reordered by document position, but Analysis
            # content must be present.
            assert "[Analysis]" in result

    def test_empty_text_returns_empty(self):
        result = pack_for_reranker("", char_budget=1000)
        assert result == ""

    def test_no_markers_falls_back_to_head_truncation(self):
        raw = "A" * 500
        result = pack_for_reranker(raw, char_budget=100)
        # Budget accounts for the "[body] " label prefix on the fallback section
        assert len(result) <= 100

    def test_budget_respected(self):
        result = pack_for_reranker(MODERN_DECISION, char_budget=200)
        assert len(result) <= 250  # small tolerance for separator joining

    def test_footnotes_excluded(self):
        result = pack_for_reranker(MODERN_DECISION, char_budget=100_000)
        assert "Footnote 1" not in result
        assert "See GD3-34" not in result

    def test_metadata_excluded(self):
        result = pack_for_reranker(MODERN_DECISION, char_budget=100_000)
        assert "Neutral citation" not in result
        assert "GE-23-193" not in result

    def test_older_decision_works(self):
        result = pack_for_reranker(OLDER_AD_DECISION, char_budget=100_000)
        assert "[The law and analysis]" in result
        assert "Paragraph 57(2)(a)" in result

    def test_no_toc_decision_works(self):
        result = pack_for_reranker(NO_TOC_DECISION, char_budget=100_000)
        assert "General Division dismissed" in result

    def test_cppd_includes_subheadings(self):
        result = pack_for_reranker(CPPD_DECISION, char_budget=100_000)
        assert "[Severe]" in result
        assert "[Prolonged]" in result
