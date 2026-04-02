"""Tests for the plain-language legal glossary module."""

from sst_navigator.glossary import GLOSSARY, detect_terms, get_entry


class TestGlossary:
    def test_glossary_has_entries(self):
        assert len(GLOSSARY) >= 10

    def test_entries_have_required_fields(self):
        for entry in GLOSSARY:
            assert entry.term, "term must be non-empty"
            assert entry.definition, "definition must be non-empty"
            assert entry.sst_context, "sst_context must be non-empty"


class TestGetEntry:
    def test_existing_term(self):
        entry = get_entry("just cause")
        assert entry is not None
        assert entry.term == "just cause"

    def test_case_insensitive(self):
        entry = get_entry("Just Cause")
        assert entry is not None
        assert entry.term == "just cause"

    def test_nonexistent_term(self):
        assert get_entry("habeas corpus") is None


class TestDetectTerms:
    def test_detects_single_term(self):
        text = "The claimant had just cause for leaving."
        found = detect_terms(text)
        terms = [e.term for e in found]
        assert "just cause" in terms

    def test_detects_multiple_terms(self):
        text = (
            "The appellant's disability was severe and prolonged. "
            "The balance of probabilities standard was applied."
        )
        found = detect_terms(text)
        terms = [e.term for e in found]
        assert "severe and prolonged disability" in terms or "balance of probabilities" in terms

    def test_no_terms_returns_empty(self):
        text = "This sentence contains no legal terminology at all."
        found = detect_terms(text)
        assert found == []

    def test_deduplicates(self):
        text = "Just cause is required. She showed just cause for leaving."
        found = detect_terms(text)
        just_cause_entries = [e for e in found if e.term == "just cause"]
        assert len(just_cause_entries) == 1

    def test_case_insensitive_detection(self):
        text = "VOLUNTARY LEAVING without reason is problematic."
        found = detect_terms(text)
        terms = [e.term for e in found]
        assert "voluntary leaving" in terms
