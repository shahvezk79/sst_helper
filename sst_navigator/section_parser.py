"""
Section-aware text packing for SST decision documents.

Parses the standard structure of Social Security Tribunal decisions
(Decision → Overview → Issue → Analysis → Conclusion) and packs the
most legally salient sections into a character budget for the reranker.
"""

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Section data structure
# ---------------------------------------------------------------------------

class Section(NamedTuple):
    name: str       # heading text, e.g. "Analysis"
    text: str       # body text (heading + paragraphs)
    priority: int   # 1 = highest value for relevance scoring
    order: int      # original position in the document


# ---------------------------------------------------------------------------
# Priority mapping
# ---------------------------------------------------------------------------

# Ordered list of (compiled pattern, priority).  First match wins.
_PRIORITY_RULES: list[tuple[re.Pattern, int]] = [
    (re.compile(r"analysis|law and analysis", re.IGNORECASE), 1),
    (re.compile(
        r"issue|issues|what i have to decide|what i must decide|what the .* must prove",
        re.IGNORECASE,
    ), 2),
    (re.compile(r"conclusion|decision", re.IGNORECASE), 3),
    (re.compile(r"overview|introduction|background", re.IGNORECASE), 4),
    (re.compile(r"the law|applicable law|what the law says|legal framework", re.IGNORECASE), 5),
    (re.compile(r"evidence|submissions|what the .* says", re.IGNORECASE), 6),
]

_DEFAULT_PRIORITY = 7


def _heading_priority(name: str) -> int:
    for pattern, priority in _PRIORITY_RULES:
        if pattern.search(name):
            return priority
    return _DEFAULT_PRIORITY


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _strip_boilerplate(text: str) -> str:
    """Remove metadata header and trailing footnote block."""
    # --- Strip metadata before "Decision Content" ---
    dc_idx = text.find("Decision Content")
    if dc_idx != -1:
        text = text[dc_idx + len("Decision Content"):]

    # --- Skip "On this page" TOC line ---
    toc_idx = text.find("On this page")
    if toc_idx != -1:
        # The TOC is: "On this page\n<heading list>\n"
        # Find the end of the heading-list line that follows.
        after_toc = text.find("\n", toc_idx)
        if after_toc != -1:
            next_newline = text.find("\n", after_toc + 1)
            if next_newline != -1:
                text = text[next_newline + 1:]
            else:
                text = text[after_toc + 1:]
        else:
            text = text[toc_idx + len("On this page"):]

    # --- Skip "Reasons and decision" bridge line (older decisions) ---
    rd_match = re.match(r"\s*Reasons and decision\s*", text)
    if rd_match:
        text = text[rd_match.end():]

    # --- Strip trailing footnote block ---
    # Standalone footnotes appear as "\nFootnote <N>\n..." at the end.
    fn_match = re.search(r"\nFootnote 1\n", text)
    if fn_match:
        text = text[:fn_match.start()]

    return text.strip()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

# Matches a heading followed (possibly with whitespace) by a [N] paragraph
# marker.  The heading starts with a capital letter and contains letters,
# spaces, hyphens, dashes, commas, apostrophes, and slashes.
_HEADING_RE = re.compile(
    r"^([A-Z][A-Za-z][A-Za-z \-–—,/\']{0,80}?)"  # heading text
    r"\s*"
    r"(?=\[\d+\])",                                 # lookahead for [N]
    re.MULTILINE,
)


def parse_sections(text: str) -> list[Section]:
    """Parse an SST decision into prioritised sections.

    Returns a list of Section objects sorted by priority (ascending)
    with document order as tiebreaker within the same tier.
    """
    cleaned = _strip_boilerplate(text)
    if not cleaned:
        return []

    matches = list(_HEADING_RE.finditer(cleaned))
    if not matches:
        # No headings found — return the entire cleaned text as one section.
        return [Section(name="body", text=cleaned, priority=_DEFAULT_PRIORITY, order=0)]

    sections: list[Section] = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        body = cleaned[start:end].strip()
        sections.append(Section(
            name=name,
            text=body,
            priority=_heading_priority(name),
            order=i,
        ))

    return sections


# ---------------------------------------------------------------------------
# Greedy packing
# ---------------------------------------------------------------------------

_PARAGRAPH_RE = re.compile(r"\[\d+\]")

_SECTION_SEP = "\n\n"


def _truncate_analysis(text: str, budget: int) -> str:
    """Truncate an Analysis section keeping the head and last 2 paragraphs."""
    if len(text) <= budget:
        return text

    # Reserve space for the tail excerpt.
    tail_budget = budget // 5
    head_budget = budget - tail_budget - len(" [...] ")

    # Find the last 2 paragraph markers for the tail.
    para_matches = list(_PARAGRAPH_RE.finditer(text))
    if len(para_matches) >= 3:
        tail_start = para_matches[-2].start()
        tail = text[tail_start:]
        if len(tail) > tail_budget:
            tail = tail[:tail_budget]
    else:
        tail = ""
        head_budget = budget

    head = text[:head_budget]
    if tail:
        return head + " [...] " + tail
    return head


def pack_for_reranker(text: str, char_budget: int) -> str:
    """Pack an SST decision into a character budget for the reranker.

    Extracts sections, prioritises legally salient content (Analysis,
    Issue, Conclusion), and greedily fills the budget.  Falls back to
    head-truncation when section parsing fails.
    """
    sections = parse_sections(text)
    if not sections:
        return text[:char_budget]

    # If the entire cleaned text (including [name] labels) fits, just return it.
    total = (
        sum(len(s.text) + len(s.name) + 3 for s in sections)  # +3 for "[] "
        + len(_SECTION_SEP) * (len(sections) - 1)
    )
    if total <= char_budget:
        return _SECTION_SEP.join(f"[{s.name}] {s.text}" for s in sorted(sections, key=lambda s: s.order))

    # Sort by (priority, original order) — highest-value sections first.
    ordered = sorted(sections, key=lambda s: (s.priority, s.order))

    packed: list[tuple[int, str]] = []  # (original order, formatted text)
    remaining = char_budget

    for s in ordered:
        if remaining <= 0:
            break

        label = f"[{s.name}] "
        label_len = len(label)
        body_budget = remaining - label_len - len(_SECTION_SEP)
        if body_budget <= 0:
            break

        if len(s.text) <= body_budget:
            packed.append((s.order, label + s.text))
            remaining -= len(label) + len(s.text) + len(_SECTION_SEP)
        else:
            # Section too large — truncate.
            if s.priority == 1:
                truncated = _truncate_analysis(s.text, body_budget)
            else:
                truncated = s.text[:body_budget]
            packed.append((s.order, label + truncated))
            remaining -= len(label) + len(truncated) + len(_SECTION_SEP)

    # Reassemble in original document order.
    packed.sort(key=lambda t: t[0])
    return _SECTION_SEP.join(text for _, text in packed)
