# Plan: Section-Aware Reranker Input Packing

## Problem

The reranker truncates documents with a blunt character slice (`reranker.py:122`):

```python
doc_text = cand["text"][:max_tokens * 4]   # first ~32K characters
```

SST decision texts begin with a large metadata/boilerplate block (case name, citation,
member info, hearing details) and end with footnotes — neither of which carries much
legal reasoning signal.  The section most important for relevance scoring — **Analysis**
— often appears in the middle-to-late portion of the text and can be clipped by head
truncation.

## Observed structure of `unofficial_text_en`

Based on scanning all 17,329 decisions in the A2AJ dataset:

```
╔═══════════════════════════════════════════════════════════╗
║  METADATA BLOCK                                          ║
║  Case name, collection, date, citation, member,          ║
║  division, hearing type, participants, etc.               ║
╠═══════════════════════════════════════════════════════════╣
║  "Decision Content"                                      ║
║  ...                                                     ║
║  "On this page"          ← TOC line (99% of decisions)   ║
║  Decision Overview Issue Analysis Conclusion              ║
╠═══════════════════════════════════════════════════════════╣
║  Decision  [1] The appeal is dismissed. ...               ║
║  Overview  [3] The Appellant worked as ...                ║
║  Issue     [11] I have to decide whether ...              ║
║  Analysis  [13] The law says ... [14] I find that ...     ║
║  Conclusion [47] The appeal is dismissed.                 ║
╠═══════════════════════════════════════════════════════════╣
║  Footnote 1 ...                                          ║
║  Footnote 2 ...                                          ║
╚═══════════════════════════════════════════════════════════╝
```

### Key stats

| Pattern                     | Count      | Coverage |
|-----------------------------|------------|----------|
| "Decision Content" marker   | 17,329     | 100%     |
| "On this page" TOC line     | 17,146     | 99%      |
| Footnotes present           | 11,813     | 68%      |
| Unique heading strings      | 8,731      | —        |

### Top section headings (frequency across corpus)

| Heading                      | Count  | Notes                                     |
|------------------------------|--------|-------------------------------------------|
| Conclusion                   | 16,412 | Nearly universal                           |
| Overview                     | 11,522 | Context/background                         |
| Analysis                     | 10,008 | Core reasoning — highest legal value       |
| Issue / Issues               | 13,322 | Combined; identifies legal questions       |
| Introduction                 | 3,136  | Older decisions use this instead of Overview |
| The law                      | 2,945  | Legal framework section                    |
| Submissions                  | 2,085  | Party arguments                            |
| Decision (as heading)        | 1,281  | Short outcome statement (1-2 paragraphs)   |
| Evidence                     | 950    | Factual evidence summary                   |
| Preliminary matters          | 781    | Procedural issues                          |

Heading names vary significantly (8,731 unique strings) due to custom subheadings
(e.g., "Wanting to go back to work", "Severe", "Prolonged") and composite headings
(e.g., "The law and analysis", "Law and analysis").

## Design

### New module: `sst_navigator/section_parser.py`

A single module with two public functions:

```
parse_sections(text: str) -> list[Section]
pack_for_reranker(text: str, char_budget: int) -> str
```

### Step 1 — Strip boilerplate and footnotes

1. Find the "Decision Content" marker (present in 100% of decisions).
   Everything before it is metadata — discard it.
2. Find the "On this page" line and skip past it (it's a TOC, not content).
3. Detect trailing footnotes: scan backwards for the last `Footnote \d+` block
   and trim it.

Fallback: if neither marker is found, use the full text (graceful degradation to
current behaviour).

### Step 2 — Split into sections

Use heading detection to split the body into named sections:

```python
HEADING_RE = re.compile(
    r'^([A-Z][A-Za-z][A-Za-z \-–—,/\']{1,80}?)'  # heading text
    r'\s*'
    r'(?=\[\d+\])',                                 # followed by [N] paragraph marker
    re.MULTILINE,
)
```

This captures both canonical headings ("Analysis") and custom subheadings
("Making efforts to find a suitable job").  The `[N]` lookahead distinguishes
true section headings from incidental title-case words in body text.

Each section becomes a `Section(name: str, text: str, priority: int)` namedtuple
or dataclass.

### Step 3 — Assign priorities

Map known heading names to priority tiers:

| Priority | Headings (case-insensitive substring match)                |
|----------|------------------------------------------------------------|
| 1 (highest) | "analysis", "law and analysis"                          |
| 2        | "issue", "issues", "what i have to decide", "what i must decide", "what the .* must prove" |
| 3        | "conclusion", "decision" (as section, not metadata)        |
| 4        | "overview", "introduction", "background"                   |
| 5        | "the law", "applicable law", "what the law says", "legal framework" |
| 6        | "evidence", "submissions", "what the appellant says"       |
| 7 (lowest) | everything else (custom subheadings, preliminary matters) |

Rationale: **Analysis** contains the tribunal member's actual reasoning and application
of the legal test — this is what makes a decision relevant to a query.  **Issue** frames
the legal question.  **Conclusion/Decision** states the outcome.  These three are most
predictive for relevance.

### Step 4 — Greedy packing

Given a character budget (default `max_tokens * 4 = 32,768` chars):

```
1. Sort sections by priority (ascending = highest first), with original
   document order as tiebreaker within the same priority tier.
2. Greedily add whole sections while they fit within budget.
3. If a section exceeds remaining budget:
   a. If it's priority 1 (Analysis) — include as much as fits,
      preferring the beginning (which states the legal test) and
      appending the last 2 paragraphs (which typically state the
      finding).
   b. Otherwise — truncate to fill remaining space.
4. Join packed sections with "\n\n" separators, preserving heading names
   as markers so the reranker sees section context.
```

Output format sent to the cross-encoder:

```
[Issue] [11] I have to decide whether the Appellant lost her job due to misconduct.

[Analysis] [13] The law says misconduct means ... [14] I find that ...
... (as much as fits)

[Conclusion] [47] The appeal is dismissed.

[Overview] [3] The Appellant worked as a hospital administrator ...
```

The section name labels (`[Issue]`, `[Analysis]`, etc.) give the cross-encoder
explicit structural cues without needing any model changes.

### Integration point

In `reranker.py`, replace:

```python
doc_text = cand["text"][:max_tokens * 4]
```

with:

```python
from .section_parser import pack_for_reranker
doc_text = pack_for_reranker(cand["text"], char_budget=max_tokens * 4)
```

No changes needed to `_build_prompt`, `_score_one`, or any other reranker logic.

## Edge cases and fallbacks

| Scenario                                | Handling                                   |
|-----------------------------------------|--------------------------------------------|
| No "Decision Content" marker found      | Use full text (current behaviour)          |
| No recognizable headings detected       | Return text with boilerplate/footnotes stripped, head-truncated to budget |
| Only 1-2 sections detected              | Pack what exists, fill remainder with body text |
| Very short decision (under budget)      | Return full cleaned text — no packing needed |
| Composite headings ("The law and analysis") | Priority assigned by best-matching substring |

## What this does NOT require

- **No embedding recomputation** — this only affects Stage-2 reranker input
- **No model changes** — same cross-encoder, same prompt template
- **No config changes** — uses existing `RERANKER_MAX_TOKENS` budget
- **No data pipeline changes** — operates on raw `unofficial_text_en` at query time

## Files to create/modify

| File | Action |
|------|--------|
| `sst_navigator/section_parser.py` | **Create** — `parse_sections()` + `pack_for_reranker()` |
| `sst_navigator/reranker.py:122` | **Modify** — replace character slice with `pack_for_reranker()` call |
| `tests/test_section_parser.py` | **Create** — unit tests with representative decision snippets |

## Risks and mitigations

1. **Regex fragility on unusual decisions**: The heading regex relies on `[N]`
   paragraph markers.  The 99% "On this page" TOC coverage and the consistent
   `[N]` format across the corpus make this robust, but a handful of edge cases
   will fall through to the fallback path (full text, head-truncated).

2. **Priority ordering disagreements**: The priority table is a starting point.
   Once an eval set exists (recommendation 7), priorities can be tuned empirically
   by measuring MRR/NDCG impact of swapping tiers.

3. **Analysis section too long**: Analysis is often the largest section and may
   alone exceed the budget.  The head-plus-tail strategy (beginning for legal test,
   tail for finding) handles this without losing the conclusion of the reasoning.

## Estimated complexity

- `section_parser.py`: ~120-150 lines
- `reranker.py` change: 2-line swap
- Tests: ~100-150 lines with 4-5 representative decision snippets
- No changes to embedding pipeline, config, or data loader
