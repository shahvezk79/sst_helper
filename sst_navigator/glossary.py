"""
Plain-Language Legal Glossary for SST Decisions
================================================

Provides hover/click glossary definitions for common legal terms that
appear in Social Security Tribunal decisions.  Each entry includes:

  - The legal term as it appears in decisions
  - A plain-English definition (1-2 sentences)
  - What the term usually means in SST context

Self-represented litigants (SRLs) encounter these terms frequently and
may not understand their specific legal meaning in the SST context.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryEntry:
    """A single glossary term with plain-language explanation."""
    term: str
    definition: str
    sst_context: str


# High-impact terms that appear frequently in SST decisions.
# Ordered roughly by frequency of appearance across EI, CPP-D, and OAS cases.
GLOSSARY: list[GlossaryEntry] = [
    GlossaryEntry(
        term="just cause",
        definition=(
            "A legally acceptable reason for doing something. In employment "
            "law, it means circumstances where a reasonable person in the same "
            "situation would have had no other realistic choice."
        ),
        sst_context=(
            "In EI cases, 'just cause' determines whether you had a valid "
            "reason for leaving your job voluntarily. The Tribunal asks: was "
            "leaving the only reasonable option given all the circumstances?"
        ),
    ),
    GlossaryEntry(
        term="voluntary leaving",
        definition=(
            "When an employee chooses to leave their job on their own, rather "
            "than being fired or laid off. This can include resigning, quitting, "
            "or refusing to continue working."
        ),
        sst_context=(
            "If you left your job voluntarily without 'just cause,' you may be "
            "disqualified from receiving EI benefits. The Tribunal looks at all "
            "the circumstances to decide if leaving was your only reasonable option."
        ),
    ),
    GlossaryEntry(
        term="severe and prolonged disability",
        definition=(
            "A disability that prevents you from regularly doing any type of "
            "substantially gainful work and is long-term or likely to result "
            "in death."
        ),
        sst_context=(
            "To qualify for CPP Disability benefits, your disability must be "
            "both 'severe' (you can't regularly do any substantially gainful "
            "work) AND 'prolonged' (it is long-continued and of indefinite "
            "duration, or is likely to result in death)."
        ),
    ),
    GlossaryEntry(
        term="minimum qualifying period",
        definition=(
            "The date by which you must prove your disability existed. It is "
            "calculated based on how long you contributed to the Canada "
            "Pension Plan."
        ),
        sst_context=(
            "For CPP-D, you must show your disability was severe and prolonged "
            "on or before your MQP. If your MQP has passed, you need to prove "
            "you were already disabled by that date."
        ),
    ),
    GlossaryEntry(
        term="balance of probabilities",
        definition=(
            "The standard of proof in civil and administrative cases. It means "
            "'more likely than not' -- greater than a 50% chance that something "
            "is true."
        ),
        sst_context=(
            "The Tribunal decides cases on the 'balance of probabilities,' not "
            "'beyond a reasonable doubt.' You need to show it is more likely "
            "than not that you meet the legal requirements."
        ),
    ),
    GlossaryEntry(
        term="misconduct",
        definition=(
            "Behaviour by an employee that is deliberate, wilful, or so "
            "reckless that it amounts to wilful disregard of the employer's "
            "interests."
        ),
        sst_context=(
            "If you were fired for 'misconduct' under the EI Act, you may be "
            "disqualified from benefits. The Tribunal must decide whether your "
            "actions were wilful or so careless they were essentially deliberate."
        ),
    ),
    GlossaryEntry(
        term="substantially gainful",
        definition=(
            "Work that provides a meaningful income and requires significant "
            "physical or mental effort. Not just any small or occasional task."
        ),
        sst_context=(
            "For CPP-D, 'severe' disability means you cannot regularly pursue "
            "any 'substantially gainful occupation.' The Tribunal considers "
            "whether you can realistically hold a job that pays a livable wage."
        ),
    ),
    GlossaryEntry(
        term="availability for work",
        definition=(
            "Being ready, willing, and able to work each working day. You "
            "must also be actively looking for a suitable job."
        ),
        sst_context=(
            "To receive EI benefits, you must prove you were 'available for "
            "work.' The Tribunal looks at whether you had a genuine desire to "
            "work, made reasonable job search efforts, and had no personal "
            "conditions that would prevent you from accepting a job."
        ),
    ),
    GlossaryEntry(
        term="real and substantial connection",
        definition=(
            "A meaningful and genuine link between two things -- often between "
            "your reason for leaving a job and the actual departure."
        ),
        sst_context=(
            "In EI voluntary leaving cases, the Tribunal checks whether there "
            "is a 'real and substantial connection' between your stated reason "
            "for quitting and your decision to leave when you did."
        ),
    ),
    GlossaryEntry(
        term="de novo hearing",
        definition=(
            "A completely fresh hearing where the Tribunal considers the case "
            "from scratch, as if no previous decision had been made."
        ),
        sst_context=(
            "Appeals at the SST General Division are usually 'de novo,' meaning "
            "the Tribunal member makes their own independent decision based on "
            "all the evidence, not just whether the original decision was wrong."
        ),
    ),
    GlossaryEntry(
        term="retroactive benefits",
        definition=(
            "Benefits paid for a period before the date of the application "
            "or decision. Back-payment for time you were already eligible."
        ),
        sst_context=(
            "If you applied late for CPP-D or OAS, you may receive retroactive "
            "benefits for up to a certain number of months before your "
            "application date, depending on the program rules."
        ),
    ),
    GlossaryEntry(
        term="suitable employment",
        definition=(
            "A job that is reasonable for you to accept given your skills, "
            "experience, health, location, pay expectations, and personal "
            "circumstances."
        ),
        sst_context=(
            "For EI, refusing a 'suitable employment' offer can lead to "
            "disqualification. The Tribunal weighs factors like distance, "
            "working conditions, pay, and your qualifications to decide "
            "whether a job was suitable for you."
        ),
    ),
]

# Build a lookup dict keyed by lowercased term for fast matching.
_GLOSSARY_LOOKUP: dict[str, GlossaryEntry] = {
    entry.term.lower(): entry for entry in GLOSSARY
}

# Pre-compiled regex for detecting glossary terms in text.
# Uses word boundaries and case-insensitive matching.
_GLOSSARY_PATTERN: re.Pattern = re.compile(
    r"\b(" + "|".join(re.escape(e.term) for e in GLOSSARY) + r")\b",
    re.IGNORECASE,
)


def detect_terms(text: str) -> list[GlossaryEntry]:
    """Return glossary entries for all legal terms found in the given text.

    Returns a deduplicated list preserving the order of first appearance.
    """
    seen: set[str] = set()
    found: list[GlossaryEntry] = []
    for match in _GLOSSARY_PATTERN.finditer(text):
        key = match.group(0).lower()
        if key not in seen:
            seen.add(key)
            entry = _GLOSSARY_LOOKUP.get(key)
            if entry:
                found.append(entry)
    return found


def get_entry(term: str) -> GlossaryEntry | None:
    """Look up a single glossary entry by term (case-insensitive)."""
    return _GLOSSARY_LOOKUP.get(term.lower())
