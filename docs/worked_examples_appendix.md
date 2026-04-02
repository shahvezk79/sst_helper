# Appendix: Additional Worked Mini-Cases

These three cases are designed to improve transparency and perceived rigor.

---

## Mini-case 1: Ambiguous query

**Input query**
> “I left my job because my manager changed my shift and childcare became impossible. Do I still qualify?”

**Top outputs (example structure)**
1. Decision A — schedule change, family obligations, voluntary leaving analysis.
2. Decision B — claimant left due to personal obligations; availability and alternatives discussed.
3. Decision C — mixed factors (workplace change + personal constraints).

**Why these were returned**
- Query mixes potential “just cause” and availability issues.
- Retrieved decisions discuss whether reasonable alternatives existed.
- Analysis sections align with the user’s core fact pattern (schedule disruption + caregiving constraints).

**Caution note**
- Outcome often depends on evidence of alternatives attempted (e.g., accommodation requests, job search, documentation).

---

## Mini-case 2: Misleadingly phrased query

**Input query**
> “My boss fired me unfairly, so EI must approve me automatically, right?”

**Top outputs (example structure)**
1. Decision D — dismissal for alleged misconduct; tribunal applies statutory misconduct test.
2. Decision E — employer allegation not determinative; burden and evidence evaluated.
3. Decision F — termination reason disputed; conduct and intent examined.

**Why these were returned**
- Phrase “unfairly fired” is emotionally framed; legal issue is misconduct disqualification.
- Returned decisions explain that employer labels are not conclusive.
- Reranker favors decisions that explicitly apply the legal misconduct standard.

**Caution note**
- “Unfair” dismissal and EI eligibility are related but not identical legal questions.

---

## Mini-case 3: Mixed precedent outcomes

**Input query**
> “I refused to return to in-person work because of health concerns. Would that be misconduct?”

**Top outputs (example structure)**
1. Decision G — claimant successful due to documented medical limitations and communication efforts.
2. Decision H — claimant unsuccessful where evidence and accommodation steps were weak.
3. Decision I — partial/mixed reasoning with procedural context.

**Why these were returned**
- Similar fact patterns can produce different outcomes depending on evidence strength.
- The system surfaces both favorable and unfavorable precedents to avoid one-sided guidance.
- Section-aware retrieval emphasizes the Analysis section where balancing factors are discussed.

**Caution note**
- Mixed precedents should be presented with risk framing; users should review cited facts before relying on an answer.

---

## Reusable template (copy/paste)

```markdown
### Mini-case X: [Case type]

**Input query**
> "[User question]"

**Top outputs**
1. [Decision 1]
2. [Decision 2]
3. [Decision 3]

**Why these were returned**
- [Reason 1]
- [Reason 2]
- [Reason 3]

**Caution note**
- [Risk/limitation statement]
```
