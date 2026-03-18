# Feedback Action Plan — SST Decision Navigator

## Feedback Analysis & Prioritized Improvements

Based on instructor feedback (grade: 8.5/10) and peer reviews (3.77/5), I've categorized the feedback into **major** (must-fix, grade-impacting) and **minor** (nice-to-have) items.

---

## MAJOR ITEMS (High Priority)

### 1. Strengthen Validation & Testing Framework
**Source:** Instructor (primary concern), Peers (Sitara, Olivia, Gurnoor)

The instructor explicitly flagged this as a weakness. Three sub-tasks:

#### 1a. Add CanLII Baseline Comparison
- Run the same queries through CanLII's search and compare results
- Document which cases each system finds, where they overlap, and where they diverge
- Add a section to the notebook (or appendix) presenting this comparison in a table

#### 1b. Design Harder Edge Cases / Adversarial Testing
- Create intentionally difficult scenarios to find system limits:
  - Queries mixing multiple benefit programs (e.g., CPP + EI in same situation)
  - Queries about very old or superseded legal tests
  - Queries with deliberate misinformation or contradictory facts
  - Queries in very informal/broken English or with typos
  - Queries about edge jurisdictional issues (provincial vs. federal)
  - Queries where no relevant SST decision exists
- Document where the system fails and why — this is what the instructor wants to see

#### 1c. Address Whether AI-Generated Test Cases Are Valid
- The instructor specifically questioned this. Add a reflective discussion on:
  - Why using AI to generate test queries for an AI system is circular
  - What biases this introduces (LLM may generate "clean" queries that work well)
  - Supplement with real queries from legal clinics or SRL forums if possible

### 2. Address Case Authority & Recency Problem
**Source:** Instructor (primary concern), Peers (Sitara, Amelia)

The system currently treats all cases as equally authoritative. This is a significant legal concern.

#### 2a. Add Discussion of the Problem
- In the reflective section, discuss:
  - Old decisions may reflect outdated law
  - Decisions can be overturned on appeal
  - Legislative amendments can supersede tribunal interpretations
  - The risk this poses to SRLs who may rely on outdated precedent

#### 2b. Implement Recency Signals in Results
- Add a visual indicator for decision age (e.g., flag decisions older than 5 years)
- Consider adding a recency weight/boost to the scoring pipeline (optional — at minimum discuss the tradeoff)
- Add a disclaimer on older decisions noting they may not reflect current law
- Display the `document_date_en` more prominently and sort or label by recency

#### 2c. Explore Mechanisms to Adjust Results
- Discuss (and optionally implement) approaches:
  - Date-based filtering or boosting in the reranker stage
  - Allow users to filter by date range
  - Flag if a returned decision references legislation that has since been amended

### 3. Expand Reflective Component Significantly
**Source:** Instructor (primary concern), Peers (Clifford, Sitara, Emma)

This was called "the main weakness." Lowest peer scores were Research (3.17) and Reflection (3.17).

#### 3a. Engage with Academic Literature
- Cite relevant scholarship on:
  - RAG systems in legal contexts
  - Access to justice and technology (e.g., Susskind, Cabral et al.)
  - Risks of AI in legal information systems
  - Self-represented litigants and information needs
- Ground claims about SRL struggles with actual studies/reports, not assumptions

#### 3b. Clearly Articulate AI's Role vs. Human Contribution
- The instructor wants to know: what was YOUR intellectual contribution?
- For each stage of the project, document:
  - What was designed/decided by the team
  - What code was AI-generated vs. human-written
  - What architectural decisions were made by humans
  - How AI output was evaluated and refined

#### 3c. Expand Ethical Discussion
- Go beyond generic AI ethics to address **project-specific** risks:
  - Outdated case information for vulnerable users (Sitara's point)
  - Hallucination risk in Stage 3 summaries
  - Whether similarity scores give false confidence
  - What happens when the system is wrong and an SRL relies on it

### 4. Investigate High Similarity Scores
**Source:** Instructor

- Analyze the distribution of reranker scores across multiple queries
- Are scores clustering near 1.0? If so, investigate:
  - Is the reranker model poorly calibrated for this domain?
  - Is the section-aware packing causing overly favorable input construction?
  - Are the bi-encoder pre-filtering candidates too aggressively?
- Document findings and discuss what the scores actually mean for users

---

## MINOR ITEMS (Lower Priority, But Worth Addressing)

### 5. Add Paragraph-Level Citations to Case Cards
**Source:** Peer (Chong Tan)

- Update the Stage 3 generation prompt to include paragraph numbers from the original decision
- This helps users verify AI summaries against source text
- Reduces hallucination risk by grounding claims

### 6. Explain Model Choices (Why Qwen3?)
**Source:** Peer (Elad Dekel)

- Add a brief section explaining:
  - Why Qwen3 over OpenAI/Gemini/other models
  - Cost, performance, and open-source considerations
  - Why three different Qwen3 models for different stages

### 7. Add Legal Term Definitions for SRLs
**Source:** Peer (Amelia Cox)

- Consider adding a glossary or inline definitions for legal jargon in case cards
- Could be a "next steps" discussion item rather than a full implementation

### 8. Improve Notebook Accessibility
**Source:** Peers (Alvina, Zakir, Emma)

- Ensure notebook renders properly in standard Jupyter
- Use more accessible language in the methodology introduction
- Add clearer sub-headers mapping to the 3-stage pipeline

### 9. More Example Queries & Outputs
**Source:** Peers (Olivia, Gurnoor)

- Add more diverse example searches showing different query types
- Show outputs for different benefit programs
- This doubles as additional validation evidence

### 10. Consider "Legal Landscape" Mode
**Source:** Peer (Damien Tang)

- Interesting idea: return cases on both sides of an issue
- Worth discussing in "future work" section even if not implemented

---

## Suggested Implementation Order

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 1 | Expand reflective component (#3) | Medium | High (instructor's "main weakness") |
| 2 | Harder edge cases & adversarial testing (#1b) | Medium | High (instructor concern) |
| 3 | Case authority & recency discussion + UI signals (#2) | Medium | High (instructor concern) |
| 4 | CanLII baseline comparison (#1a) | Medium | High (instructor concern) |
| 5 | Investigate high similarity scores (#4) | Low | Medium (instructor concern) |
| 6 | AI test case validity discussion (#1c) | Low | Medium (instructor concern) |
| 7 | Paragraph citations in case cards (#5) | Low | Medium |
| 8 | Model choice explanation (#6) | Low | Low |
| 9 | More example queries (#9) | Low | Low |
| 10 | Notebook accessibility (#8) | Low | Low |

---

## Key Takeaway

The **reflective component** and **validation rigor** are the two areas that will move the grade most. The instructor wants to see:
1. Evidence that you understand the system's limitations (not just that it works)
2. Academic grounding for your claims
3. Honest disclosure of AI vs. human contribution
4. A more robust testing methodology that stress-tests the system

The peer feedback largely aligns with the instructor's but also highlights some useful UX improvements (citations, legal term definitions, more examples) that would strengthen the project holistically.
