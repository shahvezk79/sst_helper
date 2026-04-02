# How It Works (One-Page Visual)

Use this diagram in the notebook/report to explain the pipeline in plain language.

```mermaid
flowchart LR
    A[1) User Query\nA person asks a benefits-law question in plain language.] -->
    B[2) Section-Aware Retrieval\nThe system finds relevant decisions and prioritizes useful sections like Issue/Analysis.] -->
    C[3) Reranking\nA stronger relevance model reorders candidates so the best matches rise to the top.] -->
    D[4) Summarization with Citations\nThe assistant drafts a concise answer and cites source decisions for traceability.] -->
    E[5) Risk Labels + User View\nEach result is shown with confidence/risk cues and plain-language caveats.]
```

## Caption guidance (plain language)

- **User Query**: The user asks in everyday words, not legal jargon.
- **Section-Aware Retrieval**: The system searches by meaning and focuses on reasoning-heavy sections.
- **Reranking**: Candidate decisions are rescored so likely useful ones appear first.
- **Summarization with Citations**: Output is brief and source-grounded; readers can verify claims.
- **Risk Labels + User View**: Users see caution labels and are reminded this is guidance, not legal advice.

## Slide/notebook layout suggestion

- Put the diagram at the top of the pipeline section.
- Place a single-sentence caption under each box.
- Keep explanation to one page so non-technical readers can follow quickly.
