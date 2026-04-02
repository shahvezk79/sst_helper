# Submission Pack: Packaging & Communication Polish

This file is a one-stop checklist for preparing a grader-friendly submission bundle that can be reviewed **without running the notebook**.

## A. Goal

Make the technical work easy to read, grade, and trust in under 10 minutes.

## B. Multi-format deliverable strategy

Prepare all of the following before submission:

1. **Primary notebook**: `sst_helper_notebook.ipynb`
2. **Static HTML export**: `deliverables/sst_helper_notebook.html`
3. **PDF export**: `deliverables/sst_helper_notebook.pdf`
4. **README navigation map**: quick links to key sections and appendices

This avoids reviewer friction when notebook execution is unavailable.

## C. Suggested report section order (high readability)

Use this order consistently in the notebook and/or report:

1. Problem & A2J context (plain language)
2. Pipeline overview (diagram)
3. Retrieval methodology
4. Trust/safety mechanisms (authority + explainability)
5. Validation summary (team-owned)
6. Limitations and responsible-use framing
7. Appendix examples

## D. Export bundle plan

Use the script below to generate HTML and PDF exports from the notebook:

```bash
bash scripts/export_notebook_bundle.sh
```

The script creates a `deliverables/` folder and exports:
- `deliverables/sst_helper_notebook.html`
- `deliverables/sst_helper_notebook.pdf` (if PDF tooling is installed)

## E. Deliverables included in this repo

- `README.md` → **Start Here** navigation map.
- `docs/how_it_works_diagram.md` → one-page 5-box visual + plain-language captions.
- `docs/worked_examples_appendix.md` → three additional mini-cases.
- `scripts/export_notebook_bundle.sh` → export automation for notebook packaging.

## F. A-level readiness checklist

A submission is “A-level ready” when:

- A grader can understand the project flow in 10 minutes.
- A non-technical reader can follow each stage using plain language.
- Trust/safety framing appears deliberate (integrated) rather than bolted on.
- The bundle opens in at least one static format (HTML/PDF) without notebook execution.
