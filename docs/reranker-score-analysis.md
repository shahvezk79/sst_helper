# Reranker Score Analysis: Are High Similarity Scores Meaningful or Inflated?

## Summary

The high reranker scores (often 90%+) observed in SST Navigator are **structurally expected** given the scoring mechanism and pipeline design — but they are **not inflated in a way that undermines ranking quality**. The scores are meaningful for *relative ordering* of results, though their *absolute values* should not be interpreted as calibrated probabilities of relevance.

## How the Reranker Score Is Computed

### Model
- **Qwen3-Reranker-8B** (cross-encoder), run locally via MLX or through DeepInfra API.

### Scoring Mechanism (MLX path — `reranker.py:92-113`)

1. The model receives a prompt asking: *"Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be 'yes' or 'no'."*
2. A forward pass produces logits over the full vocabulary at the final token position.
3. Only the logits for the **"yes"** and **"no"** tokens are extracted.
4. These two logits are passed through **softmax** → `P(yes)` becomes the score.

```python
stacked = mx.concatenate([no_logit, yes_logit], axis=0)  # (2,)
probs = mx.softmax(stacked)
score = probs[1].item()  # P(yes) ∈ [0, 1]
```

## Why Scores Cluster High

### 1. Two-token softmax concentrates probability mass

The softmax is computed over only **2 values** (yes/no), not the full vocabulary. This means:
- A modest logit difference of ~2.0 translates to `softmax([0, 2]) ≈ [0.12, 0.88]` → **88% score**
- A logit difference of ~3.0 → `softmax([0, 3]) ≈ [0.05, 0.95]` → **95% score**
- A logit difference of ~4.0 → `softmax([0, 4]) ≈ [0.02, 0.98]` → **98% score**

With only 2 classes, it takes very little logit separation for the softmax to produce extreme probabilities. In a full-vocabulary softmax (32K+ tokens), the same logit difference would produce much more moderate scores.

### 2. Pre-filtering by Stage 1 removes obviously irrelevant documents

By the time documents reach the reranker, they've already passed **Stage 1 semantic search** (cosine similarity on Qwen3-Embedding-8B) and represent the top 40 (or 20 in fast mode) most similar documents out of the entire corpus. These are all *plausibly relevant* — the reranker is not seeing random documents, it's distinguishing between "somewhat relevant" and "highly relevant".

### 3. Domain-specific instruction biases toward "yes"

The reranker instruction (`config.py:39-42`) is:
> "Given a legal query describing an appellant's situation, retrieve relevant Social Security Tribunal of Canada decisions"

This is a **broad retrieval task** — SST decisions share substantial structural and thematic similarity (same tribunal, same legal framework, similar fact patterns). The instruction doesn't ask for narrow precision (e.g., "find cases with identical legal issues and outcomes"), so the model is inclined to judge many pre-filtered candidates as relevant.

### 4. Section-aware packing emphasizes the most relevant content

The `pack_for_reranker()` function (`section_parser.py:171-217`) intelligently selects the **most legally salient sections** (Analysis > Issue > Conclusion) within the character budget. By stripping boilerplate and prioritizing substantive content, the packed text is *more likely* to appear relevant than the raw full document would be.

## Are the Scores Still Meaningful?

**Yes — for ranking purposes.** Even when all scores are high, the **relative ordering** remains informative:

- A document scoring 0.97 vs. 0.92 reflects a genuine difference in the model's confidence — the logit gap is meaningful even if both map to "high" percentages.
- The cross-encoder architecture (query-document joint attention) captures fine-grained semantic interactions that the Stage 1 bi-encoder misses. This is why reranking frequently reshuffles the Stage 1 ordering.

**No — for absolute calibration.** A score of 0.95 should not be read as "95% probability this document is relevant." The 2-token softmax, pre-filtering, and domain-specific prompt all compress the score distribution upward.

## Recommendations

### Do Not Add a Minimum Score Threshold

Given the compressed score distribution, a fixed threshold (e.g., "discard results below 0.80") would be fragile:
- For specific queries, all 5 results might legitimately score above 0.95.
- For vague queries, even the best match might be 0.75.
- The current top-K selection (return best 5) is more robust than threshold-based filtering.

### Consider Log-Scale or Rank-Based Display

If users are confused by uniformly high percentages, consider:
1. **Displaying rank only** (already shown as #1, #2, etc.)
2. **Relative scoring** — normalize scores within each query's result set (e.g., best = 100%, others scaled relative to it)
3. **Score delta display** — show the gap between each result and the top result

### Add Score Distribution Logging

To build empirical evidence, add logging that captures the full score distribution per query. This helps calibrate expectations and detect anomalies.

## Conclusion

The high scores are a natural consequence of:
1. **Binary softmax** on 2 tokens (compresses toward extremes)
2. **Pre-filtered candidates** (Stage 1 already removes junk)
3. **Domain similarity** (SST decisions are structurally homogeneous)
4. **Content packing** (reranker sees only the most relevant sections)

The reranker is working correctly — it's doing its job of **reordering** pre-filtered candidates by fine-grained relevance. The scores just happen to cluster high because the model is choosing between "yes" and "no" for documents that are all plausibly relevant. The ranking discrimination is where the value lies, not the absolute score values.
