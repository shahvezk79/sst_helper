# Performance tuning notes for SST Navigator

## Current hotspots

1. **Index build time (embeddings):** dominated by full-corpus embedding with `Qwen3-Embedding-4B`.
2. **Per-query latency:** each search currently runs three heavy stages in sequence:
   - query embedding,
   - cross-encoder reranking over top-k candidates,
   - case-card generation.
3. **Model swap overhead:** pipeline intentionally keeps only one heavy model in memory and unloads/reloads between stages.

## Highest-leverage optimizations before downsizing models

### 1) Decouple retrieval latency from generation latency
- Make generation optional at search time and run it only when the user requests a summary.
- Return ranked results immediately; generate case card asynchronously or on demand.
- This preserves retrieval quality while reducing time-to-first-result.

### 2) Reduce reranker work first
- Lower Stage-1 candidate count from 20 to a smaller value (e.g., 8–12) after measuring recall impact.
- Truncate reranker document inputs more aggressively than `RERANKER_MAX_TOKENS * 4` if quality holds.
- Batch reranker scoring requests instead of scoring one candidate at a time.

### 3) Reduce embedding work before changing embedding model
- Split long decisions into semantic chunks and embed chunks once, instead of sending entire decisions up to max token limits.
- Use smaller embedding `max_tokens` for both document and query embedding where feasible.
- Keep and reuse embedding cache (already present), and prebuild cache offline for production deploys.

### 4) Minimize model load/unload churn
- If memory allows, keep embedder + reranker resident together; unload only generator.
- If memory is constrained, choose one “hot” model to keep resident based on expected workload.

### 5) Add measurement before architecture/model changes
Track and log these separately per query:
- query embedding time,
- reranker time,
- generation time,
- model switch/load time.

This makes it clear whether generation model size is really the bottleneck.

## On the “do we need 4B generation?” question

**Not necessarily.** Generation quality requirements (readability and faithfulness) are usually different from reranking precision.

A practical approach:
1. Keep current reranker.
2. A/B test case-card quality with a smaller generator.
3. Compare latency and quality side-by-side.

If case-card quality remains acceptable, a smaller generator is a straightforward latency win.

## On downsizing the embedding model

Do this **after** measuring the optimizations above. A smaller embedding model can speed up indexing/query embedding but may reduce recall; test with a held-out set of representative user queries.
