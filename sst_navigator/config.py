"""
Central configuration for the SST Decision Navigator.
"""

# --- Model identifiers ---
EMBEDDING_MODEL = "mlx-community/Qwen3-Embedding-4B-mxfp8"
RERANKER_MODEL = "mlx-community/Qwen3-Reranker-4B-mxfp8"
# A small, fast MLX model for case-card generation
GENERATION_MODEL = "mlx-community/Qwen3-4B-4bit"

# --- Pipeline parameters ---
STAGE1_TOP_K = 20          # Candidates from semantic search
STAGE2_TOP_K = 3           # Final results after reranking
SNIPPET_LENGTH = 500       # Characters for the preview snippet

# --- Embedding parameters ---
EMBEDDING_INSTRUCTION = (
    "Represent this query for retrieving similar Canadian "
    "administrative legal tribunal decisions"
)
EMBEDDING_MAX_TOKENS = 8192  # Max tokens per text chunk for embedding
EMBEDDING_CACHE_DIR = ".cache/embeddings"
EMBEDDING_BATCH_SIZE_DEV = 8
EMBEDDING_BATCH_SIZE_PROD = 2

# --- Reranker parameters ---
RERANKER_INSTRUCTION = (
    "Given a legal query describing an appellant's situation, "
    "retrieve relevant Social Security Tribunal of Canada decisions"
)
RERANKER_MAX_TOKENS = 8192

# --- Generation parameters ---
GENERATION_MAX_TOKENS = 512

# --- Data ---
DEV_ROW_LIMIT = 500  # Row limit for development/testing
