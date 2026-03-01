"""
Central configuration for the SST Decision Navigator.
"""

# --- Model identifiers ---
# Local MLX fallback models (Apple Silicon)
EMBEDDING_MODEL = "mlx-community/Qwen3-Embedding-8B-mxfp8"
RERANKER_MODEL = "mlx-community/Qwen3-Reranker-8B-mxfp8"
# A small, fast MLX model for case-card generation
GENERATION_MODEL = "mlx-community/Qwen3-4B-4bit"

# --- Pipeline parameters ---
STAGE1_TOP_K = 40          # Candidates from semantic search
STAGE2_TOP_K = 5           # Final results after reranking
SNIPPET_LENGTH = 500       # Characters for the preview snippet

# Fast mode keeps quality reasonably high while reducing latency
FAST_STAGE1_TOP_K = 20
FAST_STAGE2_TOP_K = 3

# --- Embedding parameters ---
EMBEDDING_INSTRUCTION = (
    "Represent this query for retrieving similar Canadian "
    "administrative legal tribunal decisions"
)
EMBEDDING_MAX_TOKENS = 8192  # Max tokens per text chunk for embedding
FAST_EMBEDDING_MAX_TOKENS = 4096
EMBEDDING_CACHE_DIR = ".cache/embeddings"
EMBEDDING_CACHE_REPO_ID = "mystic63/sst-embeddings-cache"
EMBEDDING_CACHE_REPO_TYPE = "dataset"
# Filename reflects the model tier so 4B and 8B caches can coexist in the repo
EMBEDDING_CACHE_FILE = "sst_embeddings_qwen3_8b.npy"
EMBEDDING_METADATA_FILE = "metadata.json"
EMBEDDING_BATCH_SIZE_DEV = 8
EMBEDDING_BATCH_SIZE_PROD = 2


# --- Reranker parameters ---
RERANKER_INSTRUCTION = (
    "Given a legal query describing an appellant's situation, "
    "retrieve relevant Social Security Tribunal of Canada decisions"
)
RERANKER_MAX_TOKENS = 8192
FAST_RERANKER_MAX_TOKENS = 2048

# --- Generation parameters ---
GENERATION_MAX_TOKENS = 512
FAST_GENERATION_MAX_TOKENS = 256
GENERATION_MAX_CHARS = 24000
FAST_GENERATION_MAX_CHARS = 12000

# --- DeepInfra cloud settings ---
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# Embedding — OpenAI-compatible endpoint
# Use the batch variant for index builds (50% cheaper, async-friendly).
# Use the standard variant for real-time query encoding.
DEEPINFRA_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEEPINFRA_EMBEDDING_BATCH_MODEL = "Qwen/Qwen3-Embedding-8B-batch"

# Reranker — native DeepInfra inference endpoint
DEEPINFRA_RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
DEEPINFRA_RERANKER_ENDPOINT = (
    "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B"
)

# Generation
DEEPINFRA_GENERATION_MODEL = "Qwen/Qwen3-14B"

# --- Data ---
DEV_ROW_LIMIT = 500  # Row limit for development/testing
