"""Delta-update SST embedding cache with Scatter-Gather checkpointing.

For small weekly updates (a few dozen cases), the script embeds and appends
directly.  For large cold-start runs (thousands of cases), it uses a
Scatter-Gather strategy: each chunk is saved to its own file, then all
chunks are merged once at the end.  This keeps peak memory low and makes
the run fully resumable.

Embeddings are produced via the DeepInfra OpenAI-compatible API using
Qwen/Qwen3-Embedding-8B-batch (50 % cheaper than the real-time variant,
intended for bulk, non-latency-sensitive workloads).  Set DEEPINFRA_API_KEY
in your environment before running.

Pass --use-local to fall back to the local MLX model (useful on Apple Silicon
without a DeepInfra account, or for offline development).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sst_navigator import config
from sst_navigator.data_loader import load_sst_decisions

logger = logging.getLogger(__name__)

# Chunks smaller than this threshold are embedded and merged directly.
# Larger runs use Scatter-Gather.
SCATTER_GATHER_THRESHOLD = 1000
CHUNK_SIZE = 500

# API batch size: DeepInfra accepts up to 2048 inputs per request; 200 is
# a conservative default that keeps request payloads manageable.
API_BATCH_SIZE = 200

# Within each chunk, save partial progress every CHECKPOINT_EVERY texts.
CHECKPOINT_EVERY = API_BATCH_SIZE * 5  # 1 000 texts


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_texts_api(
    texts: list[str],
    model: str,
    batch_size: int = API_BATCH_SIZE,
    progress_desc: str = "Embedding",
) -> np.ndarray:
    """Embed *texts* via DeepInfra OpenAI-compatible API, return float32 array."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY is not set. "
            "Export your key or pass --use-local to use the MLX model."
        )

    client = OpenAI(api_key=api_key, base_url=config.DEEPINFRA_BASE_URL)
    parts: list[np.ndarray] = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=progress_desc,
        unit="batch",
    ):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model,
            encoding_format="float",
        )
        batch_vecs = np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )
        parts.append(batch_vecs)

    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _embed_texts_mlx(
    texts: list[str],
    batch_size: int = 4,
) -> np.ndarray:
    """Embed *texts* via the local MLX model (fallback path)."""
    from sst_navigator.embedder import SemanticSearcher

    searcher = SemanticSearcher(backend="mlx")
    searcher.load_model()
    try:
        return searcher.embed_texts(
            texts,
            batch_size=batch_size,
            max_tokens=config.EMBEDDING_MAX_TOKENS,
        )
    finally:
        searcher.unload_model()


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _load_metadata(metadata_path: Path) -> list[str]:
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(url) for url in raw]
    if isinstance(raw, dict):
        urls = raw.get("url_en", [])
        if isinstance(urls, list):
            return [str(url) for url in urls]
    raise ValueError(
        "metadata.json must be a list of URLs or a dict with a 'url_en' list."
    )


def _write_metadata(metadata_path: Path, urls: list[str]) -> None:
    metadata_path.write_text(
        json.dumps({"url_en": urls}, indent=2),
        encoding="utf-8",
    )


def _load_existing_cache(
    cache_dir: Path,
    emb_path: Path,
    meta_path: Path,
    repo_id: str,
    repo_type: str,
) -> tuple[np.ndarray | None, list[str]]:
    """Return (embeddings_array | None, url_list) from local or remote cache."""
    if emb_path.exists() and meta_path.exists():
        logger.info("Local cache found — resuming from disk.")
        return (
            np.load(emb_path, allow_pickle=False),
            _load_metadata(meta_path),
        )

    logger.info("No local cache. Downloading from %s …", repo_id)
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=config.EMBEDDING_CACHE_FILE,
            repo_type=repo_type,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        hf_hub_download(
            repo_id=repo_id,
            filename=config.EMBEDDING_METADATA_FILE,
            repo_type=repo_type,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        return (
            np.load(emb_path, allow_pickle=False),
            _load_metadata(meta_path),
        )
    except Exception:
        logger.warning("Could not download cache (likely empty repo). Starting fresh.")
        return None, []


# ---------------------------------------------------------------------------
# Phase 1 — Scatter: embed chunks and save each to its own file
# ---------------------------------------------------------------------------

def _embed_chunk_resumable(
    texts: list[str],
    use_local: bool,
    checkpoint_every: int,
    partial_emb_path: Path,
    partial_progress_path: Path,
) -> np.ndarray:
    """Embed *texts* with periodic intra-chunk saves.

    Resumes from *partial_emb_path* / *partial_progress_path* if they exist.
    Returns the full (len(texts), embed_dim) array.
    """
    if partial_emb_path.exists() and partial_progress_path.exists():
        completed = json.loads(
            partial_progress_path.read_text(encoding="utf-8"),
        )["completed"]
        prev_emb: np.ndarray | None = np.load(partial_emb_path, allow_pickle=False)
        logger.info("  Resuming chunk at text %d / %d.", completed, len(texts))
    else:
        completed = 0
        prev_emb = None

    remaining = texts[completed:]
    if not remaining:
        assert prev_emb is not None
        return prev_emb

    new_parts: list[np.ndarray] = []

    for i in range(0, len(remaining), checkpoint_every):
        sub = remaining[i : i + checkpoint_every]

        if use_local:
            sub_emb = _embed_texts_mlx(sub)
        else:
            sub_emb = _embed_texts_api(
                sub,
                model=config.DEEPINFRA_EMBEDDING_BATCH_MODEL,
                progress_desc="  Checkpoint",
            )

        new_parts.append(sub_emb)
        completed += len(sub)

        new_block = (
            np.concatenate(new_parts, axis=0) if len(new_parts) > 1 else new_parts[0]
        )
        checkpoint_emb = (
            np.concatenate([prev_emb, new_block], axis=0)
            if prev_emb is not None
            else new_block
        )

        np.save(partial_emb_path, checkpoint_emb)
        partial_progress_path.write_text(
            json.dumps({"completed": completed}), encoding="utf-8",
        )

        del sub_emb
        gc.collect()

    return checkpoint_emb  # type: ignore[possibly-undefined]


def _scatter(
    texts: list[str],
    urls: list[str],
    chunk_dir: Path,
    chunk_size: int,
    use_local: bool,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> int:
    """Embed *texts* in chunks, writing each to *chunk_dir*.

    Returns the number of chunks produced.
    """
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Skip chunks that are already complete
    first_needed = 0
    for idx in range(total_chunks):
        emb_file = chunk_dir / f"chunk_{idx:04d}.npy"
        meta_file = chunk_dir / f"chunk_{idx:04d}.json"
        if emb_file.exists() and meta_file.exists():
            first_needed = idx + 1
        else:
            break

    if first_needed >= total_chunks:
        logger.info("All %d chunks already exist — skipping scatter phase.", total_chunks)
        return total_chunks

    if first_needed > 0:
        logger.info("Resuming scatter from chunk %d / %d.", first_needed, total_chunks)

    for idx in tqdm(
        range(first_needed, total_chunks),
        initial=first_needed,
        total=total_chunks,
        desc="Scatter",
    ):
        start = idx * chunk_size
        end = min(start + chunk_size, len(texts))
        chunk_texts = texts[start:end]
        chunk_urls = urls[start:end]

        partial_emb = chunk_dir / f"chunk_{idx:04d}_partial.npy"
        partial_prog = chunk_dir / f"chunk_{idx:04d}_partial.json"

        chunk_emb = _embed_chunk_resumable(
            chunk_texts,
            use_local=use_local,
            checkpoint_every=checkpoint_every,
            partial_emb_path=partial_emb,
            partial_progress_path=partial_prog,
        )

        # Atomic promote to final chunk file
        emb_file = chunk_dir / f"chunk_{idx:04d}.npy"
        meta_file = chunk_dir / f"chunk_{idx:04d}.json"
        tmp_emb = chunk_dir / f"chunk_{idx:04d}_tmp.npy"
        tmp_meta = meta_file.with_suffix(".json.tmp")

        np.save(tmp_emb, chunk_emb)
        tmp_meta.write_text(json.dumps(chunk_urls), encoding="utf-8")
        tmp_emb.rename(emb_file)
        tmp_meta.rename(meta_file)

        partial_emb.unlink(missing_ok=True)
        partial_prog.unlink(missing_ok=True)

        del chunk_emb
        gc.collect()

    return total_chunks


# ---------------------------------------------------------------------------
# Phase 2 — Gather: merge all chunk files into the master cache
# ---------------------------------------------------------------------------

def _gather(
    chunk_dir: Path,
    num_chunks: int,
    existing_embeddings: np.ndarray | None,
    existing_urls: list[str],
    emb_path: Path,
    meta_path: Path,
) -> np.ndarray:
    """Load all chunk files, concatenate with existing data, write master cache."""
    logger.info("Gather phase: merging %d chunk(s) into master cache …", num_chunks)

    parts: list[np.ndarray] = []
    all_urls: list[str] = list(existing_urls)

    if existing_embeddings is not None:
        parts.append(existing_embeddings)

    for idx in range(num_chunks):
        emb_file = chunk_dir / f"chunk_{idx:04d}.npy"
        meta_file = chunk_dir / f"chunk_{idx:04d}.json"
        parts.append(np.load(emb_file, allow_pickle=False))
        all_urls.extend(json.loads(meta_file.read_text(encoding="utf-8")))

    merged = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    np.save(emb_path, merged)
    _write_metadata(meta_path, all_urls)

    shutil.rmtree(chunk_dir)
    logger.info("Gather complete — merged shape %s. Chunks cleaned up.", merged.shape)
    return merged


# ---------------------------------------------------------------------------
# Direct delta-update path (small batches)
# ---------------------------------------------------------------------------

def _direct_update(
    texts: list[str],
    urls: list[str],
    existing_embeddings: np.ndarray | None,
    existing_urls: list[str],
    emb_path: Path,
    meta_path: Path,
    use_local: bool,
) -> np.ndarray:
    """Embed a small batch of new cases and merge directly."""
    if use_local:
        new_embeddings = _embed_texts_mlx(texts)
    else:
        new_embeddings = _embed_texts_api(
            texts,
            model=config.DEEPINFRA_EMBEDDING_BATCH_MODEL,
            progress_desc="Embedding",
        )

    merged = (
        np.concatenate([existing_embeddings, new_embeddings], axis=0)
        if existing_embeddings is not None
        else new_embeddings
    )
    merged_urls = existing_urls + urls

    np.save(emb_path, merged)
    _write_metadata(meta_path, merged_urls)
    logger.info("Direct update complete — shape %s.", merged.shape)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed new SST cases and update the remote cache."
    )
    parser.add_argument("--repo-id", default=config.EMBEDDING_CACHE_REPO_ID)
    parser.add_argument("--repo-type", default=config.EMBEDDING_CACHE_REPO_TYPE)
    parser.add_argument("--cache-dir", default=config.EMBEDDING_CACHE_DIR)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Cases per scatter chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use the local MLX model instead of the DeepInfra API.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.use_local and not os.environ.get("DEEPINFRA_API_KEY"):
        logger.error(
            "DEEPINFRA_API_KEY is not set. "
            "Export your key or pass --use-local to use the local MLX model."
        )
        sys.exit(1)

    embedding_source = "local MLX" if args.use_local else (
        f"DeepInfra ({config.DEEPINFRA_EMBEDDING_BATCH_MODEL})"
    )
    logger.info("Embedding source: %s", embedding_source)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / config.EMBEDDING_CACHE_FILE
    meta_path = cache_dir / config.EMBEDDING_METADATA_FILE
    chunk_dir = cache_dir / "tmp_chunks"

    # -- Load existing cache (local-first, then remote) --------------------
    existing_embeddings, existing_urls = _load_existing_cache(
        cache_dir, emb_path, meta_path, args.repo_id, args.repo_type,
    )
    # Deduplicate existing cache
    if existing_embeddings is not None:
        n_unique = len(set(existing_urls))
        if n_unique < len(existing_urls):
            n_dups = len(existing_urls) - n_unique
            logger.warning(
                "Existing cache has %d duplicate URLs — deduplicating.", n_dups,
            )
            seen: set[str] = set()
            keep: list[int] = []
            for i, url in enumerate(existing_urls):
                if url not in seen:
                    seen.add(url)
                    keep.append(i)
            existing_embeddings = existing_embeddings[np.array(keep)]
            existing_urls = [existing_urls[i] for i in keep]
            np.save(emb_path, existing_embeddings)
            _write_metadata(meta_path, existing_urls)

    existing_url_set = set(existing_urls)
    logger.info("Existing cache: %d embeddings.", len(existing_urls))

    # -- Find new decisions ------------------------------------------------
    logger.info("Loading latest SST decisions dataset …")
    df = load_sst_decisions(max_rows=None)

    new_df = df[~df["url_en"].isin(existing_url_set)].copy()
    if new_df.empty:
        logger.info("No new decisions found. Cache is already up to date.")
        return

    new_texts = new_df["unofficial_text_en"].tolist()
    new_urls = new_df["url_en"].astype(str).tolist()
    logger.info("Found %d new decisions to embed.", len(new_df))

    # Sort by text length to group similarly-sized documents; reduces padding
    # waste when using the local MLX path.
    sorted_pairs = sorted(zip(new_texts, new_urls), key=lambda p: len(p[0]))
    new_texts = [t for t, _ in sorted_pairs]
    new_urls = [u for _, u in sorted_pairs]

    # -- Choose strategy ---------------------------------------------------
    if len(new_texts) >= SCATTER_GATHER_THRESHOLD:
        logger.info(
            "Large batch (%d cases) — using Scatter-Gather with chunk_size=%d.",
            len(new_texts),
            args.chunk_size,
        )
        num_chunks = _scatter(
            new_texts, new_urls, chunk_dir, args.chunk_size, args.use_local,
        )
        merged = _gather(
            chunk_dir, num_chunks,
            existing_embeddings, existing_urls,
            emb_path, meta_path,
        )
    else:
        logger.info("Small batch (%d cases) — using direct update.", len(new_texts))
        merged = _direct_update(
            new_texts, new_urls,
            existing_embeddings, existing_urls,
            emb_path, meta_path,
            use_local=args.use_local,
        )

    # -- Upload to Hugging Face --------------------------------------------
    logger.info("Uploading updated cache to %s …", args.repo_id)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(emb_path),
        path_in_repo=config.EMBEDDING_CACHE_FILE,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )
    api.upload_file(
        path_or_fileobj=str(meta_path),
        path_in_repo=config.EMBEDDING_METADATA_FILE,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )

    logger.info(
        "Update complete. Total shape: %s (added %d rows).",
        merged.shape,
        len(new_df),
    )


if __name__ == "__main__":
    main()
