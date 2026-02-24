"""Delta-update SST embedding cache with Scatter-Gather checkpointing.

For small weekly updates (a few dozen cases), the script embeds and appends
directly.  For large cold-start runs (thousands of cases), it uses a
Scatter-Gather strategy: each chunk is saved to its own file, then all
chunks are merged once at the end.  This keeps peak memory low during
GPU embedding and makes the run fully resumable.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sst_navigator import config
from sst_navigator.data_loader import load_sst_decisions

logger = logging.getLogger(__name__)

# Chunks smaller than this threshold are embedded and merged directly
# (the original delta-update path).  Larger runs use Scatter-Gather.
SCATTER_GATHER_THRESHOLD = 1000
CHUNK_SIZE = 500

# Tuned for MacBook Pro M5 (16 GB unified memory).  The 4B embedding
# model consumes ~4 GB; each text at 8192 tokens needs ~290 MB of KV
# cache across 36 layers, so batch_size=4 keeps peak GPU memory ≈ 5.6 GB
# and leaves headroom for the OS and other processes.
BATCH_SIZE = 1

# Within each chunk, save partial progress every CHECKPOINT_EVERY texts
# so a mid-chunk crash only loses one interval instead of the whole chunk.
CHECKPOINT_EVERY = BATCH_SIZE * 25  # 25 texts


# ---------------------------------------------------------------------------
# Helpers
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
    searcher,
    texts: list[str],
    batch_size: int,
    checkpoint_every: int,
    partial_emb_path: Path,
    partial_progress_path: Path,
) -> np.ndarray:
    """Embed *texts* with periodic intra-chunk saves.

    If *partial_emb_path* and *partial_progress_path* already exist from a
    previous interrupted run, the already-embedded prefix is loaded and
    embedding resumes from where it stopped.

    Returns the full (len(texts), embed_dim) array.
    """
    # Resume from partial progress if available
    if partial_emb_path.exists() and partial_progress_path.exists():
        completed = json.loads(
            partial_progress_path.read_text(encoding="utf-8"),
        )["completed"]
        all_emb: np.ndarray | None = np.load(partial_emb_path, allow_pickle=False)
        logger.info("  Resuming chunk at text %d / %d.", completed, len(texts))
    else:
        completed = 0
        all_emb = None

    remaining = texts[completed:]
    if not remaining:
        assert all_emb is not None
        return all_emb

    for i in range(0, len(remaining), checkpoint_every):
        sub = remaining[i : i + checkpoint_every]
        sub_emb = searcher.embed_texts(
            sub,
            batch_size=batch_size,
            max_tokens=config.EMBEDDING_MAX_TOKENS,
        )

        all_emb = (
            np.concatenate([all_emb, sub_emb], axis=0)
            if all_emb is not None
            else sub_emb
        )
        completed += len(sub)

        # Persist partial state so a crash only loses one interval
        np.save(partial_emb_path, all_emb)
        partial_progress_path.write_text(
            json.dumps({"completed": completed}), encoding="utf-8",
        )

        del sub_emb
        gc.collect()
        mx.clear_cache()

    return all_emb  # type: ignore[return-value]


def _scatter(
    texts: list[str],
    urls: list[str],
    chunk_dir: Path,
    chunk_size: int,
    batch_size: int = BATCH_SIZE,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> int:
    """Embed *texts* in chunks, writing each to *chunk_dir*.

    Returns the number of chunks produced.  Completed chunk files are
    skipped entirely, and partially-completed chunks resume from the
    last intra-chunk checkpoint.
    """
    from sst_navigator.embedder import SemanticSearcher

    total_chunks = (len(texts) + chunk_size - 1) // chunk_size
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Determine which chunks still need processing
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

    searcher = SemanticSearcher()
    searcher.load_model()
    try:
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

            # Paths for intra-chunk partial progress
            partial_emb = chunk_dir / f"chunk_{idx:04d}_partial.npy"
            partial_prog = chunk_dir / f"chunk_{idx:04d}_partial.json"

            chunk_emb = _embed_chunk_resumable(
                searcher,
                chunk_texts,
                batch_size=batch_size,
                checkpoint_every=checkpoint_every,
                partial_emb_path=partial_emb,
                partial_progress_path=partial_prog,
            )

            # Promote to final chunk file (atomic rename)
            emb_file = chunk_dir / f"chunk_{idx:04d}.npy"
            meta_file = chunk_dir / f"chunk_{idx:04d}.json"
            tmp_emb = emb_file.with_suffix(".npy.tmp")
            tmp_meta = meta_file.with_suffix(".json.tmp")

            np.save(tmp_emb, chunk_emb)
            tmp_meta.write_text(json.dumps(chunk_urls), encoding="utf-8")
            tmp_emb.rename(emb_file)
            tmp_meta.rename(meta_file)

            # Clean up partial files
            partial_emb.unlink(missing_ok=True)
            partial_prog.unlink(missing_ok=True)

            del chunk_emb
            gc.collect()
            mx.clear_cache()
    finally:
        searcher.unload_model()

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
    """Load all chunk files, concatenate with existing data, and write the
    final master cache.  Returns the merged embedding array."""
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

    # Clean up chunk directory
    shutil.rmtree(chunk_dir)
    logger.info("Gather complete — merged shape %s. Chunks cleaned up.", merged.shape)
    return merged


# ---------------------------------------------------------------------------
# Direct delta-update path (small batches — original behaviour)
# ---------------------------------------------------------------------------

def _direct_update(
    texts: list[str],
    urls: list[str],
    existing_embeddings: np.ndarray | None,
    existing_urls: list[str],
    emb_path: Path,
    meta_path: Path,
) -> np.ndarray:
    """Embed a small batch of new cases and merge directly."""
    from sst_navigator.embedder import SemanticSearcher

    searcher = SemanticSearcher()
    searcher.load_model()
    try:
        new_embeddings = searcher.embed_texts(
            texts,
            batch_size=BATCH_SIZE,
            max_tokens=config.EMBEDDING_MAX_TOKENS,
        )
    finally:
        searcher.unload_model()

    if existing_embeddings is not None:
        merged = np.concatenate([existing_embeddings, new_embeddings], axis=0)
    else:
        merged = new_embeddings

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / config.EMBEDDING_CACHE_FILE
    meta_path = cache_dir / config.EMBEDDING_METADATA_FILE
    chunk_dir = cache_dir / "tmp_chunks"

    # -- Load existing cache (local-first, then remote) --------------------
    existing_embeddings, existing_urls = _load_existing_cache(
        cache_dir, emb_path, meta_path, args.repo_id, args.repo_type,
    )
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

    # -- Choose strategy ---------------------------------------------------
    if len(new_texts) >= SCATTER_GATHER_THRESHOLD:
        logger.info(
            "Large batch (%d cases) — using Scatter-Gather with chunk_size=%d.",
            len(new_texts),
            args.chunk_size,
        )
        num_chunks = _scatter(new_texts, new_urls, chunk_dir, args.chunk_size)
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
