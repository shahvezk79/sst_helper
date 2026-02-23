"""Delta-update SST embedding cache and push it to Hugging Face."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sst_navigator import config
from sst_navigator.data_loader import load_sst_decisions
logger = logging.getLogger(__name__)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed only new SST cases and update the remote cache."
    )
    parser.add_argument("--repo-id", default=config.EMBEDDING_CACHE_REPO_ID)
    parser.add_argument("--repo-type", default=config.EMBEDDING_CACHE_REPO_TYPE)
    parser.add_argument("--cache-dir", default=config.EMBEDDING_CACHE_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / config.EMBEDDING_CACHE_FILE
    meta_path = cache_dir / config.EMBEDDING_METADATA_FILE

    logger.info("Downloading existing cache state from %s", args.repo_id)
    try:
        hf_hub_download(
            repo_id=args.repo_id,
            filename=config.EMBEDDING_CACHE_FILE,
            repo_type=args.repo_type,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        hf_hub_download(
            repo_id=args.repo_id,
            filename=config.EMBEDDING_METADATA_FILE,
            repo_type=args.repo_type,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        existing_embeddings = np.load(emb_path, allow_pickle=False)
        existing_urls = _load_metadata(meta_path)
        existing_url_set = set(existing_urls)
    except Exception as e:
        logger.warning("Could not download existing cache (likely empty repo). Starting fresh.")
        existing_embeddings = None
        existing_urls = []
        existing_url_set = set()

    logger.info("Loading latest SST decisions dataset")
    df = load_sst_decisions(max_rows=None)

    new_df = df[~df["url_en"].isin(existing_url_set)].copy()
    if new_df.empty:
        logger.info("No new decisions found. Cache is already up to date.")
        return

    logger.info("Found %d new decisions", len(new_df))
    from sst_navigator.embedder import SemanticSearcher

    searcher = SemanticSearcher()
    searcher.load_model()
    try:
        new_embeddings = searcher.embed_texts(
            new_df["unofficial_text_en"].tolist(),
            batch_size=config.EMBEDDING_BATCH_SIZE_DEV,
            max_tokens=config.EMBEDDING_MAX_TOKENS,
        )
    finally:
        searcher.unload_model()

    if existing_embeddings is not None:
        merged_embeddings = np.concatenate([existing_embeddings, new_embeddings], axis=0)
    else:
        merged_embeddings = new_embeddings
        
    merged_urls = existing_urls + new_df["url_en"].astype(str).tolist()

    np.save(emb_path, merged_embeddings)
    _write_metadata(meta_path, merged_urls)

    logger.info("Uploading updated cache files to %s", args.repo_id)
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
        "Update complete. New shape: %s (added %d rows)",
        merged_embeddings.shape,
        len(new_df),
    )


if __name__ == "__main__":
    main()
