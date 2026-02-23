"""
Data loader for SST decisions from the A2AJ Canadian Case Law dataset.
Streams the Parquet file from Hugging Face and prepares it for the pipeline.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

PARQUET_URL = (
    "https://huggingface.co/datasets/a2aj/canadian-case-law"
    "/resolve/main/SST/train.parquet"
)

REQUIRED_COLUMNS = ["name_en", "document_date_en", "url_en", "unofficial_text_en"]


def load_sst_decisions(max_rows: int | None = None) -> pd.DataFrame:
    """Load SST decisions from the A2AJ Hugging Face dataset.

    Args:
        max_rows: If set, only load the first N rows (useful for dev/testing).

    Returns:
        DataFrame with cleaned SST decisions.
    """
    logger.info("Loading SST decisions from Hugging Face (%s rows)...",
                max_rows or "all")
    try:
        df = pd.read_parquet(PARQUET_URL)
    except Exception as e:
        logger.error("Failed to load parquet from %s: %s", PARQUET_URL, e)
        raise RuntimeError(
            f"Could not load the SST dataset. Check your internet connection.\n{e}"
        ) from e

    # Verify expected columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows where the decision text is empty or null
    df = df.dropna(subset=["unofficial_text_en"])
    df = df[df["unofficial_text_en"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows)
        logger.info("Trimmed to %d rows for development.", len(df))

    logger.info("Loaded %d SST decisions.", len(df))
    return df
