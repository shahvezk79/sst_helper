"""
Data loader for SST decisions from the A2AJ Canadian Case Law dataset.
Streams the Parquet file from Hugging Face and prepares it for the pipeline.
"""

import logging
import os
import ssl

import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover - guarded fallback for legacy envs
    certifi = None

logger = logging.getLogger(__name__)

PARQUET_URL = (
    "https://huggingface.co/datasets/a2aj/canadian-case-law"
    "/resolve/main/SST/train.parquet"
)

REQUIRED_COLUMNS = ["name_en", "document_date_en", "url_en", "unofficial_text_en"]


def _is_ssl_cert_error(error: Exception) -> bool:
    return "CERTIFICATE_VERIFY_FAILED" in str(error)


def _configure_certifi_ca_bundle() -> bool:
    """Use certifi's CA bundle for urllib/requests-driven HTTPS calls."""
    if certifi is None:
        return False

    ca_path = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_path)
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=ca_path
    )
    return True


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
        if _is_ssl_cert_error(e) and _configure_certifi_ca_bundle():
            logger.warning(
                "SSL certificate verification failed. Retrying with certifi CA bundle."
            )
            try:
                df = pd.read_parquet(PARQUET_URL)
            except Exception as retry_error:
                logger.error(
                    "Failed to load parquet from %s after certifi retry: %s",
                    PARQUET_URL,
                    retry_error,
                )
                raise RuntimeError(
                    "Could not load the SST dataset due to an SSL certificate error. "
                    "Install/update system certificates, then retry.\n"
                    "Mac fix: run '/Applications/Python 3.x/Install Certificates.command' "
                    "or set SSL_CERT_FILE to certifi's bundle.\n"
                    f"Original error: {retry_error}"
                ) from retry_error
        else:
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
