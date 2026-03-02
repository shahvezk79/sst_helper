"""
Data loader for SST decisions from the A2AJ Canadian Case Law dataset.
Streams the Parquet file from Hugging Face and prepares it for the pipeline.
"""

import logging
import os
import ssl
import tempfile

import pandas as pd
import requests

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


def _download_parquet_to_tempfile(verify: bool | str = True) -> str:
    """Download parquet to a temporary file and return its path."""
    response = requests.get(PARQUET_URL, stream=True, timeout=90, verify=verify)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                tmp_file.write(chunk)
        return tmp_file.name


def _read_parquet_with_requests(verify: bool | str = True) -> pd.DataFrame:
    temp_file = _download_parquet_to_tempfile(verify=verify)
    try:
        return pd.read_parquet(temp_file)
    finally:
        try:
            os.unlink(temp_file)
        except OSError:
            logger.warning("Could not clean up temporary parquet file: %s", temp_file)


def _ssl_verify_setting() -> bool | str:
    custom_ca_bundle = os.getenv("SST_NAVIGATOR_CA_BUNDLE")
    if custom_ca_bundle:
        return custom_ca_bundle

    if os.getenv("SST_NAVIGATOR_DISABLE_SSL_VERIFY", "").lower() in {"1", "true", "yes"}:
        logger.warning(
            "SST_NAVIGATOR_DISABLE_SSL_VERIFY is enabled. HTTPS certificate verification is disabled."
        )
        return False

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
    verify_setting = _ssl_verify_setting()

    try:
        df = _read_parquet_with_requests(verify=verify_setting)
    except Exception as e:
        if _is_ssl_cert_error(e) and _configure_certifi_ca_bundle():
            logger.warning(
                "SSL certificate verification failed. Retrying with certifi CA bundle."
            )
            try:
                df = _read_parquet_with_requests(verify=certifi.where())
            except Exception as retry_error:
                logger.error(
                    "Failed to load parquet from %s after certifi retry: %s",
                    PARQUET_URL,
                    retry_error,
                )
                raise RuntimeError(
                    "Could not load the SST dataset due to an SSL certificate error. "
                    "If your organization uses a proxy/SSL inspection, set "
                    "SST_NAVIGATOR_CA_BUNDLE to your org CA path.\n"
                    "Mac fix: run '/Applications/Python 3.x/Install Certificates.command'.\n"
                    "Temporary fallback (not recommended): set SST_NAVIGATOR_DISABLE_SSL_VERIFY=1.\n"
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
