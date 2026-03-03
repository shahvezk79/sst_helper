"""Authentication helpers for cloud backends."""

from __future__ import annotations

import os


def get_deepinfra_api_key() -> str:
    """Return a sanitized DeepInfra API key from environment.

    Handles common shell mistakes (extra whitespace / quoted value).
    """
    raw = os.environ.get("DEEPINFRA_API_KEY", "")
    api_key = raw.strip().strip("\"'")
    if not api_key:
        raise RuntimeError("Set DEEPINFRA_API_KEY in your environment.")
    return api_key


def deepinfra_auth_error_hint() -> str:
    """Human-friendly hint for 401/403 DeepInfra auth failures."""
    return (
        "DeepInfra authentication failed. Verify DEEPINFRA_API_KEY is a "
        "DeepInfra key (not OPENAI_API_KEY), has access to the configured "
        "model, and does not include extra quotes/spaces."
    )

