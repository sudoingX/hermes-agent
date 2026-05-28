"""Backend selection and cursor-sdk lazy-install helpers."""

from __future__ import annotations

import os

from agent.cursor.constants import (
    DEFAULT_CURSOR_BACKEND,
    _API_KEY_SENTINELS,
    _SDK_MODES,
    _VALID_BACKENDS,
)


def cursor_sdk_installed() -> bool:
    try:
        import cursor_sdk  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_cursor_sdk(*, prompt: bool = False) -> None:
    """Lazy-install cursor-sdk when the SDK backend is selected."""
    if cursor_sdk_installed():
        return
    try:
        from tools import lazy_deps
        from tools.lazy_deps import FeatureUnavailable

        lazy_deps.ensure("provider.cursor_sdk", prompt=prompt)
    except FeatureUnavailable as exc:
        raise RuntimeError(
            "cursor-sdk is not installed. "
            "Run: uv pip install cursor-sdk  (or pip install 'hermes-agent[cursor]')"
        ) from exc


def real_api_key(api_key: str | None) -> str | None:
    key = (api_key or os.getenv("CURSOR_API_KEY", "") or "").strip()
    if not key or key in _API_KEY_SENTINELS:
        return None
    return key


def map_hermes_mode_to_sdk(mode: str) -> str | None:
    normalized = (mode or "agent").strip().lower()
    if normalized == "ask":
        return "plan"
    if normalized in _SDK_MODES:
        return normalized
    return "agent"


def resolve_cursor_backend(*, api_key: str | None = None) -> str:
    """Return the effective backend: ``cli`` or ``sdk``."""
    raw = os.getenv("HERMES_CURSOR_BACKEND", "").strip().lower() or DEFAULT_CURSOR_BACKEND
    if raw not in _VALID_BACKENDS:
        raw = DEFAULT_CURSOR_BACKEND
    if raw == "cli":
        return "cli"
    if raw == "sdk":
        ensure_cursor_sdk(prompt=False)
        if not real_api_key(api_key):
            raise RuntimeError(
                "HERMES_CURSOR_BACKEND=sdk requires CURSOR_API_KEY "
                "(Dashboard → Integrations → User API Keys)."
            )
        return "sdk"
    # auto
    if cursor_sdk_installed() and real_api_key(api_key):
        return "sdk"
    return "cli"


__all__ = [
    "cursor_sdk_installed",
    "ensure_cursor_sdk",
    "map_hermes_mode_to_sdk",
    "real_api_key",
    "resolve_cursor_backend",
]
