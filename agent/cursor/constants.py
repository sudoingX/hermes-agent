"""Shared constants for the Cursor provider."""
from __future__ import annotations

CURSOR_MARKER_BASE_URL = "cursor://agent"
DEFAULT_CURSOR_COMMAND = "cursor-agent"
DEFAULT_CURSOR_MODE = "agent"
DEFAULT_CURSOR_MODEL = "auto"

_VALID_CURSOR_MODES = frozenset({"ask", "plan", "agent"})
_CURSOR_CLI_MODES = frozenset({"ask", "plan"})
_DEFAULT_TIMEOUT_SECONDS = 1800.0

_API_KEY_SENTINELS = frozenset({
    "",
    "cursor-agent-login",
    "cursor-cli-login",
    "external-process",
    "external_process",
})

DEFAULT_CURSOR_BACKEND = "auto"
_VALID_BACKENDS = frozenset({"auto", "cli", "sdk"})
_SDK_MODES = frozenset({"agent", "plan"})

__all__ = [
    "CURSOR_MARKER_BASE_URL",
    "DEFAULT_CURSOR_COMMAND",
    "DEFAULT_CURSOR_MODE",
    "DEFAULT_CURSOR_MODEL",
    "DEFAULT_CURSOR_BACKEND",
    "_VALID_CURSOR_MODES",
    "_CURSOR_CLI_MODES",
    "_DEFAULT_TIMEOUT_SECONDS",
    "_API_KEY_SENTINELS",
    "_VALID_BACKENDS",
    "_SDK_MODES",
]
