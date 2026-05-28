"""OpenAI-compatible facade that forwards Hermes requests to Cursor (CLI or SDK).

This module re-exports the :mod:`agent.cursor` package for backward compatibility.
New code should import from ``agent.cursor`` directly.
"""

from __future__ import annotations

from agent.cursor.accumulator import CursorTurnAccumulator, _StreamJsonAccumulator
from agent.cursor.backend import cursor_sdk_installed, resolve_cursor_backend
from agent.cursor.client import CursorAgentClient
from agent.cursor.constants import (
    CURSOR_MARKER_BASE_URL,
    DEFAULT_CURSOR_COMMAND,
    DEFAULT_CURSOR_MODEL,
    DEFAULT_CURSOR_MODE,
)
from agent.cursor.env import (
    build_subprocess_env as _build_subprocess_env,
    resolve_command as _resolve_command,
)
from agent.cursor.prompt import format_messages_as_prompt as _format_messages_as_prompt
from agent.cursor.sdk_backend import SdkSession as _SdkSession, run_prompt_via_sdk
from agent.cursor.tool_events import (
    CursorToolEvent as _CursorToolEvent,
    build_cursor_tool_preview as _build_cursor_tool_preview,
    normalize_cursor_tool_name as _normalize_cursor_tool_name,
)

__all__ = [
    "CursorAgentClient",
    "CURSOR_MARKER_BASE_URL",
    "DEFAULT_CURSOR_COMMAND",
    "DEFAULT_CURSOR_MODE",
    "DEFAULT_CURSOR_MODEL",
    "_CursorToolEvent",
    "_StreamJsonAccumulator",
    "_SdkSession",
    "_build_cursor_tool_preview",
    "_build_subprocess_env",
    "_format_messages_as_prompt",
    "_normalize_cursor_tool_name",
    "_resolve_command",
    "cursor_sdk_installed",
    "resolve_cursor_backend",
    "run_prompt_via_sdk",
]
