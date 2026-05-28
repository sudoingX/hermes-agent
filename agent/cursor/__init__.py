"""Cursor provider runtime package (CLI + SDK backends)."""

from agent.cursor.accumulator import CursorTurnAccumulator, _StreamJsonAccumulator
from agent.cursor.backend import (
    cursor_sdk_installed,
    ensure_cursor_sdk,
    resolve_cursor_backend,
)
from agent.cursor.client import CursorAgentClient
from agent.cursor.constants import (
    CURSOR_MARKER_BASE_URL,
    DEFAULT_CURSOR_COMMAND,
    DEFAULT_CURSOR_MODEL,
    DEFAULT_CURSOR_MODE,
)
from agent.cursor.events import (
    run_stream_event_to_events,
    sdk_message_to_events,
    stream_json_dict_to_events,
)
from agent.cursor.prompt import format_messages_as_prompt
from agent.cursor.sdk_backend import SdkSession, run_prompt_via_sdk
from agent.cursor.tool_events import CursorToolEvent, _CursorToolEvent

__all__ = [
    "CURSOR_MARKER_BASE_URL",
    "CursorAgentClient",
    "CursorToolEvent",
    "CursorTurnAccumulator",
    "DEFAULT_CURSOR_COMMAND",
    "DEFAULT_CURSOR_MODEL",
    "DEFAULT_CURSOR_MODE",
    "SdkSession",
    "_CursorToolEvent",
    "_StreamJsonAccumulator",
    "cursor_sdk_installed",
    "ensure_cursor_sdk",
    "format_messages_as_prompt",
    "resolve_cursor_backend",
    "run_prompt_via_sdk",
    "run_stream_event_to_events",
    "sdk_message_to_events",
    "stream_json_dict_to_events",
]
