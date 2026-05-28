"""Accumulates Cursor turn state from typed events (CLI or SDK)."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from agent.cursor.events import (
    AssistantTextEvent,
    CursorTurnEvent,
    SystemEvent,
    ThinkingEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    TurnResultEvent,
    stream_json_dict_to_events,
)
from agent.cursor.tool_events import (
    CursorToolEvent,
    summarise_cursor_tool_result,
)


class CursorTurnAccumulator:
    """Accumulates state from a cursor turn event feed.

    Caller feeds typed events with :meth:`feed`. When a terminal
    :class:`TurnResultEvent` arrives the accumulator stores success/failure
    state and surface text. The instance is reusable per-call but not
    thread-safe.
    """

    def __init__(self, on_tool_event: Any = None, on_text_event: Any = None) -> None:
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.session_id: str = ""
        self.request_id: str = ""
        self.model_label: str = ""
        self.duration_ms: int = 0
        self.usage: dict[str, int] = {}
        self.terminal: bool = False
        self.is_error: bool = False
        self.error_message: str = ""
        self.final_result_text: str = ""
        self.event_log: list[tuple[str, Any]] = []
        self._on_tool_event = on_tool_event
        self._on_text_event = on_text_event
        self._tool_events: dict[str, CursorToolEvent] = {}
        self.tool_events: list[CursorToolEvent] = []
        self.messages_estimate: int = 0
        self._pending_text: list[str] = []

    def feed(self, event: CursorTurnEvent | dict[str, Any]) -> None:
        """Accept a typed event or legacy stream-json dict."""
        if isinstance(event, dict):
            for typed in stream_json_dict_to_events(event):
                self.feed(typed)
            return

        if isinstance(event, SystemEvent):
            if event.model:
                self.model_label = event.model
            if event.session_id:
                self.session_id = event.session_id
            return

        if isinstance(event, ThinkingEvent):
            if event.text:
                self.reasoning_parts.append(event.text)
            return

        if isinstance(event, AssistantTextEvent):
            if event.text:
                self.text_parts.append(event.text)
                self.event_log.append(("text", event.text))
                self._pending_text.append(event.text)
            return

        if isinstance(event, ToolStartedEvent):
            if self._pending_text:
                for buffered in self._pending_text:
                    self._dispatch_text_event(buffered)
                self._pending_text.clear()
            self._consume_tool_started(event)
            self.event_log.append(("tool", None))
            return

        if isinstance(event, ToolCompletedEvent):
            self._consume_tool_completed(event)
            return

        if isinstance(event, TurnResultEvent):
            self.terminal = True
            self.is_error = event.is_error
            self.duration_ms = event.duration_ms
            if event.request_id:
                self.request_id = event.request_id
            if event.usage:
                self.usage = dict(event.usage)
            if event.result_text:
                self.final_result_text = event.result_text
                if not self.text_parts and not self.is_error:
                    self.text_parts.append(event.result_text)
            if self.is_error and not self.error_message:
                self.error_message = event.error_message or event.result_text or "cursor-agent returned an error"
            return

    def _consume_tool_started(self, event: ToolStartedEvent) -> None:
        evt = CursorToolEvent(
            call_id=event.call_id,
            envelope_key=event.envelope_key,
            args=event.args,
        )
        self._tool_events[event.call_id] = evt
        self.tool_events.append(evt)
        self._fire_tool_event("started", evt)

    def _consume_tool_completed(self, event: ToolCompletedEvent) -> None:
        evt = self._tool_events.get(event.call_id)
        if evt is None:
            evt = CursorToolEvent(
                call_id=event.call_id,
                envelope_key=event.envelope_key,
                args=event.args,
            )
            self._tool_events[event.call_id] = evt
            self.tool_events.append(evt)
            self._fire_tool_event("started", evt)
        evt.completed_at = time.monotonic()
        evt.duration_ms = int((evt.completed_at - evt.started_at) * 1000)
        result = event.result_payload.get("result")
        if isinstance(result, dict):
            if "error" in result and result.get("error"):
                evt.is_error = True
            success = result.get("success") if isinstance(result, dict) else None
            if isinstance(success, dict):
                la = success.get("linesAdded")
                lr = success.get("linesRemoved")
                ds = success.get("diffString")
                if isinstance(la, int):
                    evt.lines_added = la
                if isinstance(lr, int):
                    evt.lines_removed = lr
                if isinstance(ds, str):
                    evt.diff_string = ds
        evt.result_text = summarise_cursor_tool_result(event.envelope_key, event.result_payload)
        self._fire_tool_event("completed", evt)

    def _fire_tool_event(self, stage: str, evt: CursorToolEvent) -> None:
        if self._on_tool_event is None:
            return
        try:
            self._on_tool_event(stage, evt)
        except Exception:
            pass

    def _dispatch_text_event(self, text: str) -> None:
        if self._on_text_event is None:
            return
        try:
            self._on_text_event(text)
        except Exception:
            pass

    def assembled_text(self) -> str:
        return "".join(self.text_parts).strip()

    def synthesis_text(self) -> str:
        tool_seen = False
        synth: list[str] = []
        for kind, payload in self.event_log:
            if kind == "tool":
                tool_seen = True
                synth.clear()
            elif kind == "text":
                synth.append(payload)
        if synth:
            return "".join(synth).strip()
        if not tool_seen:
            return self.assembled_text()
        if self.final_result_text and self.final_result_text.strip():
            return self.final_result_text.strip()
        return self.assembled_text()

    def narration_text(self) -> str:
        narration: list[str] = []
        bucket: list[str] = []
        for kind, payload in self.event_log:
            if kind == "tool":
                if bucket:
                    narration.append("".join(bucket).strip())
                bucket = []
            elif kind == "text":
                bucket.append(payload)
        return "\n".join(n for n in narration if n)

    def assembled_reasoning(self) -> str:
        return "".join(self.reasoning_parts).strip()

    def openai_usage(self) -> SimpleNamespace:
        input_tokens_raw = int(self.usage.get("inputTokens", 0))
        output_tokens = int(self.usage.get("outputTokens", 0))
        cache_read_raw = int(self.usage.get("cacheReadTokens", 0))

        rounds = max(len(self.tool_events) + 1, 1)
        per_round_input = input_tokens_raw // rounds if rounds > 0 else input_tokens_raw
        per_round_cache = cache_read_raw // rounds if rounds > 0 else cache_read_raw
        approx_context_tokens = per_round_cache + per_round_input

        if self.messages_estimate > 0:
            prompt_tokens = self.messages_estimate
        else:
            prompt_tokens = approx_context_tokens

        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=per_round_cache),
            cursor_raw_input_tokens=input_tokens_raw,
            cursor_raw_cache_read_tokens=cache_read_raw,
            cursor_internal_rounds=rounds,
            cursor_per_round_context=approx_context_tokens,
        )


# Backward-compat alias for tests and legacy imports.
_StreamJsonAccumulator = CursorTurnAccumulator


__all__ = [
    "CursorTurnAccumulator",
    "_StreamJsonAccumulator",
]
