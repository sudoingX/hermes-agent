"""Typed turn events for Cursor CLI and SDK backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Union


@dataclass(frozen=True)
class SystemEvent:
    model: str = ""
    session_id: str = ""


@dataclass(frozen=True)
class ThinkingEvent:
    text: str


@dataclass(frozen=True)
class AssistantTextEvent:
    text: str


@dataclass(frozen=True)
class ToolStartedEvent:
    call_id: str
    envelope_key: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolCompletedEvent:
    call_id: str
    envelope_key: str
    args: dict[str, Any]
    result_payload: dict[str, Any]


@dataclass(frozen=True)
class TurnResultEvent:
    is_error: bool
    result_text: str = ""
    request_id: str = ""
    duration_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    error_message: str = ""


CursorTurnEvent = Union[
    SystemEvent,
    ThinkingEvent,
    AssistantTextEvent,
    ToolStartedEvent,
    ToolCompletedEvent,
    TurnResultEvent,
]

_TOOL_NAME_TO_ENVELOPE: dict[str, str] = {
    "shell": "shellToolCall",
    "read": "readToolCall",
    "read_file": "readToolCall",
    "list": "listToolCall",
    "list_directory": "listToolCall",
    "edit": "editToolCall",
    "edit_file": "editToolCall",
    "write": "writeToolCall",
    "write_file": "writeToolCall",
    "patch": "patchToolCall",
    "grep": "grepToolCall",
    "glob": "globToolCall",
    "search": "searchToolCall",
    "delete": "deleteToolCall",
    "delete_file": "deleteToolCall",
    "web_fetch": "fetchToolCall",
    "fetch": "fetchToolCall",
}


def tool_name_to_envelope(name: str) -> str:
    if name.endswith("ToolCall"):
        return name
    return _TOOL_NAME_TO_ENVELOPE.get(name.lower(), f"{name}ToolCall")


def normalize_tool_call_envelope(tool_call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(envelope_key, payload)`` from SDK or stream-json tool_call dict."""
    if not tool_call:
        return "", {}
    for key, payload in tool_call.items():
        if isinstance(key, str) and key.endswith("ToolCall") and isinstance(payload, Mapping):
            return key, dict(payload)
    name = str(tool_call.get("name") or tool_call.get("toolName") or "")
    args = tool_call.get("args")
    if not isinstance(args, Mapping):
        args = tool_call.get("input") if isinstance(tool_call.get("input"), Mapping) else {}
    envelope = tool_name_to_envelope(name or "cursor")
    payload: dict[str, Any] = {"args": dict(args or {})}
    if "result" in tool_call:
        payload["result"] = tool_call["result"]
    return envelope, payload


def stream_json_dict_to_events(event: dict[str, Any]) -> list[CursorTurnEvent]:
    """Translate one cursor-agent stream-json dict into typed turn events."""
    evt_type = event.get("type")
    if not isinstance(evt_type, str):
        return []

    if evt_type == "system":
        model = event.get("model")
        session = event.get("session_id")
        return [SystemEvent(
            model=model if isinstance(model, str) else "",
            session_id=session if isinstance(session, str) else "",
        )]

    if evt_type == "thinking":
        text = event.get("text")
        if isinstance(text, str) and text:
            return [ThinkingEvent(text=text)]
        return []

    if evt_type == "assistant":
        out: list[CursorTurnEvent] = []
        message = event.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            out.append(AssistantTextEvent(text=text))
        return out

    if evt_type == "tool_call":
        sub = event.get("subtype")
        call_id = event.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return []
        tool_call = event.get("tool_call")
        if not isinstance(tool_call, dict) or not tool_call:
            return []
        envelope_key = next(iter(tool_call.keys()), "")
        payload = tool_call.get(envelope_key) if isinstance(envelope_key, str) else None
        if not isinstance(payload, dict):
            return []
        args_obj = payload.get("args")
        if not isinstance(args_obj, dict):
            args_obj = {}
        if sub == "started":
            return [ToolStartedEvent(call_id=call_id, envelope_key=envelope_key, args=args_obj)]
        if sub == "completed":
            return [ToolCompletedEvent(
                call_id=call_id,
                envelope_key=envelope_key,
                args=args_obj,
                result_payload=payload,
            )]
        return []

    if evt_type == "result":
        is_error = bool(event.get("is_error", False))
        subtype = event.get("subtype")
        if subtype == "error":
            is_error = True
        duration = event.get("duration_ms")
        duration_ms = duration if isinstance(duration, int) else 0
        request = event.get("request_id")
        request_id = request if isinstance(request, str) else ""
        usage_raw = event.get("usage")
        usage: dict[str, int] = {}
        if isinstance(usage_raw, dict):
            for k, v in usage_raw.items():
                if isinstance(v, (int, float)):
                    usage[str(k)] = int(v)
        result_text = event.get("result")
        result_str = result_text if isinstance(result_text, str) else ""
        error_message = ""
        if is_error and not error_message:
            error_message = result_str or "cursor-agent returned an error"
        return [TurnResultEvent(
            is_error=is_error,
            result_text=result_str,
            request_id=request_id,
            duration_ms=duration_ms,
            usage=usage,
            error_message=error_message,
        )]

    return []


def sdk_message_to_events(message: Any) -> list[CursorTurnEvent]:
    """Translate one SDKMessage into typed turn events."""
    msg_type = getattr(message, "type", None)
    if msg_type == "system":
        model = getattr(getattr(message, "model", None), "id", None) or ""
        return [SystemEvent(
            model=model,
            session_id=getattr(message, "agent_id", "") or getattr(message, "run_id", ""),
        )]
    if msg_type == "thinking":
        text = getattr(message, "text", "")
        if text:
            return [ThinkingEvent(text=text)]
        return []
    if msg_type == "assistant":
        out: list[CursorTurnEvent] = []
        msg = getattr(message, "message", None)
        for block in getattr(msg, "content", ()) or ():
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "")
                if text:
                    out.append(AssistantTextEvent(text=text))
        return out
    if msg_type == "tool_call":
        envelope = tool_name_to_envelope(getattr(message, "name", "") or "cursor")
        args = getattr(message, "args", None)
        if not isinstance(args, Mapping):
            args = {}
        status = str(getattr(message, "status", "") or "").lower()
        call_id = getattr(message, "call_id", "") or ""
        if status in {"running", "started"}:
            return [ToolStartedEvent(call_id=call_id, envelope_key=envelope, args=dict(args))]
        if status in {"completed", "error", "failed"}:
            result = getattr(message, "result", None)
            payload: dict[str, Any] = {"args": dict(args)}
            if isinstance(result, Mapping):
                payload["result"] = dict(result)
            elif status == "error":
                payload["result"] = {"error": result or "tool error"}
            else:
                payload["result"] = {"success": result} if result is not None else {}
            return [ToolCompletedEvent(
                call_id=call_id,
                envelope_key=envelope,
                args=dict(args),
                result_payload=payload,
            )]
        return []
    return []


def interaction_update_to_events(update: Any) -> list[CursorTurnEvent]:
    """Translate InteractionUpdate events into typed turn events."""
    update_type = getattr(update, "type", None)
    if update_type == "text-delta":
        text = getattr(update, "text", "")
        if text:
            return [AssistantTextEvent(text=text)]
        return []
    if update_type == "thinking-delta":
        text = getattr(update, "text", "")
        if text:
            return [ThinkingEvent(text=text)]
        return []
    if update_type == "tool-call-started":
        tool_call = getattr(update, "tool_call", {}) or {}
        envelope, payload = normalize_tool_call_envelope(tool_call)
        if not envelope:
            return []
        args = payload.get("args")
        if not isinstance(args, dict):
            args = {}
        return [ToolStartedEvent(
            call_id=getattr(update, "call_id", "") or "",
            envelope_key=envelope,
            args=args,
        )]
    if update_type == "tool-call-completed":
        tool_call = getattr(update, "tool_call", {}) or {}
        envelope, payload = normalize_tool_call_envelope(tool_call)
        if not envelope:
            return []
        args = payload.get("args")
        if not isinstance(args, dict):
            args = {}
        return [ToolCompletedEvent(
            call_id=getattr(update, "call_id", "") or "",
            envelope_key=envelope,
            args=args,
            result_payload=payload,
        )]
    if update_type == "turn-ended":
        usage = getattr(update, "usage", None)
        if isinstance(usage, Mapping) and usage:
            usage_dict = {str(k): int(v) for k, v in usage.items() if isinstance(v, (int, float))}
            return [TurnResultEvent(
                is_error=False,
                duration_ms=int(usage.get("durationMs") or usage.get("duration_ms") or 0),
                usage=usage_dict,
            )]
        return []
    return []


def run_stream_event_to_events(event: Any) -> list[CursorTurnEvent]:
    """Translate a RunStreamEvent into zero or more typed turn events."""
    kind = getattr(event, "kind", "")
    if kind == "sdk_message" and event.sdk_message is not None:
        return sdk_message_to_events(event.sdk_message)
    if kind == "interaction_update" and event.interaction_update is not None:
        return interaction_update_to_events(event.interaction_update)
    if kind == "result" and event.result is not None:
        payload = dict(event.result)
        status = str(payload.get("status") or "").lower()
        is_error = status in {"error", "failed", "cancelled", "canceled"}
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}
        usage_dict = {str(k): int(v) for k, v in usage.items() if isinstance(v, (int, float))}
        return [TurnResultEvent(
            is_error=is_error,
            result_text=str(payload.get("result") or ""),
            request_id=str(payload.get("runId") or payload.get("id") or ""),
            duration_ms=int(payload.get("durationMs") or payload.get("duration_ms") or 0),
            usage=usage_dict,
            error_message=str(payload.get("result") or "") if is_error else "",
        )]
    return []


__all__ = [
    "AssistantTextEvent",
    "CursorTurnEvent",
    "SystemEvent",
    "ThinkingEvent",
    "ToolCompletedEvent",
    "ToolStartedEvent",
    "TurnResultEvent",
    "interaction_update_to_events",
    "normalize_tool_call_envelope",
    "run_stream_event_to_events",
    "sdk_message_to_events",
    "stream_json_dict_to_events",
    "tool_name_to_envelope",
]
