"""Cursor internal tool event types and helpers."""

from __future__ import annotations

import json
import time
from typing import Any


def build_cursor_tool_preview(evt: "CursorToolEvent") -> str:
    """Compact one-line description of a cursor tool call for the UI."""
    args = evt.args or {}
    try:
        if evt.envelope_key == "shellToolCall":
            cmd = args.get("command") or args.get("cmd")
            if isinstance(cmd, list):
                cmd = " ".join(str(part) for part in cmd)
            if isinstance(cmd, str) and cmd.strip():
                return cmd.strip()[:200]
        if evt.envelope_key in (
            "readToolCall",
            "editToolCall",
            "writeToolCall",
            "patchToolCall",
            "deleteToolCall",
        ):
            path = (
                args.get("path")
                or args.get("file")
                or args.get("filePath")
                or args.get("filename")
                or args.get("target_file")
                or args.get("targetFile")
                or args.get("file_path")
                or args.get("relative_workspace_path")
                or ""
            )
            if isinstance(path, str) and path.strip():
                return path.strip()[:200]
        if evt.envelope_key == "globToolCall":
            pat = args.get("globPattern") or args.get("pattern") or ""
            target = args.get("targetDirectory") or args.get("path") or ""
            label = " in ".join(p for p in (pat, target) if isinstance(p, str) and p.strip())
            if label:
                return label[:200]
        if evt.envelope_key in ("grepToolCall", "searchToolCall"):
            pat = args.get("pattern") or args.get("query") or args.get("regex") or ""
            target = args.get("path") or args.get("targetDirectory") or ""
            if isinstance(pat, str) and pat.strip():
                if isinstance(target, str) and target.strip():
                    return f"{pat} in {target}"[:200]
                return pat.strip()[:200]
            if isinstance(target, str) and target.strip():
                return target.strip()[:200]
        if evt.envelope_key == "listToolCall":
            path = args.get("path") or args.get("directory") or args.get("targetDirectory") or ""
            if isinstance(path, str) and path.strip():
                return path.strip()[:200]
        return json.dumps(args, ensure_ascii=False)[:200]
    except Exception:
        return ""


def normalize_cursor_tool_name(envelope_key: str) -> str:
    """Map cursor's ``<thing>ToolCall`` keys to Hermes tool names."""
    if not isinstance(envelope_key, str):
        return "cursor_tool"
    suffix = "ToolCall"
    base = envelope_key[: -len(suffix)] if envelope_key.endswith(suffix) else envelope_key
    if not base:
        return "cursor_tool"
    return {
        "shell": "shell",
        "read": "read_file",
        "list": "list_directory",
        "edit": "edit_file",
        "write": "write_file",
        "patch": "patch",
        "grep": "grep",
        "glob": "glob",
        "search": "search",
        "todo": "todo",
        "delete": "delete_file",
        "task": "task",
        "fetch": "web_fetch",
    }.get(base.lower(), base)


def summarise_cursor_tool_result(envelope_key: str, payload: dict[str, Any]) -> str:
    """Return a compact human-readable result string for the UI / log."""
    result = payload.get("result")
    if not isinstance(result, dict):
        return ""
    success = result.get("success")
    if not isinstance(success, dict):
        if "error" in result and isinstance(result["error"], (str, dict)):
            return f"error: {result['error']}"[:400]
        return ""
    try:
        if envelope_key == "shellToolCall":
            stdout = success.get("stdout") or ""
            return stdout if isinstance(stdout, str) else json.dumps(stdout)
        if envelope_key == "readToolCall":
            content = success.get("content") or ""
            total = success.get("totalLines")
            if total is not None:
                return f"({total} lines)\n{content}" if content else f"({total} lines)"
            return content if isinstance(content, str) else json.dumps(content)
        if envelope_key in ("listToolCall", "globToolCall"):
            files = success.get("files") or success.get("entries") or []
            if isinstance(files, list):
                return "\n".join(str(f) for f in files[:200])
        return json.dumps(success, ensure_ascii=False)[:1000]
    except Exception:
        return ""


class CursorToolEvent:
    """A captured cursor-agent tool invocation (started + completed states)."""

    __slots__ = (
        "call_id", "envelope_key", "name", "args", "started_at",
        "completed_at", "result_text", "is_error", "duration_ms",
        "lines_added", "lines_removed", "diff_string",
    )

    def __init__(self, call_id: str, envelope_key: str, args: dict[str, Any]) -> None:
        self.call_id = call_id
        self.envelope_key = envelope_key
        self.name = normalize_cursor_tool_name(envelope_key)
        self.args = args
        self.started_at = time.monotonic()
        self.completed_at: float | None = None
        self.result_text: str = ""
        self.is_error: bool = False
        self.duration_ms: int = 0
        self.lines_added: int | None = None
        self.lines_removed: int | None = None
        self.diff_string: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "name": self.name,
            "envelope": self.envelope_key,
            "arguments": self.args,
            "result": self.result_text,
            "is_error": self.is_error,
            "duration_ms": self.duration_ms,
        }


# Backward-compat alias for tests and internal imports.
_CursorToolEvent = CursorToolEvent


__all__ = [
    "CursorToolEvent",
    "_CursorToolEvent",
    "build_cursor_tool_preview",
    "normalize_cursor_tool_name",
    "summarise_cursor_tool_result",
]
