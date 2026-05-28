"""Prompt formatting shared by CLI and SDK Cursor backends."""

from __future__ import annotations

import json
from typing import Any

from agent.copilot_acp_client import _render_message_content


def format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    """Build the prompt sent to cursor-agent stdin or SDK ``send()``."""
    sections: list[str] = []
    has_tools = bool(tools)
    if has_tools:
        sections.extend([
            "You are powering a chat session inside Hermes Agent.",
            "You have TWO sets of tools available:",
            "(A) Your own built-in cursor-agent tools (shell, read_file, "
            "edit_file, write_file, list_directory, grep, glob, web_fetch). "
            "Use these DIRECTLY for filesystem/shell/search work — they run "
            "on the real workspace, are fast, and Hermes will surface their "
            "results to the user automatically.",
            "(B) Hermes-side tools listed in the schema below. They cover "
            "capabilities your built-in tools do NOT have (skills, MCP "
            "servers, browser automation, remote APIs, etc.). To invoke "
            "one of THESE, emit a "
            "<tool_call>{...}</tool_call> block in OpenAI function-call "
            "shape: "
            '{"id":"call_<n>","type":"function",'
            '"function":{"name":"<tool>","arguments":"<json string>"}}. '
            "``arguments`` MUST be a JSON STRING (escaped), not a nested "
            "object.",
            "RULES:",
            "1. Prefer your built-in tools for any shell command, file "
            "read/write/list/edit, grep, or glob operation — they're "
            "faster than round-tripping through Hermes. CRITICAL for "
            "file creation/modification: ALWAYS use the ``write`` or "
            "``edit`` built-in tools, NEVER ``shell`` with ``echo > "
            "file`` / ``cat > file`` / ``sed -i`` / ``>>``. Only the "
            "write/edit tools report ``linesAdded`` / ``linesRemoved`` "
            "/ ``diffString`` to the harness, which is what Hermes "
            "renders as the colored ``+``/``-`` diff in the UI. Shell "
            "redirections create the file but the user sees no diff "
            "and has no idea what changed.",
            "2. Only emit <tool_call> blocks for tools listed in the "
            "schema below; do NOT invent tool names. Multiple tool_calls "
            "per turn are allowed.",
            "3. Work iteratively (ReAct-style): before each tool batch, "
            "emit ONE short line of plain text saying what you're about "
            "to check and why. After tool results come back, briefly "
            "reflect on what you found before deciding the next step. "
            "Hermes surfaces these intermediate lines to the user as "
            "live narration so they can follow your reasoning.",
            "4. Don't dump every tool call upfront — chain them: think, "
            "tool, reflect, tool, reflect, ... then synthesise the final "
            "answer at the end. If the task genuinely is independent "
            "lookups, parallel tool calls in one batch are fine.",
            "5. If no tool is needed (pure conversation, math, "
            "summarising content already in the transcript), answer as "
            "plain text.",
            "6. Never hallucinate file contents or command output — if "
            "you say \"Reading the file…\" you MUST actually run the "
            "read_file (built-in) or emit a <tool_call> if it's a "
            "Hermes-specific tool.",
            "7. The Hermes UI already shows file edits to the user as a "
            "colored +/- diff right next to each ``edit`` / ``write`` "
            "tool call (and tool calls + diffs are streamed live). Do "
            "NOT re-dump the before/after content or paste the diff "
            "again in your final response — just confirm what was "
            "changed at a high level (e.g. \"updated foo.py to fix the "
            "off-by-one\"). Same for shell output: it's already visible.",
        ])
    else:
        sections.append(
            "Hermes auxiliary call. Answer the user message below directly "
            "and concisely; do not run any tools, do not write files, do "
            "not ask follow-up questions. Plain-text reply only."
        )
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Hermes-side tools (OpenAI function schema). Emit "
                "<tool_call>{...}</tool_call> blocks to invoke these. "
                "For plain shell / file / grep / glob actions prefer your "
                "own built-in tools instead (they're faster).\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(
            f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}"
        )

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


__all__ = ["format_messages_as_prompt"]
