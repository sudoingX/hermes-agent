"""OpenAI-compatible facade for the Cursor provider (CLI or SDK transport)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import _extract_tool_calls_from_text
from agent.cursor.accumulator import CursorTurnAccumulator
from agent.cursor.backend import resolve_cursor_backend
from agent.cursor.cli_backend import run_prompt_cli
from agent.cursor.constants import (
    CURSOR_MARKER_BASE_URL,
    DEFAULT_CURSOR_COMMAND,
    DEFAULT_CURSOR_MODEL,
    DEFAULT_CURSOR_MODE,
    _API_KEY_SENTINELS,
    _CURSOR_CLI_MODES,
    _DEFAULT_TIMEOUT_SECONDS,
    _VALID_CURSOR_MODES,
)
from agent.cursor.env import (
    build_subprocess_env,
    resolve_command,
    resolve_extra_args,
    resolve_mode,
    resolve_workspace_override,
)
from agent.cursor.prompt import format_messages_as_prompt
from agent.cursor.sdk_backend import SdkSession, run_prompt_via_sdk
from agent.cursor.streaming import CursorChatNamespace
from agent.cursor.tool_events import CursorToolEvent, build_cursor_tool_preview


class CursorAgentClient:
    """Minimal OpenAI-client-compatible facade for Cursor (CLI or SDK)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        workspace: str | None = None,
        mode: str | None = None,
        timeout_seconds: float | None = None,
        tool_progress_callback: Any = None,
        context_estimate_callback: Any = None,
        **_: Any,
    ):
        candidate_key = (api_key or os.getenv("CURSOR_API_KEY", "") or "").strip()
        self.api_key = None if candidate_key in _API_KEY_SENTINELS else candidate_key
        self.base_url = base_url or CURSOR_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = (command or resolve_command()).strip() or DEFAULT_CURSOR_COMMAND
        self._extra_args = list(args) if args else resolve_extra_args()
        chosen_mode = (mode or resolve_mode()).strip().lower() or DEFAULT_CURSOR_MODE
        if chosen_mode not in _VALID_CURSOR_MODES:
            chosen_mode = DEFAULT_CURSOR_MODE
        self._mode = chosen_mode
        override = workspace or resolve_workspace_override()
        self._workspace: str | None = override or None
        self._timeout_seconds = float(timeout_seconds) if timeout_seconds else _DEFAULT_TIMEOUT_SECONDS
        env_timeout = os.environ.get("HERMES_CURSOR_TIMEOUT_SECONDS", "").strip()
        if env_timeout:
            try:
                env_timeout_val = float(env_timeout)
                if env_timeout_val > 0:
                    self._timeout_seconds = env_timeout_val
            except ValueError:
                pass

        self._tool_progress_callback = tool_progress_callback
        self._context_estimate_callback = context_estimate_callback
        self._context_high_water: int = 0
        self._last_user_msg_count: int = 0

        self.chat = CursorChatNamespace(self)
        self.is_closed = False

        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._ephemeral_dirs: list[str] = []
        self._dir_lock = threading.Lock()
        self._session_workspace: str | None = None

        self._sdk_session = SdkSession()
        self._backend = resolve_cursor_backend(api_key=self.api_key)

    @property
    def backend(self) -> str:
        """Effective transport: ``sdk`` (cursor-sdk) or ``cli`` (cursor-agent)."""
        return getattr(self, "_backend", "cli")

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        sdk_session = getattr(self, "_sdk_session", None)
        if sdk_session is not None:
            try:
                sdk_session.close()
            except Exception:
                pass
        self._context_high_water = 0
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        with self._dir_lock:
            dirs, self._ephemeral_dirs = self._ephemeral_dirs, []
            self._session_workspace = None
        for d in dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        try:
            user_msg_count = sum(
                1 for m in (messages or []) if (m or {}).get("role") == "user"
            )
        except Exception:
            user_msg_count = self._last_user_msg_count
        is_new_user_turn = user_msg_count > self._last_user_msg_count
        if is_new_user_turn:
            self._context_high_water = 0
        self._last_user_msg_count = user_msg_count

        try:
            from agent.model_metadata import estimate_request_tokens_rough
            self._last_messages_estimate = estimate_request_tokens_rough(
                messages or [], tools=tools or None
            )
        except Exception:
            self._last_messages_estimate = 0

        if self._last_messages_estimate > self._context_high_water:
            self._context_high_water = self._last_messages_estimate
        if callable(self._context_estimate_callback) and self._last_messages_estimate > 0:
            try:
                self._context_estimate_callback(
                    self._last_messages_estimate, reset=is_new_user_turn
                )
            except TypeError:
                try:
                    self._context_estimate_callback(self._last_messages_estimate)
                except Exception:
                    pass
            except Exception:
                pass

        prompt_text = format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )

        if timeout is None:
            effective_timeout = self._timeout_seconds
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else self._timeout_seconds

        chosen_model = (model or DEFAULT_CURSOR_MODEL).strip() or DEFAULT_CURSOR_MODEL

        accumulator = self._run_prompt(
            prompt_text=prompt_text,
            model=chosen_model,
            timeout_seconds=effective_timeout,
        )

        assistant_text = accumulator.synthesis_text()
        reasoning_text = accumulator.assembled_reasoning() or None

        if accumulator.is_error:
            raise RuntimeError(
                f"cursor-agent reported an error: {accumulator.error_message or assistant_text}"
            )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(assistant_text)
        cursor_internal_tools = [evt.to_public_dict() for evt in accumulator.tool_events]
        cur_estimate = getattr(self, "_last_messages_estimate", 0) or 0
        cursor_per_round = self._estimate_per_round_context(accumulator)
        new_high = max(self._context_high_water, cur_estimate, cursor_per_round)
        self._context_high_water = new_high
        accumulator.messages_estimate = new_high
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text,
            reasoning_content=reasoning_text,
            reasoning_details=None,
            cursor_internal_tools=cursor_internal_tools,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(
            message=assistant_message,
            finish_reason=finish_reason,
            index=0,
        )
        return SimpleNamespace(
            choices=[choice],
            usage=accumulator.openai_usage(),
            model=chosen_model,
            id=accumulator.request_id or f"cursor-{accumulator.session_id}",
            object="chat.completion",
            cursor_internal_tools=cursor_internal_tools,
        )

    def _build_argv(self, *, model: str, workspace: str) -> list[str]:
        from agent.cursor.cli_backend import build_argv

        return build_argv(
            command=self._command,
            mode=self._mode,
            model=model,
            workspace=workspace,
            api_key=self.api_key,
            extra_args=self._extra_args,
        )

    def _allocate_workspace(self) -> tuple[str, bool]:
        if self._workspace:
            try:
                Path(self._workspace).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return self._workspace, False
        with self._dir_lock:
            if self._session_workspace is None:
                tmp = tempfile.mkdtemp(prefix="hermes-cursor-")
                self._session_workspace = tmp
                self._ephemeral_dirs.append(tmp)
            return self._session_workspace, True

    def _run_prompt(
        self,
        *,
        prompt_text: str,
        model: str,
        timeout_seconds: float,
    ) -> CursorTurnAccumulator:
        backend = getattr(self, "_backend", "cli")
        if backend == "sdk" and self.api_key:
            workspace, _ephemeral = self._allocate_workspace()
            try:
                return run_prompt_via_sdk(
                    prompt_text=prompt_text,
                    model=model,
                    api_key=self.api_key,
                    workspace=workspace,
                    mode=self._mode,
                    timeout_seconds=timeout_seconds,
                    on_tool_event=self._build_tool_event_bridge(),
                    on_text_event=self._build_text_event_bridge(),
                    sdk_session=self._sdk_session,
                )
            except RuntimeError as exc:
                forced = os.getenv("HERMES_CURSOR_BACKEND", "").strip().lower()
                if forced == "sdk":
                    raise
                lowered = str(exc).lower()
                if "sdk" in lowered and (
                    "preview" in lowered
                    or "not enabled" in lowered
                    or "not installed" in lowered
                ):
                    self._backend = "cli"
                    return self._run_prompt_cli(
                        prompt_text=prompt_text,
                        model=model,
                        timeout_seconds=timeout_seconds,
                    )
                raise
        return self._run_prompt_cli(
            prompt_text=prompt_text,
            model=model,
            timeout_seconds=timeout_seconds,
        )

    def _run_prompt_cli(
        self,
        *,
        prompt_text: str,
        model: str,
        timeout_seconds: float,
    ) -> CursorTurnAccumulator:
        workspace, _ephemeral = self._allocate_workspace()

        def _set_active(proc: subprocess.Popen[str] | None) -> None:
            with self._active_process_lock:
                self._active_process = proc

        return run_prompt_cli(
            command=self._command,
            mode=self._mode,
            model=model,
            workspace=workspace,
            api_key=self.api_key,
            extra_args=self._extra_args,
            prompt_text=prompt_text,
            timeout_seconds=timeout_seconds,
            on_tool_event=self._build_tool_event_bridge(),
            on_text_event=self._build_text_event_bridge(),
            set_active_process=_set_active,
            terminate_active_proc=self._terminate_active_proc,
            mark_open=lambda: setattr(self, "is_closed", False),
        )

    def _estimate_per_round_context(self, accumulator: CursorTurnAccumulator) -> int:
        input_tokens_raw = int(accumulator.usage.get("inputTokens", 0))
        cache_read_raw = int(accumulator.usage.get("cacheReadTokens", 0))
        rounds = max(len(accumulator.tool_events) + 1, 1)
        per_round_input = input_tokens_raw // rounds if rounds > 0 else input_tokens_raw
        per_round_cache = cache_read_raw // rounds if rounds > 0 else cache_read_raw
        return per_round_cache + per_round_input

    def reset_context_baseline(self) -> None:
        self._context_high_water = 0

    def _build_text_event_bridge(self) -> Any:
        cb = self._tool_progress_callback
        if cb is None:
            return None

        def _bridge(text: str) -> None:
            try:
                preview = text.strip().splitlines()[0] if text else ""
                if len(preview) > 240:
                    preview = preview[:237] + "..."
                if not preview:
                    return
                cb("tool.started", "narrate", preview, {"text": text})
                cb(
                    "tool.completed", "narrate", None, None,
                    duration=0.0, is_error=False, result=text,
                )
            except Exception:
                pass

        return _bridge

    def _build_tool_event_bridge(self) -> Any:
        cb = self._tool_progress_callback
        if cb is None:
            return None

        def _bridge(stage: str, evt: CursorToolEvent) -> None:
            try:
                if stage == "started":
                    preview = build_cursor_tool_preview(evt)
                    cb("tool.started", evt.name, preview, evt.args)
                elif stage == "completed":
                    if (
                        evt.lines_added is not None
                        or evt.lines_removed is not None
                    ) and isinstance(evt.args, dict):
                        evt.args["_diff_stats"] = {
                            "added": evt.lines_added or 0,
                            "removed": evt.lines_removed or 0,
                        }
                        if evt.diff_string:
                            evt.args["_diff_string"] = evt.diff_string
                    cb(
                        "tool.completed",
                        evt.name,
                        None,
                        None,
                        duration=evt.duration_ms / 1000.0,
                        is_error=evt.is_error,
                        result=evt.result_text,
                    )
            except Exception:
                try:
                    cb(f"tool.{stage}", evt.name, evt.result_text or "", evt.args)
                except Exception:
                    pass

        return _bridge

    def _terminate_active_proc(self, proc: subprocess.Popen[str]) -> None:
        with self._active_process_lock:
            current = self._active_process
            if current is proc:
                self._active_process = None
        if proc.poll() is not None:
            return
        try:
            proc.wait(timeout=0.7)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=1.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def whoami(self) -> dict[str, Any]:
        try:
            out = subprocess.check_output(
                [self._command, "status"],
                text=True,
                timeout=10,
                env=build_subprocess_env(self.api_key),
            )
        except Exception:
            return {}
        info: dict[str, Any] = {"raw": out.strip()}
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("✓ Logged in as "):
                info["email"] = line.removeprefix("✓ Logged in as ").strip()
                info["authenticated"] = True
        return info


__all__ = ["CursorAgentClient"]
