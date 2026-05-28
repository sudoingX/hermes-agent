"""CLI subprocess backend for the Cursor provider."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable

from agent.cursor.accumulator import CursorTurnAccumulator
from agent.cursor.constants import _CURSOR_CLI_MODES
from agent.cursor.env import build_subprocess_env
from agent.cursor.events import stream_json_dict_to_events
from agent.redact import redact_sensitive_text


def build_argv(
    *,
    command: str,
    mode: str,
    model: str,
    workspace: str,
    api_key: str | None,
    extra_args: list[str],
) -> list[str]:
    argv = [
        command,
        "-p",
        "--output-format",
        "stream-json",
    ]
    if mode in _CURSOR_CLI_MODES:
        argv.extend(["--mode", mode])
    argv.extend(
        [
            "--model",
            model,
            "--workspace",
            workspace,
            "--force",
            "--trust",
        ]
    )
    if api_key:
        argv.extend(["--api-key", api_key])
    argv.extend(extra_args)
    return argv


def run_prompt_cli(
    *,
    command: str,
    mode: str,
    model: str,
    workspace: str,
    api_key: str | None,
    extra_args: list[str],
    prompt_text: str,
    timeout_seconds: float,
    on_tool_event: Any,
    on_text_event: Any,
    set_active_process: Callable[[subprocess.Popen[str] | None], None],
    terminate_active_proc: Callable[[subprocess.Popen[str]], None],
    mark_open: Callable[[], None],
) -> CursorTurnAccumulator:
    """Execute one Hermes turn via cursor-agent subprocess."""
    argv = build_argv(
        command=command,
        mode=mode,
        model=model,
        workspace=workspace,
        api_key=api_key,
        extra_args=extra_args,
    )

    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=workspace,
            env=build_subprocess_env(api_key),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Could not start Cursor Agent CLI '{command}'. "
            "Install Cursor CLI (https://cursor.com/dashboard/integrations) "
            "or set HERMES_CURSOR_COMMAND / CURSOR_AGENT_PATH."
        ) from exc

    if proc.stdin is None or proc.stdout is None:
        proc.kill()
        raise RuntimeError("cursor-agent process did not expose stdin/stdout pipes.")

    mark_open()
    set_active_process(proc)

    try:
        stderr_tail: deque[str] = deque(maxlen=80)
        inbox: queue.Queue[dict[str, Any]] = queue.Queue()

        def _stderr_reader_early() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        err_thread = threading.Thread(target=_stderr_reader_early, daemon=True)
        err_thread.start()

        stdin_error: BaseException | None = None
        try:
            proc.stdin.write(prompt_text)
            proc.stdin.flush()
        except BrokenPipeError as exc:
            stdin_error = exc
        except Exception as exc:  # pragma: no cover - defensive
            stdin_error = exc
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

        if stdin_error is not None:
            try:
                proc.wait(timeout=3)
            except Exception:
                pass
            err_thread.join(timeout=1)
            exit_code = getattr(proc, "returncode", None)
            if exit_code is None:
                try:
                    exit_code = proc.poll()
                except Exception:
                    exit_code = None
            stderr_text = "\n".join(stderr_tail).strip()
            redacted = redact_sensitive_text(stderr_text, force=True) if stderr_text else ""
            detail = f" stderr: {redacted}" if redacted else ""
            raise RuntimeError(
                "cursor-agent closed stdin before reading the prompt "
                f"(exit {exit_code}).{detail}"
            ) from stdin_error

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    stderr_tail.append("[stdout-non-json] " + line)

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        out_thread.start()

        accumulator = CursorTurnAccumulator(
            on_tool_event=on_tool_event,
            on_text_event=on_text_event,
        )
        idle_seconds = float(timeout_seconds)
        deadline = time.monotonic() + idle_seconds

        while not accumulator.terminal:
            if time.monotonic() >= deadline:
                terminate_active_proc(proc)
                raise TimeoutError(
                    f"cursor-agent emitted no events for {idle_seconds:.0f}s; "
                    f"presumed hung. Set HERMES_CURSOR_TIMEOUT_SECONDS to "
                    f"increase the idle threshold."
                )
            if proc.poll() is not None and inbox.empty():
                break
            try:
                event = inbox.get(timeout=0.25)
            except queue.Empty:
                continue
            deadline = time.monotonic() + idle_seconds
            try:
                for typed in stream_json_dict_to_events(event):
                    accumulator.feed(typed)
            except Exception:
                continue

        if not accumulator.terminal:
            stderr_text = "\n".join(stderr_tail).strip()
            redacted = redact_sensitive_text(stderr_text, force=True) if stderr_text else ""
            raise RuntimeError(
                "cursor-agent exited before emitting a terminal result. "
                + (f"stderr tail:\n{redacted}" if redacted else "(no stderr)")
            )

        return accumulator
    finally:
        terminate_active_proc(proc)
        set_active_process(None)


__all__ = ["build_argv", "run_prompt_cli"]
