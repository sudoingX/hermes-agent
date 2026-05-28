"""cursor-sdk backend for the Cursor provider."""

from __future__ import annotations

import time
from typing import Any

from agent.cursor.accumulator import CursorTurnAccumulator
from agent.cursor.backend import ensure_cursor_sdk, map_hermes_mode_to_sdk
from agent.cursor.events import TurnResultEvent, run_stream_event_to_events


def _finalize_terminal_result(accumulator: CursorTurnAccumulator, result: Any) -> None:
    if accumulator.terminal:
        return
    status = str(getattr(result, "status", "") or "").lower()
    is_error = status in {"error", "failed", "cancelled", "canceled"}
    accumulator.feed(TurnResultEvent(
        is_error=is_error,
        result_text=str(getattr(result, "result", "") or ""),
        request_id=str(getattr(result, "id", "") or ""),
        duration_ms=int(getattr(result, "duration_ms", 0) or 0),
        usage={},
        error_message=str(getattr(result, "result", "") or "") if is_error else "",
    ))


class SdkSession:
    """Reused SDK bridge client scoped to one Hermes chat session."""

    def __init__(self) -> None:
        self._client: Any = None
        self._workspace: str | None = None

    def get_client(self, *, workspace: str, api_key: str) -> Any:
        ensure_cursor_sdk(prompt=False)
        from cursor_sdk import CursorClient

        if self._client is not None and self._workspace == workspace:
            return self._client
        self.close()
        self._client = CursorClient.launch_bridge(
            workspace=workspace,
            allow_api_key_env_fallback=False,
        )
        self._workspace = workspace
        return self._client

    def close(self) -> None:
        client = self._client
        self._client = None
        self._workspace = None
        if client is None:
            return
        try:
            client.close()
        except Exception:
            pass


def run_prompt_via_sdk(
    *,
    prompt_text: str,
    model: str,
    api_key: str,
    workspace: str,
    mode: str,
    timeout_seconds: float,
    on_tool_event: Any,
    on_text_event: Any,
    sdk_session: SdkSession,
) -> CursorTurnAccumulator:
    """Execute one Hermes turn via cursor-sdk; return a populated accumulator."""
    ensure_cursor_sdk(prompt=False)
    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    from cursor_sdk.errors import CursorAgentError, IntegrationNotConnectedError

    sdk_mode = map_hermes_mode_to_sdk(mode)
    client = sdk_session.get_client(workspace=workspace, api_key=api_key)
    options = AgentOptions(
        model=model,
        api_key=api_key,
        mode=sdk_mode,
        local=LocalAgentOptions(cwd=workspace),
    )
    accumulator = CursorTurnAccumulator(
        on_tool_event=on_tool_event,
        on_text_event=on_text_event,
    )
    idle_seconds = float(timeout_seconds)
    deadline = time.monotonic() + idle_seconds

    agent = Agent.create(options, client=client)
    try:
        run = agent.send(prompt_text)
        for event in run.events():
            deadline = time.monotonic() + idle_seconds
            for typed in run_stream_event_to_events(event):
                accumulator.feed(typed)
                if accumulator.terminal:
                    break
            if accumulator.terminal:
                break
            if time.monotonic() >= deadline:
                if run.supports("cancel"):
                    run.cancel()
                raise TimeoutError(
                    f"cursor-sdk emitted no events for {idle_seconds:.0f}s; "
                    f"presumed hung. Set HERMES_CURSOR_TIMEOUT_SECONDS to "
                    f"increase the idle threshold."
                )

        result = run.wait()
        if str(getattr(result, "status", "") or "").lower() == "error":
            accumulator.feed(TurnResultEvent(
                is_error=True,
                result_text=str(getattr(result, "result", "") or "cursor-sdk run failed"),
                request_id=str(getattr(result, "id", "") or ""),
                duration_ms=int(getattr(result, "duration_ms", 0) or 0),
                usage={},
                error_message=str(getattr(result, "result", "") or "cursor-sdk run failed"),
            ))
        elif not accumulator.terminal:
            _finalize_terminal_result(accumulator, result)

        if accumulator.is_error:
            raise RuntimeError(
                f"cursor-sdk reported an error: {accumulator.error_message or result.result}"
            )
        return accumulator
    except IntegrationNotConnectedError as exc:
        raise RuntimeError(
            "cursor-sdk access is not enabled for this account "
            "(sdk_python_preview_access). Set HERMES_CURSOR_BACKEND=cli or "
            "generate a User API Key once SDK access is granted."
        ) from exc
    except CursorAgentError:
        raise
    finally:
        try:
            agent.close()
        except Exception:
            pass


# Backward-compat alias.
_SdkSession = SdkSession


__all__ = [
    "SdkSession",
    "_SdkSession",
    "run_prompt_via_sdk",
]
