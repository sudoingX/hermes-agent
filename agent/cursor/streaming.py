"""OpenAI-style streaming shims for the Cursor provider facade."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.cursor.client import CursorAgentClient


class CursorChatCompletions:
    def __init__(self, client: "CursorAgentClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        stream_requested = bool(kwargs.pop("stream", False))
        kwargs.pop("stream_options", None)
        response = self._client._create_chat_completion(**kwargs)
        if not stream_requested:
            return response
        return synthesise_stream_chunks(response)


class CursorChatNamespace:
    def __init__(self, client: "CursorAgentClient"):
        self.completions = CursorChatCompletions(client)


def synthesise_stream_chunks(response: Any):
    """Yield OpenAI-style streaming chunks from a non-streaming response."""
    try:
        choice = response.choices[0]
    except Exception:
        return

    message = getattr(choice, "message", None)
    if message is None:
        return

    role = "assistant"
    content = getattr(message, "content", "") or ""
    tool_calls = getattr(message, "tool_calls", None) or []
    reasoning = getattr(message, "reasoning", None)
    reasoning_content = getattr(message, "reasoning_content", None)
    finish_reason = getattr(choice, "finish_reason", "stop")
    model = getattr(response, "model", "cursor")
    usage = getattr(response, "usage", None)

    if reasoning_content:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=role,
                        content=None,
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=reasoning_content,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            model=model,
            usage=None,
        )
    elif reasoning:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=role,
                        content=None,
                        tool_calls=None,
                        reasoning=reasoning,
                        reasoning_content=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            model=model,
            usage=None,
        )

    if content:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=role,
                        content=content,
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            model=model,
            usage=None,
        )

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role=role,
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=i,
                                    id=getattr(tc, "id", f"call_{i}"),
                                    type="function",
                                    function=SimpleNamespace(
                                        name=getattr(tc.function, "name", ""),
                                        arguments=getattr(tc.function, "arguments", ""),
                                    ),
                                )
                            ],
                            reasoning=None,
                            reasoning_content=None,
                        ),
                        finish_reason=None,
                        index=0,
                    )
                ],
                model=model,
                usage=None,
            )

    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    role=None,
                    content=None,
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                ),
                finish_reason=finish_reason,
                index=0,
            )
        ],
        model=model,
        usage=usage,
    )


__all__ = [
    "CursorChatCompletions",
    "CursorChatNamespace",
    "synthesise_stream_chunks",
]
