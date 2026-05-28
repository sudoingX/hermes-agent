"""Unit tests for cursor-sdk backend selection and typed event translation."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.cursor.backend import cursor_sdk_installed, resolve_cursor_backend
from agent.cursor.events import (
    AssistantTextEvent,
    ToolStartedEvent,
    interaction_update_to_events,
    sdk_message_to_events,
)
from agent.cursor.sdk_backend import SdkSession, run_prompt_via_sdk


class TestBackendResolution(unittest.TestCase):
    def setUp(self):
        self._saved = {
            k: os.environ.get(k)
            for k in ("HERMES_CURSOR_BACKEND", "CURSOR_API_KEY")
        }
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_auto_without_key_uses_cli(self):
        with patch("agent.cursor.backend.cursor_sdk_installed", return_value=True):
            self.assertEqual(resolve_cursor_backend(api_key=None), "cli")

    def test_auto_with_key_uses_sdk_when_installed(self):
        with patch("agent.cursor.backend.cursor_sdk_installed", return_value=True):
            self.assertEqual(
                resolve_cursor_backend(api_key="crsr_real_key_12345"),
                "sdk",
            )

    def test_auto_with_sentinel_uses_cli(self):
        with patch("agent.cursor.backend.cursor_sdk_installed", return_value=True):
            self.assertEqual(
                resolve_cursor_backend(api_key="cursor-agent-login"),
                "cli",
            )

    def test_forced_cli(self):
        os.environ["HERMES_CURSOR_BACKEND"] = "cli"
        self.assertEqual(resolve_cursor_backend(api_key="crsr_x"), "cli")

    def test_forced_sdk_requires_package(self):
        os.environ["HERMES_CURSOR_BACKEND"] = "sdk"
        os.environ["CURSOR_API_KEY"] = "crsr_x"
        with patch("agent.cursor.backend.cursor_sdk_installed", return_value=False), patch(
            "agent.cursor.backend.ensure_cursor_sdk",
            side_effect=RuntimeError("cursor-sdk is not installed"),
        ):
            with self.assertRaises(RuntimeError):
                resolve_cursor_backend(api_key="crsr_x")


class TestEventTranslation(unittest.TestCase):
    def test_assistant_sdk_message(self):
        msg = SimpleNamespace(
            type="assistant",
            message=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="hello")]
            ),
        )
        events = sdk_message_to_events(msg)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], AssistantTextEvent)
        self.assertEqual(events[0].text, "hello")

    def test_tool_call_started(self):
        msg = SimpleNamespace(
            type="tool_call",
            call_id="c1",
            name="shell",
            status="running",
            args={"command": "ls"},
            result=None,
        )
        events = sdk_message_to_events(msg)
        self.assertIsInstance(events[0], ToolStartedEvent)
        self.assertEqual(events[0].envelope_key, "shellToolCall")

    def test_text_delta_interaction(self):
        update = SimpleNamespace(type="text-delta", text="partial ")
        events = interaction_update_to_events(update)
        self.assertIsInstance(events[0], AssistantTextEvent)
        self.assertEqual(events[0].text, "partial ")


class TestRunPromptViaSdk(unittest.TestCase):
    def test_streams_events_into_accumulator(self):
        sdk_session = SdkSession()
        fake_agent = MagicMock()
        fake_run = MagicMock()
        fake_result = SimpleNamespace(status="finished", result="done", id="r1", duration_ms=10)

        def _events():
            yield SimpleNamespace(
                kind="sdk_message",
                sdk_message=SimpleNamespace(
                    type="assistant",
                    message=SimpleNamespace(
                        content=[SimpleNamespace(type="text", text="hi")]
                    ),
                ),
                interaction_update=None,
                result=None,
            )
            yield SimpleNamespace(
                kind="result",
                sdk_message=None,
                interaction_update=None,
                result={
                    "status": "finished",
                    "result": "done",
                    "runId": "r1",
                    "durationMs": 10,
                    "usage": {"inputTokens": 10, "outputTokens": 2},
                },
            )

        fake_run.events.side_effect = _events
        fake_run.wait.return_value = fake_result
        fake_run.supports.return_value = True
        fake_agent.send.return_value = fake_run

        with patch("cursor_sdk.Agent.create", return_value=fake_agent), patch.object(
            SdkSession,
            "get_client",
            return_value=MagicMock(),
        ), patch("agent.cursor.sdk_backend.ensure_cursor_sdk"):
            acc = run_prompt_via_sdk(
                prompt_text="ping",
                model="composer-2.5",
                api_key="crsr_test_key",
                workspace="/tmp/ws",
                mode="agent",
                timeout_seconds=30,
                on_tool_event=None,
                on_text_event=None,
                sdk_session=sdk_session,
            )
        self.assertFalse(acc.is_error)
        fake_agent.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
