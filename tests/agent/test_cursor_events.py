"""Tests for typed Cursor turn events and CursorTurnAccumulator."""

from __future__ import annotations

import unittest

from agent.cursor.accumulator import CursorTurnAccumulator
from agent.cursor.events import (
    AssistantTextEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    TurnResultEvent,
    stream_json_dict_to_events,
)


class TestStreamJsonConversion(unittest.TestCase):
    def test_assistant_text(self):
        events = stream_json_dict_to_events({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hello"}]},
        })
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], AssistantTextEvent)
        self.assertEqual(events[0].text, "hello")

    def test_tool_started_and_completed(self):
        started = stream_json_dict_to_events({
            "type": "tool_call",
            "subtype": "started",
            "call_id": "c1",
            "tool_call": {"shellToolCall": {"args": {"command": "ls"}}},
        })
        self.assertIsInstance(started[0], ToolStartedEvent)

        completed = stream_json_dict_to_events({
            "type": "tool_call",
            "subtype": "completed",
            "call_id": "c1",
            "tool_call": {
                "shellToolCall": {
                    "args": {"command": "ls"},
                    "result": {"success": {"stdout": "ok"}},
                }
            },
        })
        self.assertIsInstance(completed[0], ToolCompletedEvent)


class TestCursorTurnAccumulator(unittest.TestCase):
    def test_synthesis_after_tools(self):
        acc = CursorTurnAccumulator()
        acc.feed(AssistantTextEvent(text="Searching…"))
        acc.feed(ToolStartedEvent(call_id="c1", envelope_key="grepToolCall", args={}))
        acc.feed(ToolCompletedEvent(
            call_id="c1",
            envelope_key="grepToolCall",
            args={},
            result_payload={"result": {"success": {}}},
        ))
        acc.feed(AssistantTextEvent(text="Found it."))
        acc.feed(TurnResultEvent(is_error=False, result_text="Found it."))
        self.assertEqual(acc.synthesis_text(), "Found it.")
        self.assertIn("Searching", acc.narration_text())

    def test_legacy_dict_feed(self):
        acc = CursorTurnAccumulator()
        acc.feed({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "plain"}]},
        })
        acc.feed({
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "plain",
            "usage": {"inputTokens": 10, "outputTokens": 2},
        })
        self.assertEqual(acc.synthesis_text(), "plain")
        self.assertTrue(acc.terminal)


if __name__ == "__main__":
    unittest.main()
