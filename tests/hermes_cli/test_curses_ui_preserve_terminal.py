"""Tests for curses_ui.preserve_terminal_state — save/restore TTY attrs.

Regression target: "4^M" bug where pressing Enter after a curses menu
echoed the literal CR and input() never returned because ICRNL/ICANON
were left off by curses.wrapper() on certain emulators.

The fix saves termios attrs on entry to the curses operation and
restores them on exit. These tests verify the save/restore contract
without requiring an actual curses screen.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestPreserveTerminalStateTTY:
    """Save/restore behavior when stdin is a real TTY."""

    def test_restores_attrs_after_clean_exit(self):
        from hermes_cli.curses_ui import preserve_terminal_state

        saved_attrs = [0, 1, 2, 3, 4, 5, [b"x"] * 32]

        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(sys.stdin, "fileno", return_value=0), \
             patch("termios.tcgetattr", return_value=saved_attrs) as mock_get, \
             patch("termios.tcsetattr") as mock_set:
            with preserve_terminal_state():
                pass

            mock_get.assert_called_once_with(sys.stdin.fileno())
            mock_set.assert_called_once()
            # Verify it restored with the saved attrs (not some other value)
            args = mock_set.call_args[0]
            assert args[2] == saved_attrs

    def test_restores_attrs_when_body_raises(self):
        from hermes_cli.curses_ui import preserve_terminal_state

        saved_attrs = [0, 1, 2, 3, 4, 5, [b"x"] * 32]

        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(sys.stdin, "fileno", return_value=0), \
             patch("termios.tcgetattr", return_value=saved_attrs), \
             patch("termios.tcsetattr") as mock_set:
            with pytest.raises(RuntimeError, match="boom"):
                with preserve_terminal_state():
                    raise RuntimeError("boom")

            # Restore must still fire despite the exception
            mock_set.assert_called_once()
            assert mock_set.call_args[0][2] == saved_attrs

    def test_uses_tcsadrain_for_restore(self):
        """TCSADRAIN waits for pending output, avoiding corrupted terminal paint."""
        import termios
        from hermes_cli.curses_ui import preserve_terminal_state

        saved_attrs = [0, 1, 2, 3, 4, 5, [b"x"] * 32]

        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(sys.stdin, "fileno", return_value=0), \
             patch("termios.tcgetattr", return_value=saved_attrs), \
             patch("termios.tcsetattr") as mock_set:
            with preserve_terminal_state():
                pass

            when = mock_set.call_args[0][1]
            assert when == termios.TCSADRAIN


class TestPreserveTerminalStateNonTTY:
    """No-op behavior when stdin is not a TTY (pipes, redirects, Windows)."""

    def test_is_noop_when_stdin_not_tty(self):
        from hermes_cli.curses_ui import preserve_terminal_state

        with patch.object(sys.stdin, "isatty", return_value=False), \
             patch("termios.tcgetattr") as mock_get, \
             patch("termios.tcsetattr") as mock_set:
            with preserve_terminal_state():
                pass

            mock_get.assert_not_called()
            mock_set.assert_not_called()

    def test_yields_control_to_body_even_when_noop(self):
        from hermes_cli.curses_ui import preserve_terminal_state

        entered = False
        with patch.object(sys.stdin, "isatty", return_value=False):
            with preserve_terminal_state():
                entered = True
        assert entered is True


class TestPreserveTerminalStateFailureModes:
    """Graceful degradation when termios raises."""

    def test_tcgetattr_failure_is_silent(self):
        """If we can't read current attrs, proceed without save/restore."""
        from hermes_cli.curses_ui import preserve_terminal_state

        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch("termios.tcgetattr", side_effect=OSError("ioctl failed")), \
             patch("termios.tcsetattr") as mock_set:
            # Must not raise
            with preserve_terminal_state():
                pass
            # Nothing to restore = no set call
            mock_set.assert_not_called()

    def test_tcsetattr_failure_is_logged_not_raised(self):
        """Restore failures are debug-logged, never propagated."""
        from hermes_cli.curses_ui import preserve_terminal_state

        saved_attrs = [0, 1, 2, 3, 4, 5, [b"x"] * 32]

        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(sys.stdin, "fileno", return_value=0), \
             patch("termios.tcgetattr", return_value=saved_attrs), \
             patch("termios.tcsetattr", side_effect=OSError("restore failed")):
            # Must not raise; failure absorbed
            with preserve_terminal_state():
                pass

    def test_termios_import_failure_is_silent(self):
        """Non-POSIX platforms where termios import fails must not crash."""
        from hermes_cli.curses_ui import preserve_terminal_state
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "termios":
                raise ImportError("no termios on this platform")
            return real_import(name, *args, **kwargs)

        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(builtins, "__import__", side_effect=fake_import):
            # Must not raise
            with preserve_terminal_state():
                pass
