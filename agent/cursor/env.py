"""Environment and subprocess helpers for Cursor backends."""

from __future__ import annotations

import os
import shlex

from agent.cursor.constants import (
    DEFAULT_CURSOR_COMMAND,
    DEFAULT_CURSOR_MODE,
    _VALID_CURSOR_MODES,
)


def resolve_command() -> str:
    return (
        os.getenv("HERMES_CURSOR_COMMAND", "").strip()
        or os.getenv("CURSOR_AGENT_PATH", "").strip()
        or DEFAULT_CURSOR_COMMAND
    )


def resolve_extra_args() -> list[str]:
    raw = os.getenv("HERMES_CURSOR_ARGS", "").strip()
    if not raw:
        return []
    return shlex.split(raw)


def resolve_mode() -> str:
    mode = os.getenv("HERMES_CURSOR_MODE", "").strip().lower() or DEFAULT_CURSOR_MODE
    if mode not in _VALID_CURSOR_MODES:
        mode = DEFAULT_CURSOR_MODE
    return mode


def resolve_workspace_override() -> str:
    return os.getenv("HERMES_CURSOR_WORKSPACE", "").strip()


def resolve_home_dir() -> str:
    """Pick a stable HOME for the child process."""
    try:
        from hermes_constants import get_subprocess_home

        profile_home = get_subprocess_home()
        if profile_home:
            return profile_home
    except Exception:
        pass

    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()
        if resolved:
            return resolved
    except Exception:
        pass

    return "/tmp"


def build_subprocess_env(api_key: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = resolve_home_dir()
    if api_key:
        env["CURSOR_API_KEY"] = api_key
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    return env


__all__ = [
    "build_subprocess_env",
    "resolve_command",
    "resolve_extra_args",
    "resolve_home_dir",
    "resolve_mode",
    "resolve_workspace_override",
]
