"""Cursor provider profile — CLI or SDK transport.

Cursor doesn't expose a chat completions endpoint; it ships an agent. Hermes
routes requests through ``cursor-agent`` (CLI subprocess) or ``cursor-sdk``
(Python SDK) depending on ``HERMES_CURSOR_BACKEND`` and ``CURSOR_API_KEY``.

See ``agent/cursor/`` for the runtime client and
``docs/cursor_architecture.md`` for the design.
"""

from __future__ import annotations

import shutil
import subprocess

from providers import register_provider
from providers.base import ProviderProfile


class CursorProfile(ProviderProfile):
    """Cursor — external process via ``cursor-agent`` CLI."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Shell out to ``cursor-agent --list-models`` for the live catalog.

        Returns ``None`` if the CLI is missing or not authenticated. Callers
        are expected to fall back to the static ``_PROVIDER_MODELS["cursor"]``
        list in ``hermes_cli/models.py``.
        """
        command = shutil.which("cursor-agent") or "cursor-agent"
        try:
            out = subprocess.check_output(
                [command, "--list-models"],
                text=True,
                timeout=timeout,
            )
        except Exception:
            return None
        ids: list[str] = []
        for raw in out.splitlines():
            line = raw.strip()
            if not line:
                continue
            if " - " in line:
                model_id = line.split(" - ", 1)[0].strip()
                # Strip "(current)" / "(default)" decoration the CLI sometimes appends
                model_id = model_id.split()[0]
                if model_id:
                    ids.append(model_id)
        return ids or None


cursor = CursorProfile(
    name="cursor",
    aliases=("cursor-agent", "cursor-cli", "cursor-sub", "cursor-subscription"),
    display_name="Cursor",
    description="Cursor (100+ models, subscription)",
    signup_url="https://cursor.com/dashboard/integrations",
    api_mode="chat_completions",  # external-process routing handled in client
    env_vars=("CURSOR_API_KEY",),
    base_url="cursor://agent",  # marker URL; never dereferenced
    auth_type="external_process",
    fallback_models=(
        "auto",
        "composer-2.5",
        "composer-2.5-fast",
        "composer-2",
        "composer-2-fast",
    ),
    supports_health_check=False,
)

register_provider(cursor)
