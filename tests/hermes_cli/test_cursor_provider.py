"""Tests for the Cursor provider registration across all four registries.

Covers:
- ``providers/`` plugin discovery (``ProviderProfile``)
- ``hermes_cli.providers`` HERMES_OVERLAYS + ALIASES + labels
- ``hermes_cli.auth`` PROVIDER_REGISTRY + alias resolver
- ``hermes_cli.models`` ProviderEntry + ``_PROVIDER_MODELS`` snapshot + aliases
- ``resolve_external_process_provider_credentials`` for cursor
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class CursorComposerContextWindowTests(unittest.TestCase):
    """Composer family must report cursor's actual 200K cap, not 256K.

    Cursor docs pin Composer 2 and Composer 2.5 (both fast and standard)
    to a 200K context window (the base model supports 256K but cursor
    truncates). Status-bar % and compression thresholds depend on this
    being accurate — otherwise users see 67K/256K (26%) when in reality
    they're at 67K/200K (33%) and 12K closer to the actual ceiling.

    Regression: prior to 186bf25c the composer ids fell through to the
    256K DEFAULT_FALLBACK_CONTEXT and inflated the usable window by 56K.
    """

    def test_composer_25_fast_is_200k(self) -> None:
        from agent.model_metadata import get_model_context_length

        ctx = get_model_context_length(
            "composer-2.5-fast", "cursor://agent", "", None, "cursor"
        )
        self.assertEqual(ctx, 200_000)

    def test_composer_25_is_200k(self) -> None:
        from agent.model_metadata import get_model_context_length

        ctx = get_model_context_length(
            "composer-2.5", "cursor://agent", "", None, "cursor"
        )
        self.assertEqual(ctx, 200_000)

    def test_composer_2_family_is_200k(self) -> None:
        from agent.model_metadata import get_model_context_length

        for model in ("composer-2", "composer-2-fast"):
            with self.subTest(model=model):
                ctx = get_model_context_length(
                    model, "cursor://agent", "", None, "cursor"
                )
                self.assertEqual(ctx, 200_000)


class CursorProviderRegistryTests(unittest.TestCase):
    # ---- providers/ plugin profile ----

    def test_plugin_profile_registered(self) -> None:
        from providers import get_provider_profile

        profile = get_provider_profile("cursor")
        self.assertIsNotNone(profile, "cursor plugin profile not discovered")
        self.assertEqual(profile.name, "cursor")
        self.assertEqual(profile.api_mode, "chat_completions")
        self.assertEqual(profile.auth_type, "external_process")
        self.assertEqual(profile.base_url, "cursor://agent")
        self.assertIn("CURSOR_API_KEY", profile.env_vars)
        self.assertEqual(profile.supports_health_check, False)
        # Fallback model list must include the new 2.5 family.
        self.assertIn("composer-2.5", profile.fallback_models)
        self.assertIn("composer-2.5-fast", profile.fallback_models)

    def test_plugin_profile_aliases_resolve(self) -> None:
        from providers import get_provider_profile

        for alias in ("cursor-agent", "cursor-cli", "cursor-sub", "cursor-subscription"):
            with self.subTest(alias=alias):
                self.assertEqual(get_provider_profile(alias).name, "cursor")

    # ---- hermes_cli.providers HERMES_OVERLAYS ----

    def test_hermes_overlay_present(self) -> None:
        from hermes_cli.providers import get_provider, normalize_provider, get_label

        pdef = get_provider("cursor")
        self.assertIsNotNone(pdef)
        self.assertEqual(pdef.id, "cursor")
        self.assertEqual(pdef.auth_type, "external_process")
        self.assertEqual(pdef.base_url, "cursor://agent")
        self.assertIn("CURSOR_API_KEY", pdef.api_key_env_vars)

        # Aliases
        for alias in ("cursor-agent", "cursor-cli", "cursor-sub", "anysphere"):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_provider(alias), "cursor")

        # Label override
        self.assertEqual(get_label("cursor"), "Cursor")

    # ---- hermes_cli.auth PROVIDER_REGISTRY ----

    def test_auth_registry_present(self) -> None:
        from hermes_cli.auth import PROVIDER_REGISTRY

        self.assertIn("cursor", PROVIDER_REGISTRY)
        entry = PROVIDER_REGISTRY["cursor"]
        self.assertEqual(entry.auth_type, "external_process")
        self.assertEqual(entry.inference_base_url, "cursor://agent")
        self.assertIn("CURSOR_API_KEY", entry.api_key_env_vars)

    # ---- hermes_cli.models picker + catalog ----

    def test_picker_entry_present(self) -> None:
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        self.assertIn("cursor", slugs)
        entry = next(p for p in CANONICAL_PROVIDERS if p.slug == "cursor")
        self.assertEqual(entry.label, "Cursor")
        # Picker description: short, no em-dash, mentions "100+ models",
        # mirrors OpenRouter's ``OpenRouter (100+ models, pay-per-use)``.
        self.assertEqual(entry.tui_desc, "Cursor (100+ models, subscription)")
        self.assertNotIn("—", entry.tui_desc)

    def test_model_catalog_snapshot(self) -> None:
        from hermes_cli.models import _PROVIDER_MODELS

        self.assertIn("cursor", _PROVIDER_MODELS)
        models = _PROVIDER_MODELS["cursor"]
        # Must include composer-2.5 family and frontier models.
        self.assertIn("auto", models)
        self.assertIn("composer-2.5", models)
        self.assertIn("composer-2.5-fast", models)
        self.assertIn("gpt-5.5-medium", models)
        self.assertIn("claude-opus-4-7-high", models)
        self.assertIn("gemini-3.1-pro", models)
        # No stale junk
        self.assertNotIn("", models)
        self.assertEqual(len(models), len(set(models)))

    def test_model_alias_map(self) -> None:
        # Use private alias map directly — it's the same one the picker uses.
        from hermes_cli.models import _PROVIDER_ALIASES

        for alias in ("cursor-agent", "cursor-cli", "cursor-sub", "anysphere"):
            with self.subTest(alias=alias):
                self.assertEqual(_PROVIDER_ALIASES.get(alias), "cursor")

    def test_provider_model_ids_lists_cursor(self) -> None:
        from hermes_cli.models import provider_model_ids

        # Patch shutil.which to None so the live-fetch path returns nothing
        # and the function falls back to the static snapshot — keeps the test
        # independent of the user's actual cursor-agent install.
        with patch("shutil.which", return_value=None):
            models = provider_model_ids("cursor")
        self.assertIn("composer-2.5", models)
        self.assertIn("auto", models)


class CursorStatusTests(unittest.TestCase):
    """``get_external_process_provider_status`` cursor branch."""

    def setUp(self) -> None:
        keys = [
            "HERMES_CURSOR_COMMAND",
            "CURSOR_AGENT_PATH",
            "CURSOR_API_KEY",
            "HERMES_CURSOR_BASE_URL",
        ]
        self._saved = {k: os.environ.pop(k, None) for k in keys}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_status_reports_configured_when_cli_present(self) -> None:
        from hermes_cli.auth import get_external_process_provider_status

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="✓ Logged in as alice@example.com\n"):
            status = get_external_process_provider_status("cursor")
        self.assertTrue(status["configured"])
        self.assertEqual(status["provider"], "cursor")
        self.assertEqual(status["resolved_command"], "/fake/bin/cursor-agent")
        self.assertEqual(status["base_url"], "cursor://agent")
        self.assertTrue(status["logged_in"])
        self.assertEqual(status["email"], "alice@example.com")

    def test_status_reports_logged_out_when_status_lacks_marker(self) -> None:
        from hermes_cli.auth import get_external_process_provider_status

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="Some other text\n"):
            status = get_external_process_provider_status("cursor")
        self.assertTrue(status["configured"])
        self.assertFalse(status["logged_in"])
        self.assertEqual(status["email"], "")

    def test_status_treats_api_key_env_as_authenticated(self) -> None:
        from hermes_cli.auth import get_external_process_provider_status

        os.environ["CURSOR_API_KEY"] = "crsr_token"
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="not logged in\n"):
            status = get_external_process_provider_status("cursor")
        self.assertTrue(status["logged_in"])

    def test_status_unconfigured_when_cli_missing(self) -> None:
        from hermes_cli.auth import get_external_process_provider_status

        with patch("shutil.which", return_value=None):
            status = get_external_process_provider_status("cursor")
        self.assertFalse(status["configured"])
        self.assertIsNone(status["resolved_command"])

    def test_get_auth_status_dispatches_cursor(self) -> None:
        from hermes_cli.auth import get_auth_status

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="✓ Logged in as bob@example.com\n"):
            status = get_auth_status("cursor")
        self.assertTrue(status["logged_in"])
        self.assertEqual(status["email"], "bob@example.com")


class CursorPickerFlowTests(unittest.TestCase):
    """``_model_flow_cursor`` — end-to-end picker → config persistence."""

    def setUp(self) -> None:
        import tempfile

        keys = [
            "HERMES_CURSOR_COMMAND",
            "CURSOR_AGENT_PATH",
            "CURSOR_API_KEY",
            "HERMES_CURSOR_BASE_URL",
            "HERMES_HOME",
        ]
        self._saved = {k: os.environ.pop(k, None) for k in keys}
        self._tmp_home = tempfile.mkdtemp(prefix="hermes-test-")
        hermes_home = os.path.join(self._tmp_home, ".hermes")
        os.makedirs(hermes_home, exist_ok=True)
        os.environ["HERMES_HOME"] = hermes_home

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp_home, ignore_errors=True)
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_picker_persists_choice_into_config(self) -> None:
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch(
                 "subprocess.check_output",
                 side_effect=[
                     "✓ Logged in as alice@example.com\n",      # status call
                     "auto - Auto (current)\ncomposer-2.5 - Composer 2.5\ncomposer-2.5-fast - Composer 2.5 Fast (default)\n",  # --list-models
                 ],
             ), \
             patch("hermes_cli.auth._prompt_model_selection", return_value="composer-2.5"):
            _model_flow_cursor({}, current_model="")

        cfg = load_config()
        model = cfg.get("model")
        self.assertIsInstance(model, dict)
        self.assertEqual(model["provider"], "cursor")
        self.assertEqual(model["default"], "composer-2.5")
        self.assertEqual(model["base_url"], "cursor://agent")
        self.assertEqual(model["api_mode"], "chat_completions")

    def test_picker_no_op_when_cli_missing_and_user_cancels_install(self) -> None:
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        # No CLI on PATH → install-choice menu pops; user picks Cancel.
        # Nothing should be persisted.
        with patch("shutil.which", return_value=None), \
             patch("hermes_cli.main._prompt_cursor_install_choice",
                   return_value="cancel"):
            _model_flow_cursor({}, current_model="")
        cfg = load_config()
        model = cfg.get("model")
        if isinstance(model, dict):
            self.assertNotEqual(model.get("provider"), "cursor")

    def test_picker_offers_manual_install_when_cli_missing(self) -> None:
        # When the user picks "I'll install it manually" we exit
        # gracefully WITHOUT saving config and WITHOUT trying to run
        # the installer.
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        with patch("shutil.which", return_value=None), \
             patch("hermes_cli.main._prompt_cursor_install_choice",
                   return_value="manual"), \
             patch("hermes_cli.main._run_cursor_agent_installer") as install_fn:
            _model_flow_cursor({}, current_model="")
        # Installer should NOT have been invoked.
        install_fn.assert_not_called()
        cfg = load_config()
        model = cfg.get("model")
        if isinstance(model, dict):
            self.assertNotEqual(model.get("provider"), "cursor")

    def test_picker_runs_installer_then_continues_when_user_picks_install(self) -> None:
        # End-to-end install scenario: CLI missing → user picks
        # "install now" → installer succeeds → status re-checked →
        # auth flow proceeds → model picker saves choice.
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        def _post_install_status(provider_id, *args, **kwargs):
            # After install, status should report configured + logged in.
            return {
                "configured": True,
                "resolved_command": "/fake/bin/cursor-agent",
                "command": "/fake/bin/cursor-agent",
                "base_url": None,
                "email": "alice@example.com",
                "logged_in": True,
            }

        # First status call (BEFORE install) reports not configured;
        # second (AFTER install) reports configured. We mimic that by
        # swapping the underlying mock between the two stages.
        status_call_count = {"n": 0}

        def status_side_effect(provider_id, *args, **kwargs):
            status_call_count["n"] += 1
            if status_call_count["n"] == 1:
                return {
                    "configured": False,
                    "resolved_command": "cursor-agent",
                    "command": "cursor-agent",
                    "base_url": None,
                }
            return _post_install_status(provider_id)

        with patch("hermes_cli.auth.get_external_process_provider_status",
                   side_effect=status_side_effect), \
             patch("hermes_cli.main._prompt_cursor_install_choice",
                   return_value="install"), \
             patch("hermes_cli.main._run_cursor_agent_installer",
                   return_value=True), \
             patch(
                 "subprocess.check_output",
                 side_effect=[
                     "auto - Auto\ncomposer-2.5 - Composer 2.5\n",
                 ],
             ), \
             patch("hermes_cli.auth._prompt_model_selection",
                   return_value="composer-2.5"):
            _model_flow_cursor({}, current_model="")

        cfg = load_config()
        model = cfg.get("model")
        self.assertIsInstance(model, dict)
        self.assertEqual(model["provider"], "cursor")
        self.assertEqual(model["default"], "composer-2.5")

    def test_picker_aborts_when_installer_fails(self) -> None:
        # If the installer returns False (curl failed, user said n, no
        # bash, etc.) we MUST NOT continue to the auth/model phases.
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        with patch("shutil.which", return_value=None), \
             patch("hermes_cli.main._prompt_cursor_install_choice",
                   return_value="install"), \
             patch("hermes_cli.main._run_cursor_agent_installer",
                   return_value=False), \
             patch("hermes_cli.main._prompt_cursor_auth_choice") as auth_fn:
            _model_flow_cursor({}, current_model="")
        # Auth menu should never be reached after install failure.
        auth_fn.assert_not_called()
        cfg = load_config()
        model = cfg.get("model")
        if isinstance(model, dict):
            self.assertNotEqual(model.get("provider"), "cursor")

    def test_picker_no_op_when_user_cancels(self) -> None:
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch(
                 "subprocess.check_output",
                 side_effect=[
                     "✓ Logged in as alice@example.com\n",
                     "auto - Auto\n",
                 ],
             ), \
             patch("hermes_cli.auth._prompt_model_selection", return_value=""):
            _model_flow_cursor({}, current_model="")
        cfg = load_config()
        model = cfg.get("model")
        if isinstance(model, dict):
            self.assertNotEqual(model.get("provider"), "cursor")

    def test_picker_auto_runs_login_when_not_authed_and_user_picks_browser(self) -> None:
        """Picker should offer browser login like Codex/xAI when not authed.

        The auth menu now has three choices — verify [1]/Enter picks browser.
        """
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        login_calls: list[list[str]] = []

        def _fake_subprocess_run(argv, **_kwargs):
            login_calls.append(list(argv))
            return SimpleNamespace(returncode=0)

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch(
                 "subprocess.check_output",
                 side_effect=[
                     # First status probe: not logged in.
                     "not logged in\n",
                     # After login: status reports the email.
                     "✓ Logged in as alice@example.com\n",
                     # --list-models call afterwards
                     "auto - Auto\ncomposer-2.5-fast - Composer 2.5 Fast (default)\n",
                 ],
             ), \
             patch("subprocess.run", side_effect=_fake_subprocess_run), \
             patch("builtins.input", return_value="1"), \
             patch("hermes_cli.auth._prompt_model_selection", return_value="composer-2.5-fast"):
            _model_flow_cursor({}, current_model="")

        # cursor-agent login must have been invoked.
        self.assertEqual(len(login_calls), 1)
        self.assertEqual(login_calls[0], ["/fake/bin/cursor-agent", "login"])

        cfg = load_config()
        self.assertEqual(cfg.get("model", {}).get("provider"), "cursor")
        self.assertEqual(cfg.get("model", {}).get("default"), "composer-2.5-fast")

    def test_picker_default_choice_is_browser(self) -> None:
        """Pressing Enter on the auth menu picks browser (option 1)."""
        from hermes_cli.main import _model_flow_cursor

        login_calls: list[list[str]] = []

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch(
                 "subprocess.check_output",
                 side_effect=[
                     "not logged in\n",
                     "✓ Logged in as alice@example.com\n",
                     "auto - Auto\n",
                 ],
             ), \
             patch(
                 "subprocess.run",
                 side_effect=lambda argv, **_k: (login_calls.append(list(argv)), SimpleNamespace(returncode=0))[1],
             ), \
             patch("builtins.input", return_value=""), \
             patch("hermes_cli.auth._prompt_model_selection", return_value="auto"):
            _model_flow_cursor({}, current_model="")

        self.assertEqual(len(login_calls), 1)
        self.assertIn("login", login_calls[0])

    def test_picker_paste_key_path_saves_to_env(self) -> None:
        """Choosing [2] reads CURSOR_API_KEY interactively and writes to .env."""
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        saved_keys: dict[str, str] = {}

        def _fake_save_env_value(name: str, value: str) -> None:
            saved_keys[name] = value

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch(
                 "subprocess.check_output",
                 side_effect=[
                     "not logged in\n",       # initial status
                     # After saving CURSOR_API_KEY env var the status helper
                     # treats it as authenticated; we still mock the second
                     # status call for symmetry.
                     "not logged in\n",
                     "auto - Auto\ncomposer-2.5 - Composer 2.5\n",
                 ],
             ), \
             patch("builtins.input", return_value="2"), \
             patch("getpass.getpass", return_value="crsr_definitely_long_enough_value_42"), \
             patch("hermes_cli.config.save_env_value", side_effect=_fake_save_env_value), \
             patch("hermes_cli.auth._prompt_model_selection", return_value="composer-2.5"):
            _model_flow_cursor({}, current_model="")

        # Key must have landed in .env via save_env_value.
        self.assertEqual(saved_keys.get("CURSOR_API_KEY"), "crsr_definitely_long_enough_value_42")
        # Env var is populated in-process so the rest of the flow sees it.
        self.assertEqual(os.environ.get("CURSOR_API_KEY"), "crsr_definitely_long_enough_value_42")
        # Config saved.
        cfg = load_config()
        self.assertEqual(cfg.get("model", {}).get("provider"), "cursor")
        self.assertEqual(cfg.get("model", {}).get("default"), "composer-2.5")

    def test_picker_paste_key_rejects_too_short_value(self) -> None:
        """Tiny values fail the length sanity check and abort cleanly."""
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="not logged in\n"), \
             patch("builtins.input", return_value="2"), \
             patch("getpass.getpass", return_value="too_short"):
            _model_flow_cursor({}, current_model="")

        cfg = load_config()
        model = cfg.get("model")
        if isinstance(model, dict):
            self.assertNotEqual(model.get("provider"), "cursor")

    def test_picker_aborts_cleanly_when_user_picks_cancel(self) -> None:
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        login_calls: list[list[str]] = []

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="not logged in\n"), \
             patch("subprocess.run", side_effect=lambda argv, **_k: login_calls.append(list(argv))), \
             patch("builtins.input", return_value="3"):
            _model_flow_cursor({}, current_model="")

        # No login attempt because the user cancelled.
        self.assertEqual(login_calls, [])
        cfg = load_config()
        model = cfg.get("model")
        if isinstance(model, dict):
            self.assertNotEqual(model.get("provider"), "cursor")

    def test_picker_handles_failed_login_gracefully(self) -> None:
        from hermes_cli.config import load_config
        from hermes_cli.main import _model_flow_cursor

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"), \
             patch("subprocess.check_output", return_value="not logged in\n"), \
             patch(
                 "subprocess.run",
                 return_value=SimpleNamespace(returncode=1),
             ), \
             patch("builtins.input", return_value="1"):
            _model_flow_cursor({}, current_model="")

        cfg = load_config()
        model = cfg.get("model")
        if isinstance(model, dict):
            self.assertNotEqual(model.get("provider"), "cursor")


class CursorIdeDetectionTests(unittest.TestCase):
    """``_detect_cursor_ide_install`` cross-platform paths."""

    def test_detects_linux_install(self) -> None:
        from hermes_cli.main import _detect_cursor_ide_install

        with patch("platform.system", return_value="Linux"), \
             patch("os.path.isdir", side_effect=lambda p: p.endswith("/.config/Cursor")), \
             patch("os.path.exists", return_value=False):
            info = _detect_cursor_ide_install()
        self.assertTrue(info["installed"])
        self.assertTrue(info["path"].endswith("/.config/Cursor"))
        self.assertEqual(info["platform"], "linux-config")

    def test_detects_macos_install_via_application_support(self) -> None:
        from hermes_cli.main import _detect_cursor_ide_install

        with patch("platform.system", return_value="Darwin"), \
             patch(
                 "os.path.isdir",
                 side_effect=lambda p: "Application Support/Cursor" in p,
             ), \
             patch("os.path.exists", return_value=False):
            info = _detect_cursor_ide_install()
        self.assertTrue(info["installed"])
        self.assertEqual(info["platform"], "macos-app-support")

    def test_detects_macos_install_via_applications_dir(self) -> None:
        from hermes_cli.main import _detect_cursor_ide_install

        with patch("platform.system", return_value="Darwin"), \
             patch("os.path.isdir", return_value=False), \
             patch("os.path.exists", side_effect=lambda p: p == "/Applications/Cursor.app"):
            info = _detect_cursor_ide_install()
        self.assertTrue(info["installed"])
        self.assertEqual(info["path"], "/Applications/Cursor.app")
        self.assertEqual(info["platform"], "macos-applications")

    def test_detects_windows_install(self) -> None:
        from hermes_cli.main import _detect_cursor_ide_install

        with patch("platform.system", return_value="Windows"), \
             patch.dict(os.environ, {"APPDATA": "C:/Users/test/AppData/Roaming"}, clear=False), \
             patch("os.path.isdir", side_effect=lambda p: "Cursor" in p and "Roaming" in p), \
             patch("os.path.exists", return_value=False):
            info = _detect_cursor_ide_install()
        self.assertTrue(info["installed"])
        self.assertEqual(info["platform"], "windows-roaming")

    def test_reports_not_installed_when_nothing_found(self) -> None:
        from hermes_cli.main import _detect_cursor_ide_install

        with patch("platform.system", return_value="Linux"), \
             patch("os.path.isdir", return_value=False), \
             patch("os.path.exists", return_value=False):
            info = _detect_cursor_ide_install()
        self.assertFalse(info["installed"])
        self.assertEqual(info["path"], "")


class CursorRuntimeProviderTests(unittest.TestCase):
    """``resolve_runtime_provider`` must build a cursor dict, not fall through.

    Without this branch the chat REPL falls through to the openrouter default
    and dies with "Provider resolver returned an empty API key. Set
    OPENROUTER_API_KEY..." — the bug we hit in interactive testing.
    """

    def setUp(self) -> None:
        keys = [
            "HERMES_CURSOR_COMMAND",
            "CURSOR_AGENT_PATH",
            "CURSOR_API_KEY",
            "HERMES_CURSOR_BASE_URL",
        ]
        self._saved = {k: os.environ.pop(k, None) for k in keys}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_runtime_provider_returns_cursor_dict(self) -> None:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            rt = resolve_runtime_provider(requested="cursor")

        self.assertEqual(rt["provider"], "cursor")
        self.assertEqual(rt["api_mode"], "chat_completions")
        self.assertEqual(rt["base_url"], "cursor://agent")
        self.assertEqual(rt["command"], "/fake/bin/cursor-agent")
        # api_key must be either a real key or the sentinel — never empty.
        self.assertTrue(rt["api_key"])
        self.assertEqual(rt["source"], "process")

    def test_runtime_provider_threads_real_api_key(self) -> None:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        os.environ["CURSOR_API_KEY"] = "crsr_real_runtime_test"
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            rt = resolve_runtime_provider(requested="cursor")
        self.assertEqual(rt["api_key"], "crsr_real_runtime_test")


class CursorCredsResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        # Stash relevant env so we don't leak state across tests.
        keys = [
            "HERMES_CURSOR_COMMAND",
            "HERMES_CURSOR_ARGS",
            "CURSOR_AGENT_PATH",
            "CURSOR_API_KEY",
            "HERMES_CURSOR_BASE_URL",
        ]
        self._saved = {k: os.environ.pop(k, None) for k in keys}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_creds_resolver_finds_cli_and_returns_dict(self) -> None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        # Pretend cursor-agent lives at a known fake path.
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            creds = resolve_external_process_provider_credentials("cursor")

        self.assertEqual(creds["provider"], "cursor")
        self.assertEqual(creds["command"], "/fake/bin/cursor-agent")
        self.assertEqual(creds["base_url"], "cursor://agent")
        self.assertEqual(creds["source"], "process")
        # No api_key in env → sentinel value
        self.assertEqual(creds["api_key"], "cursor-agent-login")
        self.assertEqual(creds["args"], [])

    def test_creds_resolver_threads_api_key_when_set(self) -> None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        os.environ["CURSOR_API_KEY"] = "crsr_real_value"
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            creds = resolve_external_process_provider_credentials("cursor")
        self.assertEqual(creds["api_key"], "crsr_real_value")

    def test_creds_resolver_raises_when_cli_missing(self) -> None:
        from hermes_cli.auth import AuthError, resolve_external_process_provider_credentials

        with patch("shutil.which", return_value=None):
            with self.assertRaises(AuthError) as ctx:
                resolve_external_process_provider_credentials("cursor")
        self.assertEqual(ctx.exception.code, "missing_cursor_cli")
        self.assertIn("Cursor CLI", str(ctx.exception))

    def test_creds_resolver_honors_extra_args_env(self) -> None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        os.environ["HERMES_CURSOR_ARGS"] = "--header X-Foo:1 --header X-Bar:2"
        with patch("shutil.which", return_value="/fake/bin/cursor-agent"):
            creds = resolve_external_process_provider_credentials("cursor")
        self.assertEqual(creds["args"], ["--header", "X-Foo:1", "--header", "X-Bar:2"])

    def test_creds_resolver_does_not_disturb_copilot_acp(self) -> None:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        with patch("shutil.which", return_value="/fake/bin/copilot"):
            creds = resolve_external_process_provider_credentials("copilot-acp")
        self.assertEqual(creds["provider"], "copilot-acp")
        self.assertEqual(creds["api_key"], "copilot-acp")
        self.assertEqual(creds["args"], ["--acp", "--stdio"])


class CursorInstallFlowTests(unittest.TestCase):
    """Unit tests for the missing-CLI handling helpers.

    Mirrors the Claude Code "CLI required" pattern: when ``cursor-agent``
    isn't on PATH we offer to install it inline, point to manual install
    docs, or cancel — instead of bouncing the user out of the picker.
    """

    def test_install_choice_default_is_install(self) -> None:
        # Pressing Enter on the missing-CLI menu should pick the
        # recommended "install now" option (matches the recommended
        # default of the auth-choice menu).
        from hermes_cli.main import _prompt_cursor_install_choice
        with patch("builtins.input", return_value=""):
            self.assertEqual(_prompt_cursor_install_choice({}), "install")

    def test_install_choice_accepts_y_or_yes_as_install(self) -> None:
        # Users commonly type ``y``/``yes`` for confirmation prompts.
        # We accept both as "install now".
        from hermes_cli.main import _prompt_cursor_install_choice
        for ans in ("1", "i", "install", "y", "yes"):
            with patch("builtins.input", return_value=ans):
                self.assertEqual(
                    _prompt_cursor_install_choice({}), "install",
                    f"answer {ans!r} should map to install",
                )

    def test_install_choice_manual_path(self) -> None:
        from hermes_cli.main import _prompt_cursor_install_choice
        for ans in ("2", "m", "manual"):
            with patch("builtins.input", return_value=ans):
                self.assertEqual(_prompt_cursor_install_choice({}), "manual")

    def test_install_choice_cancel_on_other(self) -> None:
        from hermes_cli.main import _prompt_cursor_install_choice
        for ans in ("3", "no", "cancel", "q"):
            with patch("builtins.input", return_value=ans):
                self.assertEqual(_prompt_cursor_install_choice({}), "cancel")

    def test_install_choice_cancel_on_ctrl_c(self) -> None:
        from hermes_cli.main import _prompt_cursor_install_choice
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            self.assertEqual(_prompt_cursor_install_choice({}), "cancel")

    def test_installer_aborts_when_user_declines_consent(self) -> None:
        # ``curl | bash`` is a security-sensitive pipeline. The
        # installer must NOT run if the user answers anything but
        # ``y`` / ``yes``. Verified via ``subprocess.run`` not being
        # invoked.
        from hermes_cli.main import _run_cursor_agent_installer
        with patch("builtins.input", return_value="n"), \
             patch("subprocess.run") as run_mock:
            ok = _run_cursor_agent_installer()
        self.assertFalse(ok)
        run_mock.assert_not_called()

    def test_installer_returns_true_when_curl_succeeds_and_cli_on_path(self) -> None:
        from hermes_cli.main import _run_cursor_agent_installer
        fake_result = SimpleNamespace(returncode=0)
        with patch("builtins.input", return_value="y"), \
             patch("subprocess.run", return_value=fake_result), \
             patch("shutil.which", return_value="/usr/local/bin/cursor-agent"):
            self.assertTrue(_run_cursor_agent_installer())

    def test_installer_falls_back_to_local_bin_when_path_not_refreshed(self) -> None:
        # cursor.com installs to ~/.local/bin which may not be on the
        # current shell's PATH yet. We must surface the absolute path
        # AND set HERMES_CURSOR_COMMAND so the SAME picker invocation
        # can continue instead of forcing the user to re-run.
        import tempfile
        from hermes_cli.main import _run_cursor_agent_installer

        prev = os.environ.pop("HERMES_CURSOR_COMMAND", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fake_local_bin = os.path.join(tmp, ".local", "bin")
                os.makedirs(fake_local_bin, exist_ok=True)
                fake_cli = os.path.join(fake_local_bin, "cursor-agent")
                with open(fake_cli, "w") as fh:
                    fh.write("#!/bin/sh\necho fake\n")
                fake_result = SimpleNamespace(returncode=0)
                with patch("builtins.input", return_value="y"), \
                     patch("subprocess.run", return_value=fake_result), \
                     patch("shutil.which", return_value=None), \
                     patch("os.path.expanduser", lambda p: p.replace("~", tmp)):
                    ok = _run_cursor_agent_installer()
            self.assertTrue(ok)
            # Picker can now resolve cursor-agent in this session.
            self.assertEqual(os.environ.get("HERMES_CURSOR_COMMAND"), fake_cli)
        finally:
            if prev is None:
                os.environ.pop("HERMES_CURSOR_COMMAND", None)
            else:
                os.environ["HERMES_CURSOR_COMMAND"] = prev

    def test_windows_skips_curl_bash_and_prints_wsl_instructions(self) -> None:
        # Cursor has no native Windows CLI; Cursor's docs say WSL-only.
        # On Windows we must NOT attempt the curl|bash flow (it would
        # produce a confusing "bash: not found" in PowerShell). We
        # short-circuit to WSL install instructions and exit cleanly.
        import io
        from contextlib import redirect_stdout
        from hermes_cli.main import _model_flow_cursor

        # Mock platform AND ensure no installer/menu helpers are called.
        with patch("shutil.which", return_value=None), \
             patch("platform.system", return_value="Windows"), \
             patch("hermes_cli.main._prompt_cursor_install_choice") as menu, \
             patch("hermes_cli.main._run_cursor_agent_installer") as installer:
            buf = io.StringIO()
            with redirect_stdout(buf):
                _model_flow_cursor({}, current_model="")
            output = buf.getvalue()

        menu.assert_not_called()
        installer.assert_not_called()
        # Must surface the WSL path and the exact install command.
        self.assertIn("WSL", output)
        self.assertIn("wsl --install", output)
        self.assertIn("curl https://cursor.com/install", output)
        # Must explicitly NOT mention PowerShell installers (which
        # phishing sites use as cover).
        self.assertNotIn("iwr ", output)
        self.assertNotIn("iex ", output)

    def test_windows_warning_includes_security_note(self) -> None:
        # Cursor's WSL path is the only safe install vector; we
        # explicitly warn against third-party PowerShell installers
        # because malicious lookalikes have been spreading for similar
        # CLI tools (Gemini, Claude Code) per Jan 2026 reports.
        import io
        from contextlib import redirect_stdout
        from hermes_cli.main import _print_cursor_windows_install_instructions

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_cursor_windows_install_instructions()
        output = buf.getvalue()
        self.assertIn("cursor.com", output)
        self.assertIn("malware", output)
        self.assertIn("third-party", output)

    def test_installer_returns_false_on_nonzero_curl_exit(self) -> None:
        # When curl fails (no internet, mirror down, etc.) the
        # installer must surface failure cleanly so the caller can
        # abort instead of falling through to a half-broken auth flow.
        from hermes_cli.main import _run_cursor_agent_installer
        fake_result = SimpleNamespace(returncode=2)
        with patch("builtins.input", return_value="y"), \
             patch("subprocess.run", return_value=fake_result):
            self.assertFalse(_run_cursor_agent_installer())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
