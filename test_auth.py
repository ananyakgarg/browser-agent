"""Tests for auth.py — storage state discovery, validation, and CLI commands."""

import json
import os
import stat
import time
from pathlib import Path
from unittest import mock

import pytest

from auth import (
    AGENT_STATE_PATH,
    DOWNLOADS_STATE_PATH,
    MAX_STATE_AGE_SEC,
    find_storage_state,
    validate_storage_state,
    print_auth_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_storage_state(cookies=None, origins=None):
    """Build a valid Playwright storage state dict."""
    return {
        "cookies": cookies or [
            {
                "name": "session",
                "value": "abc123",
                "domain": ".github.com",
                "path": "/",
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax",
                "expires": int(time.time()) + 3600,
            },
            {
                "name": "_ga",
                "value": "GA1.2.xxx",
                "domain": ".google.com",
                "path": "/",
                "httpOnly": False,
                "secure": False,
                "sameSite": "None",
                "expires": int(time.time()) + 86400,
            },
        ],
        "origins": origins or [
            {
                "origin": "https://github.com",
                "localStorage": [
                    {"name": "theme", "value": "dark"},
                ],
            },
        ],
    }


@pytest.fixture
def state_dir(tmp_path):
    """Create a temp directory structure mimicking ~/.agent-auth/."""
    agent_dir = tmp_path / ".agent-auth"
    agent_dir.mkdir()
    downloads_dir = tmp_path / "Downloads"
    downloads_dir.mkdir()
    return tmp_path


def _write_state(path, state=None, age_seconds=0):
    """Write a storage state file, optionally back-dating its mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = state or _make_storage_state()
    path.write_text(json.dumps(data, indent=2))
    if age_seconds > 0:
        old_time = time.time() - age_seconds
        os.utime(path, (old_time, old_time))


# ---------------------------------------------------------------------------
# find_storage_state
# ---------------------------------------------------------------------------

class TestFindStorageState:
    def test_returns_none_when_no_files(self, state_dir):
        with mock.patch("auth.AGENT_STATE_PATH", state_dir / ".agent-auth" / "state.json"), \
             mock.patch("auth.DOWNLOADS_STATE_PATH", state_dir / "Downloads" / "agent-auth-state.json"):
            assert find_storage_state() is None

    def test_finds_agent_state(self, state_dir):
        agent_path = state_dir / ".agent-auth" / "state.json"
        _write_state(agent_path)

        with mock.patch("auth.AGENT_STATE_PATH", agent_path), \
             mock.patch("auth.DOWNLOADS_STATE_PATH", state_dir / "Downloads" / "agent-auth-state.json"):
            result = find_storage_state()
            assert result == agent_path

    def test_finds_downloads_state(self, state_dir):
        dl_path = state_dir / "Downloads" / "agent-auth-state.json"
        _write_state(dl_path)

        with mock.patch("auth.AGENT_STATE_PATH", state_dir / ".agent-auth" / "state.json"), \
             mock.patch("auth.DOWNLOADS_STATE_PATH", dl_path):
            result = find_storage_state()
            assert result == dl_path

    def test_prefers_newer_file(self, state_dir):
        agent_path = state_dir / ".agent-auth" / "state.json"
        dl_path = state_dir / "Downloads" / "agent-auth-state.json"

        # Agent state is older
        _write_state(agent_path, age_seconds=600)
        _write_state(dl_path, age_seconds=0)

        with mock.patch("auth.AGENT_STATE_PATH", agent_path), \
             mock.patch("auth.DOWNLOADS_STATE_PATH", dl_path):
            result = find_storage_state()
            assert result == dl_path

    def test_secures_file_permissions(self, state_dir):
        agent_path = state_dir / ".agent-auth" / "state.json"
        _write_state(agent_path)
        # Make world-readable
        os.chmod(agent_path, 0o644)

        with mock.patch("auth.AGENT_STATE_PATH", agent_path), \
             mock.patch("auth.DOWNLOADS_STATE_PATH", state_dir / "Downloads" / "x.json"):
            find_storage_state()
            mode = agent_path.stat().st_mode
            assert not (mode & stat.S_IRGRP), "Group read should be removed"
            assert not (mode & stat.S_IROTH), "Other read should be removed"


# ---------------------------------------------------------------------------
# validate_storage_state
# ---------------------------------------------------------------------------

class TestValidateStorageState:
    def test_valid_fresh_state(self, tmp_path):
        path = tmp_path / "state.json"
        _write_state(path)

        info = validate_storage_state(path)
        assert info["valid"] is True
        assert info["cookie_count"] == 2
        assert info["origin_count"] == 1
        assert "github.com" in info["domains"]
        assert "google.com" in info["domains"]
        assert info["warnings"] == []

    def test_file_not_found(self, tmp_path):
        info = validate_storage_state(tmp_path / "nope.json")
        assert info["valid"] is False
        assert "File not found" in info["warnings"][0]

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json{{{")

        info = validate_storage_state(path)
        assert info["valid"] is False
        assert "Invalid JSON" in info["warnings"][0]

    def test_warns_on_old_state(self, tmp_path):
        path = tmp_path / "state.json"
        _write_state(path, age_seconds=MAX_STATE_AGE_SEC + 3600)

        info = validate_storage_state(path)
        assert info["valid"] is True
        assert any("hours old" in w for w in info["warnings"])

    def test_warns_on_expired_cookies(self, tmp_path):
        path = tmp_path / "state.json"
        state = _make_storage_state(cookies=[
            {
                "name": "expired_session",
                "value": "old",
                "domain": ".example.com",
                "path": "/",
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax",
                "expires": int(time.time()) - 3600,  # expired 1h ago
            },
            {
                "name": "valid_session",
                "value": "new",
                "domain": ".example.com",
                "path": "/",
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax",
                "expires": int(time.time()) + 3600,
            },
        ])
        _write_state(path, state=state)

        info = validate_storage_state(path)
        assert info["valid"] is True
        assert any("expired" in w for w in info["warnings"])

    def test_empty_state_is_valid(self, tmp_path):
        path = tmp_path / "state.json"
        _write_state(path, state={"cookies": [], "origins": []})

        info = validate_storage_state(path)
        assert info["valid"] is True
        assert info["cookie_count"] == 0
        assert info["origin_count"] == 0

    def test_session_cookies_no_expiry(self, tmp_path):
        """Cookies with expires=-1 (session cookies) should not be flagged as expired."""
        path = tmp_path / "state.json"
        state = _make_storage_state(cookies=[
            {
                "name": "sid",
                "value": "abc",
                "domain": ".example.com",
                "path": "/",
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax",
                "expires": -1,
            },
        ])
        _write_state(path, state=state)

        info = validate_storage_state(path)
        assert not any("expired" in w for w in info["warnings"])


# ---------------------------------------------------------------------------
# print_auth_status
# ---------------------------------------------------------------------------

class TestPrintAuthStatus:
    def test_no_state_found(self, state_dir, capsys):
        with mock.patch("auth.AGENT_STATE_PATH", state_dir / ".agent-auth" / "state.json"), \
             mock.patch("auth.DOWNLOADS_STATE_PATH", state_dir / "Downloads" / "x.json"):
            result = print_auth_status()
            assert result is False
            out = capsys.readouterr().out
            assert "No storage state found" in out

    def test_valid_state(self, state_dir, capsys):
        agent_path = state_dir / ".agent-auth" / "state.json"
        _write_state(agent_path)

        with mock.patch("auth.AGENT_STATE_PATH", agent_path), \
             mock.patch("auth.DOWNLOADS_STATE_PATH", state_dir / "Downloads" / "x.json"):
            result = print_auth_status()
            assert result is True
            out = capsys.readouterr().out
            assert "Cookies: 2" in out
            assert "github.com" in out


# ---------------------------------------------------------------------------
# CLI integration (main.py)
# ---------------------------------------------------------------------------

class TestCLIAuthCommands:
    def test_auth_status_exit_code(self, state_dir):
        """--auth-status exits 1 when no state found."""
        import subprocess
        result = subprocess.run(
            ["python3", "main.py", "--auth-status"],
            capture_output=True, text=True,
            env={**os.environ,
                 "HOME": str(state_dir)},  # redirect HOME so paths don't exist
        )
        assert result.returncode == 1
        assert "No storage state found" in result.stdout

    def test_auth_status_exit_0_with_state(self, state_dir):
        """--auth-status exits 0 when valid state exists."""
        agent_dir = state_dir / ".agent-auth"
        agent_dir.mkdir(exist_ok=True)
        _write_state(agent_dir / "state.json")

        import subprocess
        result = subprocess.run(
            ["python3", "main.py", "--auth-status"],
            capture_output=True, text=True,
            env={**os.environ, "HOME": str(state_dir)},
        )
        assert result.returncode == 0
        assert "Cookies: 2" in result.stdout

    def test_auth_check_no_urls(self, state_dir):
        """--auth-check with no URLs should error."""
        agent_dir = state_dir / ".agent-auth"
        agent_dir.mkdir(exist_ok=True)
        _write_state(agent_dir / "state.json")

        import subprocess
        result = subprocess.run(
            ["python3", "main.py", "--auth-check"],
            capture_output=True, text=True,
            env={**os.environ, "HOME": str(state_dir)},
        )
        assert result.returncode == 1

    def test_requires_instruction_and_csv(self):
        """Without --auth-status, --instruction and --csv are required."""
        import subprocess
        result = subprocess.run(
            ["python3", "main.py"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
