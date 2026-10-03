"""
Tests: core/context.py
========================
Tests for OS detection, shell detection, and git context extraction.
All tests run without making real subprocess calls (where possible).
"""

import os
import platform
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.core.context import (
    _detect_os,
    _detect_shell,
    _git_branch,
    _git_status,
    _is_git_repo,
    _run_git,
    get_context,
    is_command_destructive,
)
from src.core.schema import ShellContext


# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

class TestDetectOS:
    def test_linux(self):
        with patch("platform.system", return_value="Linux"):
            assert _detect_os() == "linux"

    def test_macos(self):
        with patch("platform.system", return_value="Darwin"):
            assert _detect_os() == "darwin"

    def test_windows(self):
        with patch("platform.system", return_value="Windows"):
            assert _detect_os() == "windows"

    def test_unknown_passthrough(self):
        with patch("platform.system", return_value="FreeBSD"):
            assert _detect_os() == "freebsd"


# ---------------------------------------------------------------------------
# Shell detection
# ---------------------------------------------------------------------------

class TestDetectShell:
    def test_bash_from_env(self):
        with patch.dict(os.environ, {"SHELL": "/bin/bash"}, clear=False):
            assert _detect_shell() == "bash"

    def test_zsh_from_env(self):
        with patch.dict(os.environ, {"SHELL": "/usr/bin/zsh"}, clear=False):
            assert _detect_shell() == "zsh"

    def test_fish_from_env(self):
        with patch.dict(os.environ, {"SHELL": "/usr/local/bin/fish"}, clear=False):
            assert _detect_shell() == "fish"

    def test_powershell_detection(self):
        env = {"PSModulePath": "C:\\Windows\\system32\\WindowsPowerShell"}
        with patch.dict(os.environ, env, clear=True):
            # Remove SHELL to force PSModulePath path
            os.environ.pop("SHELL", None)
            assert _detect_shell() == "powershell"

    def test_unknown_shell_defaults_to_bash(self):
        with patch.dict(os.environ, {"SHELL": "/usr/bin/nushell"}, clear=True):
            # nushell is not in the known set → defaults to bash
            assert _detect_shell() == "bash"


# ---------------------------------------------------------------------------
# Git detection
# ---------------------------------------------------------------------------

class TestGitDetection:
    def test_is_git_repo_true(self):
        with patch("src.core.context._run_git", return_value="true"):
            assert _is_git_repo() is True

    def test_is_git_repo_false(self):
        with patch("src.core.context._run_git", return_value=None):
            assert _is_git_repo() is False

    def test_git_branch_returns_name(self):
        with patch("src.core.context._run_git", return_value="feature/new-cache"):
            assert _git_branch() == "feature/new-cache"

    def test_git_branch_empty_outside_repo(self):
        with patch("src.core.context._run_git", return_value=None):
            assert _git_branch() == ""

    def test_git_status_parses_correctly(self):
        porcelain_output = " M src/cli.py\n?? tests/new_test.py\n M README.md"
        with patch("src.core.context._run_git", return_value=porcelain_output):
            status = _git_status()
        assert "M" in status  # Modified files detected
        assert "??" in status  # Untracked files detected

    def test_git_status_empty_repo(self):
        with patch("src.core.context._run_git", return_value=""):
            assert _git_status() == ""

    def test_run_git_handles_missing_git(self):
        """_run_git should return None gracefully if git is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _run_git("status")
        assert result is None

    def test_run_git_handles_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 3)):
            result = _run_git("status")
        assert result is None


# ---------------------------------------------------------------------------
# Full context
# ---------------------------------------------------------------------------

class TestGetContext:
    def test_returns_shell_context(self):
        ctx = get_context()
        assert isinstance(ctx, ShellContext)
        assert ctx.os_name in ("linux", "darwin", "windows", "freebsd")
        assert ctx.shell in ("bash", "zsh", "fish", "powershell", "cmd", "sh", "ksh", "tcsh")
        assert ctx.cwd  # Must be non-empty

    def test_context_has_cwd(self):
        ctx = get_context()
        from pathlib import Path
        assert Path(ctx.cwd).exists()

    def test_to_prompt_string_format(self, sample_context):
        prompt = sample_context.to_prompt_string()
        assert "OS:" in prompt
        assert "Shell:" in prompt
        assert "CWD:" in prompt
        assert "Git:" in prompt
        assert "main" in prompt  # Branch name


# ---------------------------------------------------------------------------
# Destructive pattern detection
# ---------------------------------------------------------------------------

class TestDestructiveDetection:
    @pytest.mark.parametrize("command,expected", [
        ("rm -rf /tmp/cache",           True),
        ("kill -9 1234",                True),
        ("git reset --hard HEAD~3",     True),
        ("DROP TABLE users;",           True),
        ("ls -la",                      False),
        ("grep -r TODO .",              False),
        ("cat README.md",               False),
        ("find . -name '*.py'",         False),
        ("docker ps",                   False),
        ("git log --oneline -10",       False),
    ])
    def test_destructive_patterns(self, command: str, expected: bool):
        assert is_command_destructive(command) == expected

    def test_case_insensitive(self):
        """DROP TABLE should match regardless of case."""
        assert is_command_destructive("drop table users") is True
        assert is_command_destructive("DROP TABLE USERS") is True
