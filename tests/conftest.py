"""
pytest configuration and shared fixtures.

conftest.py is auto-loaded by pytest before any test.
Fixtures defined here are available to ALL test files without imports.
"""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on the Python path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_terminal_action():
    """A valid, non-destructive TerminalAction for testing."""
    from src.core.schema import TerminalAction
    return TerminalAction(
        command="ls -la",
        explanation="ls: list directory contents. -l: long format. -a: show hidden files.",
        is_destructive=False,
        risk_level="low",
        alternatives=["ls -lah", "find . -maxdepth 1"],
    )


@pytest.fixture
def sample_destructive_action():
    """A destructive TerminalAction for safety testing."""
    from src.core.schema import TerminalAction
    return TerminalAction(
        command="rm -rf /tmp/build",
        explanation="rm: remove files. -r: recursive. -f: force without prompt.",
        is_destructive=True,
        risk_level="high",
        alternatives=["trash /tmp/build"],
    )


@pytest.fixture
def sample_context():
    """A mock ShellContext for testing without real system calls."""
    from src.core.schema import ShellContext
    return ShellContext(
        os_name="linux",
        os_version="Ubuntu 22.04",
        shell="bash",
        cwd="/home/user/project",
        is_git_repo=True,
        git_branch="main",
        git_status="M 2",
        home_dir="/home/user",
        username="user",
    )


@pytest.fixture
def default_config():
    """Default AppConfig for testing (no real API keys)."""
    from src.core.schema import AppConfig
    return AppConfig(
        provider="ollama",
        model="qwen2.5-coder:7b",
        endpoint="http://localhost:11434/v1",
        api_key="ollama",
        cloud_api_key="",   # No real key in tests
        auto_fix=False,
        max_retries=3,
    )
