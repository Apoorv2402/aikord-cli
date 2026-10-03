"""
Shell & System Context Extractor
==================================
Detects the user's OS, active shell dialect, current working directory,
and Git repository state.

AI Concept: Context Injection
===============================
This module implements the "retrieval" step of a simple RAG pipeline.
Before calling the LLM, we scrape facts from the environment and inject
them into the system prompt. The LLM then uses these facts to produce
shell-specific, directory-aware command suggestions.

Without context injection (naive approach):
  User: "list running processes"
  LLM might suggest: ps aux   ← correct for Linux, wrong for PowerShell

With context injection (what we do):
  LLM sees: "Shell: powershell | OS: windows"
  LLM suggests: Get-Process   ← correct for the actual environment

This is the same principle behind ChatGPT's "memory" feature and GitHub
Copilot's workspace context — more context = better suggestions.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from src.core.schema import ShellContext


# ---------------------------------------------------------------------------
# Known destructive command patterns (used by multiple modules)
# ---------------------------------------------------------------------------

DESTRUCTIVE_PATTERNS: frozenset[str] = frozenset({
    "rm", "rmdir", "del", "rd",                       # File deletion
    "drop", "truncate", "delete from",                 # SQL destructive
    "kill", "pkill", "killall", "taskkill",            # Process termination
    "format", "mkfs", "dd",                            # Disk operations
    "git reset --hard", "git clean -f", "git push --force",  # Git destructive
    "chmod 777", "chown", "sudo rm",                   # Permission/system
    "shred", "wipe", "overwrite",                      # Secure deletion
    ":(){:|:&};:",                                     # Fork bomb
    "mv /", "cp /dev/null",                            # Root-level moves
})


def get_context() -> ShellContext:
    """
    Build a ShellContext snapshot of the current terminal environment.

    Returns:
        ShellContext: Fully populated context ready for prompt injection.

    Example:
        >>> ctx = get_context()
        >>> print(ctx.to_prompt_string())
        OS: linux (Ubuntu 22.04)
        Shell: bash
        CWD: /home/user/project
        Git: yes | branch: main | status: M 2
    """
    return ShellContext(
        os_name=_detect_os(),
        os_version=_detect_os_version(),
        shell=_detect_shell(),
        cwd=str(Path.cwd()),
        is_git_repo=_is_git_repo(),
        git_branch=_git_branch(),
        git_status=_git_status(),
        home_dir=str(Path.home()),
        username=os.getenv("USER") or os.getenv("USERNAME") or "unknown",
    )


# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

def _detect_os() -> str:
    """
    Return a normalised OS name: 'linux', 'darwin', or 'windows'.

    Why normalise? LLMs are trained on docs using these canonical names.
    'darwin' (Python's name for macOS) would confuse some models → 'darwin'.
    We keep 'darwin' because macOS users know it from 'uname -s'.
    """
    system = platform.system().lower()
    mapping = {"linux": "linux", "darwin": "darwin", "windows": "windows"}
    return mapping.get(system, system)


def _detect_os_version() -> str:
    """Return a human-readable OS version string."""
    try:
        if platform.system() == "Linux":
            # Try to read /etc/os-release for distro name
            os_release = Path("/etc/os-release")
            if os_release.exists():
                data = dict(
                    line.split("=", 1)
                    for line in os_release.read_text().splitlines()
                    if "=" in line
                )
                pretty = data.get("PRETTY_NAME", "").strip('"')
                if pretty:
                    return pretty
        return platform.version()[:60]  # Cap length
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Shell detection
# ---------------------------------------------------------------------------

def _detect_shell() -> str:
    """
    Detect the active shell dialect.

    Detection chain:
      1. SHELL env var (Unix)          → /bin/bash → 'bash'
      2. COMSPEC env var (Windows cmd) → 'cmd'
      3. PSModulePath env var          → 'powershell' (set by PS automatically)
      4. sys.executable fallback       → 'unknown'

    Why does this matter for the LLM?
      bash:       ls -la, grep, awk, |, &&
      zsh:        same as bash + extended globs
      fish:       set VAR value  (not export VAR=value)
      powershell: Get-ChildItem, Where-Object, |, -Filter
      cmd:        dir, findstr, & (no pipes like bash)
    """
    # Unix shells
    shell_path = os.getenv("SHELL", "")
    if shell_path:
        shell_name = Path(shell_path).name.lower()
        known = {"bash", "zsh", "fish", "ksh", "tcsh", "sh"}
        if shell_name in known:
            return shell_name

    # Windows PowerShell detection
    if os.getenv("PSModulePath"):
        return "powershell"

    # Windows CMD
    if os.getenv("COMSPEC") and platform.system() == "Windows":
        return "cmd"

    # Windows fallback
    if platform.system() == "Windows":
        return "powershell"

    return "bash"  # Safe default for Unix


# ---------------------------------------------------------------------------
# Git detection
# ---------------------------------------------------------------------------

def _run_git(*args: str, timeout: int = 3) -> str | None:
    """
    Run a git command and return stdout, or None on failure.

    We set a short timeout (3s) so a slow git (large monorepos) doesn't
    block the CLI. We also suppress all stderr to keep output clean.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _is_git_repo() -> bool:
    """Check whether the CWD is inside a git repository."""
    return _run_git("rev-parse", "--is-inside-work-tree") == "true"


def _git_branch() -> str:
    """Return the current git branch name, or '' if not in a repo."""
    return _run_git("rev-parse", "--abbrev-ref", "HEAD") or ""


def _git_status() -> str:
    """
    Return a terse git status summary.

    Example: "M 3, A 1, ?? 2" (3 modified, 1 added, 2 untracked)
    We avoid running 'git status' (slow on big repos) and use
    'git status --porcelain' (machine-readable, same speed).
    """
    raw = _run_git("status", "--porcelain")
    if not raw:
        return ""

    counts: dict[str, int] = {}
    for line in raw.splitlines():
        if len(line) >= 2:
            code = line[:2].strip() or "?"
            counts[code] = counts.get(code, 0) + 1

    return ", ".join(f"{code} {n}" for code, n in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Utility: destructive keyword check
# ---------------------------------------------------------------------------

def is_command_destructive(command: str) -> bool:
    """
    Secondary safety check: scan a command string for destructive patterns.

    This is the third layer of destructive detection (after Pydantic's
    field_validator and the LLM's own is_destructive flag). Defence-in-depth.

    Args:
        command: The raw shell command string.

    Returns:
        True if the command matches a known destructive pattern.
    """
    cmd_lower = command.lower()
    return any(pattern in cmd_lower for pattern in DESTRUCTIVE_PATTERNS)
