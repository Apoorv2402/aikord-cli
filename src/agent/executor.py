"""
Subprocess Executor
=====================
Runs shell commands as subprocesses and captures stdout, stderr, and exit code.

This module is the boundary between aikord-cli's safe AI world and the
real operating system. Every execution decision made here has real consequences.

Design Principles:
  1. NEVER shell=True — always pass command as a list to prevent injection.
     shell=True would let a malicious command like "ls; rm -rf /" execute both.
  2. Always capture stderr — it's the primary source of error information.
  3. Always set a timeout — hanging commands would freeze the CLI.
  4. Return structured results — callers should never parse raw strings.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import time
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """
    The result of a subprocess execution.

    Attributes:
        command:     The command that was run.
        exit_code:   0 = success, anything else = failure.
        stdout:      Standard output (text).
        stderr:      Standard error (text). Check this on failure.
        duration_ms: Wall-clock execution time in milliseconds.
        timed_out:   True if the command exceeded the timeout.
    """

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """True when exit_code is 0 and command didn't time out."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def error_summary(self) -> str:
        """
        Best-effort error description for the self-healing LLM prompt.
        Combines stderr and exit code into one human-readable string.
        """
        parts = [f"Exit code: {self.exit_code}"]
        if self.timed_out:
            parts.append("Command timed out")
        if self.stderr.strip():
            parts.append(f"Stderr:\n{self.stderr.strip()[:1500]}")
        elif self.stdout.strip():
            # Some tools write errors to stdout (e.g. git)
            parts.append(f"Stdout (no stderr):\n{self.stdout.strip()[:1500]}")
        return "\n".join(parts)


class Executor:
    """
    Runs shell commands safely with timeout, output capture, and OS-aware splitting.

    Usage:
        executor = Executor(timeout_seconds=30)
        result = await executor.run("ls -la /tmp")
        if result.success:
            print(result.stdout)
        else:
            print(result.error_summary)
    """

    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout = timeout_seconds

    async def run(self, command: str) -> ExecutionResult:
        """
        Execute a shell command asynchronously and return the result.

        Args:
            command: The shell command string to execute.

        Returns:
            ExecutionResult: Structured execution result (never raises on failure).

        Security Notes:
          - We parse the command using shlex.split() which handles quoting correctly.
          - We never use shell=True (prevents shell injection attacks).
          - Output is captured (not streamed to terminal) for programmatic use.
        """
        start = time.perf_counter()

        # Parse command string into argument list
        # shlex.split handles: 'ls -la "my dir"' → ['ls', '-la', 'my dir']
        try:
            args = shlex.split(command)
        except ValueError as e:
            # Malformed command (unmatched quotes, etc.)
            return ExecutionResult(
                command=command,
                exit_code=1,
                stdout="",
                stderr=f"Command parse error: {e}",
                duration_ms=0.0,
            )

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
                duration_ms = (time.perf_counter() - start) * 1000

                return ExecutionResult(
                    command=command,
                    exit_code=process.returncode or 0,
                    stdout=stdout_bytes.decode("utf-8", errors="replace"),
                    stderr=stderr_bytes.decode("utf-8", errors="replace"),
                    duration_ms=duration_ms,
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()  # Drain pipes
                duration_ms = (time.perf_counter() - start) * 1000
                return ExecutionResult(
                    command=command,
                    exit_code=124,  # Standard timeout exit code (same as 'timeout' command)
                    stdout="",
                    stderr=f"Command timed out after {self.timeout}s",
                    duration_ms=duration_ms,
                    timed_out=True,
                )

        except FileNotFoundError:
            # The command executable doesn't exist
            cmd_name = args[0] if args else command
            return ExecutionResult(
                command=command,
                exit_code=127,  # Standard "command not found" exit code
                stdout="",
                stderr=f"Command not found: '{cmd_name}'. Is it installed and on PATH?",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except PermissionError:
            return ExecutionResult(
                command=command,
                exit_code=126,  # Standard "permission denied" exit code
                stdout="",
                stderr=f"Permission denied: '{command}'",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def run_sync(self, command: str) -> ExecutionResult:
        """
        Synchronous wrapper around run() for non-async contexts.
        Uses subprocess directly (no asyncio overhead).
        """
        start = time.perf_counter()
        try:
            args = shlex.split(command)
        except ValueError as e:
            return ExecutionResult(
                command=command, exit_code=1, stdout="",
                stderr=str(e), duration_ms=0.0
            )

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return ExecutionResult(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                command=command, exit_code=124, stdout="",
                stderr=f"Timed out after {self.timeout}s",
                duration_ms=(time.perf_counter() - start) * 1000,
                timed_out=True,
            )
        except FileNotFoundError:
            return ExecutionResult(
                command=command, exit_code=127, stdout="",
                stderr=f"Command not found: {command.split()[0]}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
