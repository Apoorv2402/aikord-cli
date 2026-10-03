"""
ReAct Self-Healing Loop
=========================
When a command fails (non-zero exit code), this module kicks in:
  1. Captures the failure (stderr + exit code)
  2. Submits it back to the LLM with a structured remediation prompt
  3. Proposes a corrected command (with safety check)
  4. Loops up to max_retries times
  5. Detects cycles to prevent infinite loops

AI Concept: The ReAct Pattern (Reason + Act)
=============================================
ReAct is one of the most influential AI agent papers (Yao et al., 2022).
The key insight: interleave REASONING and ACTING in a loop.

Traditional pipeline:
  User → LLM → Command → Done (fails silently or errors out)

ReAct agent loop:
  ┌─────────────────────────────────────────────────────────────┐
  │  THOUGHT: "The user wants X. I'll try command Y."           │
  │  ACTION:  Run Y                                             │
  │  OBSERVE: "Y failed with error: permission denied"          │
  │  THOUGHT: "The error is a permission issue. Try sudo Y."    │
  │  ACTION:  Run sudo Y                                        │
  │  OBSERVE: "Success! Exit code 0."                           │
  └─────────────────────────────────────────────────────────────┘

This is the SAME loop used in:
  - AutoGPT (GPT-4 + tool use)
  - SWE-agent (LLM fixes GitHub issues autonomously)
  - Claude Computer Use (Claude clicks/types in a browser)
  - Devin (the "AI software engineer")

The key constraint: we cap retries at 3 to prevent runaway execution.
Real agents (like Devin) can run for hours — for a terminal copilot,
3 retries is enough. If it can't fix in 3 tries, it's likely a domain
problem the LLM can't solve without human insight.

Cycle Detection:
  We hash (command, error_fingerprint) pairs. If the same pair repeats,
  the LLM is going in circles (proposing the same broken fix). Abort.
"""

from __future__ import annotations

import hashlib
import logging

from src.agent.executor import ExecutionResult, Executor
from src.core.context import is_command_destructive
from src.core.schema import AppConfig, ErrorRemediation, ShellContext, TerminalAction
from src.engine.base import LLMProvider
from src.engine.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


class SelfHealingAgent:
    """
    Autonomous ReAct agent that retries failed commands with LLM-proposed fixes.

    Usage:
        agent = SelfHealingAgent(config, provider)

        # When a command fails:
        result = await agent.heal(
            original_action=action,
            failure=execution_result,
            context=shell_context,
        )
        # result is either a successful ExecutionResult or the last failure
    """

    def __init__(
        self,
        config: AppConfig,
        provider: LLMProvider,
        executor: Executor | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.executor = executor or Executor(timeout_seconds=config.request_timeout)
        self.max_retries = config.max_retries

    async def heal(
        self,
        original_action: TerminalAction,
        failure: ExecutionResult,
        context: ShellContext,
        confirm_callback=None,  # Callable[[str, bool], bool] — for TUI confirmation
    ) -> tuple[ExecutionResult, int]:
        """
        Run the ReAct self-correction loop.

        Args:
            original_action:   The TerminalAction that produced the failure.
            failure:           The ExecutionResult from the failed run.
            context:           Current shell context.
            confirm_callback:  Optional async callable for TUI confirmation on
                               destructive corrected commands. Signature:
                               (command: str, is_destructive: bool) -> bool

        Returns:
            Tuple of (final ExecutionResult, number of retries used).
            Check result.success to determine if healing succeeded.

        Loop invariant:
            Each iteration generates a NEW candidate command.
            We track (command, error_fingerprint) pairs for cycle detection.
        """
        current_command = original_action.command
        current_failure = failure
        retry_count = 0

        # Cycle detection: store seen (command_hash, error_hash) pairs
        seen_cycles: set[str] = set()

        logger.info(
            f"[Healer] Starting ReAct loop for failed command: '{current_command}'"
        )

        while retry_count < self.max_retries:
            retry_count += 1
            logger.info(f"[Healer] Retry {retry_count}/{self.max_retries}")

            # Step 1: REASON — ask LLM to diagnose and fix
            try:
                remediation = await self._reason(
                    current_command, current_failure, context
                )
            except Exception as e:
                logger.error(f"[Healer] LLM remediation call failed: {e}")
                break

            corrected = remediation.corrected_command
            logger.info(f"[Healer] LLM proposes fix: '{corrected}'")

            # Step 2: Cycle detection — abort if same fix + same error
            cycle_key = self._make_cycle_key(corrected, current_failure.stderr)
            if cycle_key in seen_cycles:
                logger.warning(
                    f"[Healer] Cycle detected! LLM proposes same fix again. Aborting."
                )
                break
            seen_cycles.add(cycle_key)

            # Step 3: Safety check — re-verify destructive status
            # The LLM's is_destructive may differ from our own keyword check.
            # Use OR (either trigger = treat as destructive). Defence-in-depth.
            final_destructive = (
                remediation.is_destructive
                or is_command_destructive(corrected)
            )

            # Step 4: TUI confirmation for destructive fixes (if callback provided)
            if confirm_callback and final_destructive:
                approved = confirm_callback(corrected, final_destructive)
                if not approved:
                    logger.info("[Healer] User declined destructive fix. Aborting.")
                    return current_failure, retry_count

            # Step 5: ACT — run the corrected command
            logger.info(f"[Healer] Running corrected command: '{corrected}'")
            result = await self.executor.run(corrected)

            # Step 6: OBSERVE — did it work?
            if result.success:
                logger.info(
                    f"[Healer] Self-healing SUCCEEDED on retry {retry_count}. "
                    f"Exit code: {result.exit_code}"
                )
                return result, retry_count

            # Command still failed — feed this failure into the next iteration
            logger.info(
                f"[Healer] Fix attempt {retry_count} FAILED. "
                f"Exit code: {result.exit_code}. Continuing..."
            )
            current_command = corrected
            current_failure = result

        # All retries exhausted
        logger.warning(
            f"[Healer] Self-healing FAILED after {retry_count} retries. "
            f"Last command: '{current_command}'"
        )
        return current_failure, retry_count

    async def _reason(
        self,
        command: str,
        failure: ExecutionResult,
        context: ShellContext,
    ) -> ErrorRemediation:
        """
        The REASON step: submit the failure to the LLM and get a corrected command.

        We use complete_remediation() which has its own specialised prompt
        for error diagnosis (defined in openai_provider.py).
        """
        if isinstance(self.provider, OpenAICompatibleProvider):
            return await self.provider.complete_remediation(
                original_command=command,
                error_output=failure.error_summary,
                context=context,
            )
        else:
            # Generic fallback for other provider types
            raise NotImplementedError(
                "complete_remediation is only implemented for OpenAICompatibleProvider"
            )

    @staticmethod
    def _make_cycle_key(command: str, stderr: str) -> str:
        """
        Create a fingerprint for cycle detection.

        We hash the (command, first 100 chars of stderr) pair.
        Using only the first 100 chars of stderr avoids false negatives
        from varying line numbers or timestamps in error messages.
        """
        fingerprint = f"{command.strip()}|{stderr.strip()[:100]}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
