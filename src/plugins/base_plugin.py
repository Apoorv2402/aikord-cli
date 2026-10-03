"""
Plugin Base Class — Hook System
=================================
Defines the PluginBase ABC that third-party plugins must implement.
Plugins can hook into the aikord-cli pipeline at defined points:

  pre_suggest   → run before the LLM is called (modify query or context)
  post_suggest  → run after LLM responds (modify or augment the action)
  pre_execute   → run before the command is executed (can block execution)
  post_execute  → run after execution (log, notify, trigger side effects)

Commercialization Note:
========================
This plugin system is the foundation for the aikord-cli Pro plugin store.
In v2, users will be able to:
  - Install official plugins: aikord plugin install aikord-gh-copilot
  - Build and publish their own: aikord plugin publish my-plugin
  - Access premium plugins with a Pro subscription

The hook-based design means plugin authors never need to fork the core tool.
They just implement PluginBase and drop their .py file in the plugin directory.

AI Engineering Note:
=====================
This is the same "hooks" pattern used by:
  - LangChain's Callbacks (on_llm_start, on_llm_end, on_tool_start, ...)
  - Hugging Face Trainer's TrainerCallback
  - FastAPI's middleware

Learning this pattern now means you'll immediately understand LangChain's
internals when you encounter them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.core.schema import AppConfig, ShellContext, TerminalAction
from src.agent.executor import ExecutionResult


@dataclass
class PluginManifest:
    """
    Metadata about a plugin. Read from the plugin's __manifest__ attribute.

    Attributes:
        name:        Human-readable plugin name.
        version:     Semver string (e.g., '1.0.0').
        description: One-line description.
        author:      Plugin author or organisation.
        hooks:       List of hook names this plugin implements.
        requires:    Minimum aikord-cli version (semver).
    """

    name: str
    version: str
    description: str
    author: str
    hooks: list[str]
    requires: str = ">=1.0.0"


class PluginBase(ABC):
    """
    Abstract base class for all aikord-cli plugins.

    To create a plugin:
      1. Subclass PluginBase
      2. Set __manifest__ = PluginManifest(...)
      3. Override the hooks you need (all have no-op defaults)
      4. Drop the .py file in ~/.config/aikord-cli/plugins/

    Example plugin (aikord-notify):
      class NotifyPlugin(PluginBase):
          __manifest__ = PluginManifest(
              name="aikord-notify",
              version="1.0.0",
              description="Send desktop notification on command completion",
              author="aikord",
              hooks=["post_execute"],
          )

          async def post_execute(self, result: ExecutionResult, ...) -> None:
              if result.success:
                  notify("aikord-cli", f"✅ '{result.command}' completed")
              else:
                  notify("aikord-cli", f"❌ '{result.command}' failed")
    """

    __manifest__: PluginManifest  # Must be set by subclass

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.enabled = True

    # ------------------------------------------------------------------
    # Hooks (all optional — default implementations are no-ops)
    # ------------------------------------------------------------------

    async def pre_suggest(
        self,
        query: str,
        context: ShellContext,
    ) -> tuple[str, ShellContext]:
        """
        Hook: before the LLM is called.

        Use cases:
          - Expand abbreviations ("dps" → "docker ps")
          - Inject organisation-specific context
          - Block certain query patterns (compliance)

        Returns:
            Modified (query, context) tuple.
        """
        return query, context

    async def post_suggest(
        self,
        query: str,
        action: TerminalAction,
        context: ShellContext,
    ) -> TerminalAction:
        """
        Hook: after the LLM responds (or cache hit).

        Use cases:
          - Add organisation-specific flags to commands
          - Override risk_level based on internal policy
          - Log suggestions to a team audit trail

        Returns:
            Optionally modified TerminalAction.
        """
        return action

    async def pre_execute(
        self,
        action: TerminalAction,
        context: ShellContext,
    ) -> bool:
        """
        Hook: before the command is executed.

        Returns:
            True  → allow execution to proceed.
            False → block execution (plugin vetoed the command).

        Use cases:
          - Policy enforcement (block 'sudo' commands in CI)
          - Audit logging before execution
          - Rate limiting
        """
        return True  # Allow by default

    async def post_execute(
        self,
        action: TerminalAction,
        result: ExecutionResult,
        context: ShellContext,
    ) -> None:
        """
        Hook: after the command has executed.

        Use cases:
          - Desktop notifications on completion
          - Slack/Teams alerts on failure
          - Telemetry enrichment
          - Auto-commit successful git operations
        """
        pass  # No-op by default
