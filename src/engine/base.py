"""
LLMProvider — Abstract Base Class
====================================

AI Concept: Provider Abstraction Pattern
==========================================
This is the most important design pattern in production AI engineering.

The problem: you want to swap between Ollama (local), DeepSeek (cloud),
Groq (fast), and OpenAI (reliable) without changing a single line of
business logic. Each provider has different auth, URLs, and quirks.

The solution: define a shared contract (this ABC) that every provider
must implement. Callers (router.py, healer.py) only see the ABC — they
never know which concrete provider they're talking to.

This pattern is used everywhere in AI engineering:
  - LangChain's BaseChatModel
  - LlamaIndex's LLM base class
  - OpenAI's own SDK wraps multiple backends behind one interface

It also makes testing trivial: swap the real provider with MockProvider
in tests and never make a real HTTP call.

    ┌──────────────────────────────────────────┐
    │           LLMProvider (ABC)               │  ← callers use this
    │   async def complete(messages) → str      │
    │   async def stream(messages) → AsyncIter  │
    └──────────────────┬───────────────────────┘
                       │ implements
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    OllamaProvider  DeepSeekProvider  MockProvider
    (openai_provider.py, different config)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from src.core.schema import AppConfig, ShellContext, TerminalAction


class LLMProvider(ABC):
    """
    Abstract base class that every LLM backend must implement.

    All providers share the same interface regardless of whether they are:
      - Local:  Ollama running at localhost:11434
      - Cloud:  DeepSeek, Groq, OpenAI running at their respective APIs

    This is possible because all of them implement the OpenAI
    /v1/chat/completions wire format (same JSON request/response structure).
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Core interface — every subclass MUST implement these two methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(
        self,
        query: str,
        context: ShellContext,
        system_prompt: str | None = None,
    ) -> TerminalAction:
        """
        Send a query to the LLM and return a validated TerminalAction.

        Args:
            query:         The user's natural language request.
            context:       Shell/OS/Git context snapshot (from context.py).
            system_prompt: Optional override of the default system prompt.

        Returns:
            TerminalAction: Validated, structured LLM response.

        Raises:
            ValueError:    If the LLM returns malformed JSON.
            httpx.Error:   On network failures.
        """
        ...

    @abstractmethod
    async def stream_complete(
        self,
        query: str,
        context: ShellContext,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the LLM as they are generated.

        AI Concept: Streaming & TTFT
        ==============================
        Streaming is critical for UX. Without it, the user stares at a blank
        screen until the full response is ready (may be 2–5 seconds).

        With streaming, the user sees the first token in ~200–500ms (TTFT).
        Even if total time is the same, perceived performance is much better.

        This is why ChatGPT, Claude, and Gemini all stream by default.

        Yields:
            str: Individual text chunks as they arrive from the LLM.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers — available to all subclasses
    # ------------------------------------------------------------------

    def _build_system_prompt(self, context: ShellContext) -> str:
        """
        Build the system prompt injected before every user query.

        AI Concept: Prompt Engineering
        ================================
        The system prompt is the most powerful lever in structured output.
        A well-crafted system prompt:
          1. Tells the LLM its role (terminal copilot, not a chatbot)
          2. Gives it the environment context (shell, OS, git)
          3. Provides the exact JSON schema it must output
          4. Lists critical rules (destructive detection, no markdown)

        Tips you'll learn from reading this:
          - Be explicit about what NOT to do (no markdown fences)
          - Always include the schema, not just a description
          - Give examples for tricky fields (is_destructive=true cases)
        """
        schema = TerminalAction.model_json_schema()

        return f"""You are aikord-cli, a production-grade AI terminal copilot.
Your job: convert the user's natural language request into the BEST shell command for their environment.

USER ENVIRONMENT:
{context.to_prompt_string()}

OUTPUT FORMAT — you MUST respond with ONLY valid JSON matching this schema:
{schema}

CRITICAL RULES:
1. Output ONLY raw JSON — no markdown, no code fences, no prose before or after.
2. Set is_destructive=true for ANY command that can: delete files/data, kill processes,
   reset git history, drop databases, format disks, or run with elevated privileges.
   When in doubt, set is_destructive=true. FALSE NEGATIVES ARE NOT ACCEPTABLE.
3. Set risk_level based on reversibility:
   - low      → read-only (ls, cat, ps, git log)
   - medium   → creates/modifies files, reversible
   - high     → modifies system state, hard to reverse
   - critical → permanent data loss possible (rm -rf, DROP TABLE, git reset --hard)
4. The command MUST use the correct syntax for Shell: {context.shell}.
   PowerShell uses Get-*, bash uses standard POSIX tools.
5. Provide 1-3 alternatives when practical.
6. The explanation must describe EVERY flag used (e.g. '-la: l=long format, a=show hidden').

DESTRUCTIVE EXAMPLES (is_destructive MUST be true for these):
  rm -rf /tmp/cache         → always destructive
  git reset --hard HEAD~3   → always destructive (loses commits)
  kill -9 1234              → always destructive (force kills process)
  DROP TABLE users;         → always destructive
  dd if=/dev/zero of=/dev/sda → always destructive (wipes disk)
"""

    @property
    def provider_name(self) -> str:
        """Human-readable provider name for logging and telemetry."""
        return self.__class__.__name__.replace("Provider", "").lower()
