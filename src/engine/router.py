"""
Dynamic Complexity Router
===========================
Routes user queries between the local SLM tier (Ollama) and the cloud tier
(DeepSeek / Groq) based on a computed complexity score.

AI Concept: LLM Routing — Lightweight Mixture of Experts (MoE)
================================================================
In production AI systems, you rarely want to send EVERY query to your most
powerful (and expensive) model. Simple queries ("list files") don't need
GPT-4o. Complex multi-step pipelines do.

The solution is LLM Routing:
  - Compute a "complexity score" for the query
  - Below threshold → fast, cheap local model (Ollama, qwen2.5-coder:7b)
  - Above threshold → powerful cloud model (DeepSeek-V3, Groq llama-3.3-70b)

This pattern is used in production by:
  - RouteLLM (Stanford / LMSys) — trained a small classifier to route queries
  - OpenRouter — routes between models based on cost/quality preferences
  - Portkey / LiteLLM — routing + fallback across providers

Our approach uses a heuristic scorer (no ML model needed) because:
  - ML routers need training data you don't have yet
  - Heuristics work surprisingly well for terminal commands
  - Zero latency overhead (runs in microseconds)

Complexity Score Formula:
  score = (query_length × 0.30)
        + (pipe_count   × 0.25)
        + (destructive  × 0.30)  ← always route destructive ops to cloud
        + (git_context  × 0.15)

  score < 0.4  →  LOCAL  (Ollama)
  score ≥ 0.4  →  CLOUD  (DeepSeek)
  --provider flag overrides score unconditionally
"""

from __future__ import annotations

import re

from src.config import get_config
from src.core.context import DESTRUCTIVE_PATTERNS
from src.core.schema import AppConfig, ShellContext
from src.engine.base import LLMProvider
from src.engine.openai_provider import OpenAICompatibleProvider


# ---------------------------------------------------------------------------
# Known destructive CLI keywords (used in complexity scoring)
# ---------------------------------------------------------------------------

_DESTRUCTIVE_KEYWORDS: tuple[str, ...] = (
    "rm", "rmdir", "del", "drop", "truncate", "kill", "pkill",
    "format", "dd", "shred", "git reset", "git push --force",
    "chmod 777", "mkfs",
)

_PIPE_OPERATORS_RE = re.compile(r"[|&;]|>>?")


class Router:
    """
    Routes a query to the appropriate LLM provider based on complexity.

    Usage:
        router = Router(config)

        # Auto-route based on complexity score
        provider = router.resolve(query, context)
        action = await provider.complete(query, context)

        # Force a specific provider
        provider = router.resolve(query, context, force_provider="groq")
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def resolve(
        self,
        query: str,
        context: ShellContext,
        force_provider: str | None = None,
    ) -> LLMProvider:
        """
        Resolve the best provider for this query.

        Args:
            query:          The user's natural language request.
            context:        Shell/OS/Git environment snapshot.
            force_provider: 'local', 'cloud', 'groq', or provider name.
                            When set, bypasses complexity scoring entirely.

        Returns:
            LLMProvider: The selected provider instance, ready to call.

        Decision flow:
            --provider flag set? → use it unconditionally
            complexity_score < threshold? → local (Ollama)
            complexity_score ≥ threshold? → cloud (DeepSeek)
            cloud API key missing? → fallback to local with a warning
        """
        if force_provider:
            return self._build_forced(force_provider)

        score = self.score_complexity(query, context)

        if score < self.config.routing_threshold:
            return self._build_local()
        else:
            # Safety: if no cloud key configured, fall back to local
            if not self.config.cloud_api_key:
                return self._build_local()
            return self._build_cloud()

    def score_complexity(self, query: str, context: ShellContext) -> float:
        """
        Compute a 0.0–1.0 complexity score for the query.

        Higher score = more complex = should go to cloud model.

        Score breakdown:
          - query_length_factor  (0.0–0.30):  longer queries → more complex
          - pipe_factor          (0.0–0.25):  pipes/redirects → multi-step
          - destructive_factor   (0.0–0.30):  dangerous keywords → cloud for safety
          - git_factor           (0.0–0.15):  git repo context → slightly harder

        Returns:
            float: Clamped complexity score in [0.0, 1.0].

        Examples:
            "ls -la"              → ~0.05  (local)
            "find . | grep .py"   → ~0.30  (local, borderline)
            "rm -rf node_modules" → ~0.60  (cloud, destructive)
            complex git rebase    → ~0.70  (cloud)
        """
        score = 0.0

        # Factor 1: Query token length (normalised to 50 tokens = max local)
        token_count = len(query.split())
        length_factor = min(token_count / 50.0, 1.0) * 0.30
        score += length_factor

        # Factor 2: Pipe / redirect / chain operators
        pipe_count = len(_PIPE_OPERATORS_RE.findall(query))
        pipe_factor = min(pipe_count / 4.0, 1.0) * 0.25
        score += pipe_factor

        # Factor 3: Destructive keywords (safety-critical → always cloud)
        query_lower = query.lower()
        is_destructive = any(kw in query_lower for kw in _DESTRUCTIVE_KEYWORDS)
        if is_destructive:
            score += 0.30

        # Factor 4: Git repo context (branch-specific commands are harder)
        if context.is_git_repo and any(
            kw in query_lower for kw in ("git", "branch", "commit", "merge", "rebase")
        ):
            score += 0.15

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Provider factories
    # ------------------------------------------------------------------

    def _build_local(self) -> OpenAICompatibleProvider:
        """Build the local Ollama provider."""
        return OpenAICompatibleProvider(
            config=self.config,
            endpoint=self.config.endpoint,
            model=self.config.model,
            api_key=self.config.api_key,
            provider_name="ollama",
        )

    def _build_cloud(self) -> OpenAICompatibleProvider:
        """Build the primary cloud provider (DeepSeek by default)."""
        return OpenAICompatibleProvider(
            config=self.config,
            endpoint=self.config.cloud_endpoint,
            model=self.config.cloud_model,
            api_key=self.config.cloud_api_key,
            provider_name=self.config.cloud_provider,
        )

    def _build_groq(self) -> OpenAICompatibleProvider:
        """Build the Groq speed-fallback provider."""
        return OpenAICompatibleProvider(
            config=self.config,
            endpoint=self.config.groq_endpoint,
            model=self.config.groq_model,
            api_key=self.config.groq_api_key,
            provider_name="groq",
        )

    def _build_forced(self, name: str) -> LLMProvider:
        """Build the provider explicitly requested via --provider flag."""
        mapping = {
            "local":   self._build_local,
            "ollama":  self._build_local,
            "cloud":   self._build_cloud,
            "deepseek": self._build_cloud,
            "openai":  self._build_cloud,
            "groq":    self._build_groq,
        }
        factory = mapping.get(name.lower())
        if not factory:
            raise ValueError(
                f"Unknown provider '{name}'. "
                f"Valid options: {list(mapping.keys())}"
            )
        return factory()
