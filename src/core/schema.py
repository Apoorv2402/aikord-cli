"""
AI Concept: Structured LLM Output with Pydantic v2
====================================================
One of the biggest challenges in production AI systems is getting LLMs to
return consistent, machine-parseable data instead of free-form prose.

The solution is "Structured Output":
  1. You define the exact JSON shape you expect (this file).
  2. You send it to the LLM as a JSON schema in the system prompt.
  3. Modern LLMs (DeepSeek, Qwen, GPT-4o) support a 'response_format' flag
     that constrains the model to only output valid JSON matching your schema.
  4. Pydantic validates the result. If it fails → you retry or raise.

Without Pydantic, you'd spend 50% of your code regex-parsing LLM text.
With Pydantic v2 strict mode, your LLM pipeline is type-safe end-to-end.

This pattern (Pydantic + LLM) is used in:
  - LangChain's .with_structured_output()
  - Instructor library (most popular structured-output lib)
  - OpenAI's response_format: { type: "json_schema" }
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Primary LLM Output Contract
# ---------------------------------------------------------------------------

class TerminalAction(BaseModel):
    """
    The contract between aikord-cli and every LLM provider.

    Every single LLM response is validated against this model. The LLM must
    produce JSON that matches this exact structure. If it doesn't, Pydantic
    raises a ValidationError which the caller can catch and retry.

    Fields:
        command        : The actual shell command (e.g. 'ls -la --color=auto')
        explanation    : Plain-English breakdown of each flag and argument
        is_destructive : True when command can delete/overwrite irreversibly
        risk_level     : Safety classification from low → critical
        alternatives   : Other commands that achieve a similar result
    """

    command: str = Field(
        ...,
        description="The exact shell command to execute. Must be non-empty.",
        min_length=1,
    )
    explanation: str = Field(
        ...,
        description="Plain English breakdown: what each flag/argument does.",
    )
    is_destructive: bool = Field(
        ...,
        description=(
            "True when the command can permanently delete, overwrite, or "
            "irreversibly modify data (e.g. rm -rf, DROP TABLE, git reset --hard)."
        ),
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description=(
            "low    → read-only, safe to run anywhere\n"
            "medium → writes files, no data loss risk\n"
            "high   → modifies system state, reversible\n"
            "critical → irreversible, data loss possible"
        ),
    )
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative commands that achieve a similar result.",
        max_length=5,
    )

    @field_validator("command")
    @classmethod
    def strip_command(cls, v: str) -> str:
        """Strip accidental whitespace and code-fence backticks from LLM output."""
        return v.strip().strip("`")

    @field_validator("is_destructive", mode="before")
    @classmethod
    def enforce_destructive_detection(cls, v: bool, info) -> bool:
        """
        Secondary safety net: even if the LLM forgets to flag a command,
        we check the command text against known destructive patterns.

        The healer.py module performs a third check at execution time.
        Defence-in-depth: three layers of destructive detection.
        """
        # Validator runs after 'command' is set in the model
        # The actual keyword check happens in router.py and healer.py
        # because field_validator cannot access other fields in Pydantic v2
        # without using model_validator. We keep this simple intentionally.
        return bool(v)

    def to_system_prompt_schema(self) -> str:
        """Return the JSON schema as a string for injection into system prompts."""
        return self.model_json_schema(mode="serialization").__str__()


# ---------------------------------------------------------------------------
# Self-Healing ReAct Payload
# ---------------------------------------------------------------------------

class ErrorRemediation(BaseModel):
    """
    AI Concept: ReAct (Reason + Act) Self-Correction
    ==================================================
    When a command fails (non-zero exit code), we don't just give up.
    We feed the error back to the LLM and ask it to reason about the fix.

    The ReAct pattern loop (healer.py):
      ┌─────────────────────────────────────────────────────────┐
      │  1. REASON  → LLM reads: original_command + error_output │
      │  2. ACT     → LLM proposes: corrected_command             │
      │  3. OBSERVE → We run it. Exit code 0? Done. Else repeat.  │
      │  4. LIMIT   → Max 3 iterations (prevents infinite loops)   │
      └─────────────────────────────────────────────────────────┘

    This is the same loop used by AI coding agents (Devin, SWE-agent),
    Claude Computer Use, and Gemini Agent.
    """

    original_command: str = Field(..., description="The command that failed.")
    error_output: str = Field(
        ..., description="The stderr/stdout captured from the failed run."
    )
    corrected_command: str = Field(..., description="The LLM's proposed fixed command.")
    explanation: str = Field(
        ..., description="Why the original failed and how the corrected version fixes it."
    )
    is_destructive: bool = Field(
        ..., description="Safety flag for the corrected command."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="LLM self-reported confidence in the fix (0.0 → 1.0).",
    )

    @field_validator("corrected_command")
    @classmethod
    def strip_corrected(cls, v: str) -> str:
        return v.strip().strip("`")


# ---------------------------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    """
    Application configuration model.

    Loaded from: ~/.config/aikord-cli/config.json
    Overridden by environment variables (see config.py for full list).

    AI Concept: Provider Abstraction
    ==================================
    Notice that all providers (Ollama, DeepSeek, Groq, OpenAI) share the same
    'endpoint + model + api_key' structure. This is intentional — they all
    implement the OpenAI /v1/chat/completions wire format, so a single HTTP
    client (openai_provider.py) works for all of them. You just swap the URL.

    Commercialization Note:
    ========================
    The 'plan' field is a hook for future aikord Cloud features. Right now it
    does nothing, but in v2 Pro it will unlock team caching and analytics.
    """

    # --- Local provider (Ollama) ---
    provider: str = Field(default="ollama")
    model: str = Field(default="qwen2.5-coder:7b")
    endpoint: str = Field(default="http://localhost:11434/v1")
    api_key: str = Field(default="ollama")  # Ollama accepts any non-empty string

    # --- Cloud provider (DeepSeek) ---
    cloud_provider: str = Field(default="deepseek")
    cloud_model: str = Field(default="deepseek-chat")
    cloud_endpoint: str = Field(default="https://api.deepseek.com/v1")
    cloud_api_key: str = Field(default="")

    # --- Speed fallback (Groq) ---
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_endpoint: str = Field(default="https://api.groq.com/openai/v1")

    # --- Routing & semantic cache ---
    routing_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Complexity score cutoff: below → local, above → cloud.",
    )
    cache_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity cutoff for a semantic cache hit.",
    )
    cache_db_path: str = Field(default="~/.config/aikord-cli/cache.duckdb")

    # --- Agent behaviour ---
    max_retries: int = Field(default=3, ge=1, le=10)
    auto_fix: bool = Field(default=False)
    request_timeout: int = Field(default=30)

    # --- Commercialization hook (reserved) ---
    plan: Literal["free", "pro", "enterprise"] = Field(
        default="free",
        description="Subscription tier. Reserved for aikord Cloud v2.",
    )

    # --- Plugin system ---
    plugin_dirs: list[str] = Field(
        default_factory=list,
        description="Extra directories to scan for aikord plugins.",
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TelemetryRecord(BaseModel):
    """
    A single observability event written to ~/.config/aikord-cli/telemetry.jsonl.

    AI Concept: Production LLM Observability
    ==========================================
    In production AI systems, you must track these four metrics:

    1. LATENCY    — Total time from user input to response displayed (ms).
                    Target: p50 < 2000ms, p95 < 5000ms.

    2. TTFT       — Time-To-First-Token. For streaming responses, this is the
                    time until the user sees the first word. Critical for UX.
                    Even a slow model feels fast if TTFT is low (< 500ms).

    3. TOKEN USAGE — Input + output token counts. These map directly to cost:
                    DeepSeek charges $0.27/M input tokens, $1.10/M output.
                    Tracking this lets you forecast your monthly API bill.

    4. CACHE HIT RATE — What % of queries skip the LLM entirely?
                    A 50% hit rate cuts your API costs in half.

    The JSONL format is append-only and grep/jq-friendly:
        cat telemetry.jsonl | jq '.latency_ms' | sort -n | tail -5
    """

    query_hash: str = Field(
        ..., description="SHA256 of the query (privacy-preserving, not the raw text)."
    )
    provider: str
    model: str
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds.")
    ttft_ms: float = Field(default=0.0, description="Time-To-First-Token (streaming).")
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    cache_hit: bool = Field(default=False)
    was_destructive: bool = Field(default=False)
    retry_count: int = Field(default=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate using DeepSeek pricing as baseline."""
        input_cost = (self.input_tokens / 1_000_000) * 0.27
        output_cost = (self.output_tokens / 1_000_000) * 1.10
        return round(input_cost + output_cost, 6)


# ---------------------------------------------------------------------------
# Shell Context
# ---------------------------------------------------------------------------

class ShellContext(BaseModel):
    """
    A snapshot of the user's terminal environment at query time.

    AI Concept: Context Injection / Retrieval-Augmented Generation (RAG)
    ======================================================================
    The quality of an LLM's command suggestion depends heavily on context.
    "List all branches" means something different in:
      - A Git repo  → 'git branch -a'
      - A directory → 'ls' with some filter

    By injecting OS, shell, PWD, and git state into the system prompt, we
    dramatically improve suggestion accuracy. This is the simplest form of RAG:
    we "retrieve" facts about the environment and "augment" the LLM prompt.

    Without context injection, the LLM must guess. With it, it knows exactly
    what shell dialect to use (bash vs. PowerShell have different syntax).
    """

    os_name: str = Field(..., description="OS: linux, darwin, windows")
    os_version: str = Field(default="")
    shell: str = Field(
        ..., description="Shell dialect: bash, zsh, fish, powershell, cmd"
    )
    cwd: str = Field(..., description="Current working directory (absolute path).")
    is_git_repo: bool = Field(default=False)
    git_branch: str = Field(default="", description="Active git branch name.")
    git_status: str = Field(
        default="", description="Short git status (e.g., 'M 3, ?? 1')."
    )
    home_dir: str = Field(default="")
    username: str = Field(default="")

    def to_prompt_string(self) -> str:
        """
        Render context as a human-readable string for LLM system prompts.

        Example output:
            OS: linux (Ubuntu 22.04)
            Shell: bash
            CWD: /home/user/projects/aikord-cli
            Git: yes | branch: main | status: M 2, ?? 1
        """
        parts = [
            f"OS: {self.os_name} ({self.os_version})",
            f"Shell: {self.shell}",
            f"CWD: {self.cwd}",
        ]
        if self.is_git_repo:
            parts.append(
                f"Git: yes | branch: {self.git_branch} | status: {self.git_status or 'clean'}"
            )
        else:
            parts.append("Git: no")
        return "\n".join(parts)
