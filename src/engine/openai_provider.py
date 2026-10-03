"""
OpenAI-Compatible HTTP Provider
=================================
A single httpx client that works with ANY provider implementing the
OpenAI /v1/chat/completions wire format:
  - Ollama       (http://localhost:11434/v1)
  - DeepSeek     (https://api.deepseek.com/v1)
  - Groq         (https://api.groq.com/openai/v1)
  - OpenAI       (https://api.openai.com/v1)
  - Together AI  (https://api.together.xyz/v1)

AI Concept: The OpenAI Wire Format as an Industry Standard
============================================================
In 2023, OpenAI published their /v1/chat/completions API specification.
Because OpenAI dominated the market, every other provider adopted the
SAME request/response format to be drop-in compatible.

Request shape (simplified):
  POST /v1/chat/completions
  {
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "Hello!"}
    ],
    "response_format": {"type": "json_object"},  ← enforces JSON output
    "stream": false
  }

This means ONE HTTP client (this file) can talk to all providers.
You just change the base URL and API key — the JSON is identical.

AI Concept: Time-To-First-Token (TTFT) Measurement
====================================================
We measure two latency metrics:
  1. Total latency: time from request sent → full response received
  2. TTFT: time from request sent → first chunk received (streaming only)

TTFT is the most important UX metric for LLMs. Users perceive an LLM as
"fast" if the first word appears quickly, even if total time is similar.

ChatGPT, Claude, and Gemini all stream by default specifically to minimize
perceived TTFT. A 2-second response feels instant if words start appearing
at 300ms.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx
from pydantic import ValidationError

from src.core.schema import AppConfig, ErrorRemediation, ShellContext, TerminalAction
from src.engine.base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """
    Generic provider for any OpenAI /v1/chat/completions compatible endpoint.

    Usage:
        # Ollama (local)
        provider = OpenAICompatibleProvider(
            config, endpoint="http://localhost:11434/v1",
            model="qwen2.5-coder:7b", api_key="ollama"
        )

        # DeepSeek (cloud)
        provider = OpenAICompatibleProvider(
            config, endpoint="https://api.deepseek.com/v1",
            model="deepseek-chat", api_key="sk-..."
        )

        action = await provider.complete("list all docker containers", ctx)
    """

    def __init__(
        self,
        config: AppConfig,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        provider_name: str = "openai-compatible",
    ) -> None:
        super().__init__(config)
        self._endpoint = (endpoint or config.endpoint).rstrip("/")
        self._model = model or config.model
        self._api_key = api_key or config.api_key
        self._provider_label = provider_name

        # httpx async client — reused across requests for connection pooling
        # Connection pooling: reuse TCP connections instead of opening a new
        # one for every request. This saves ~100-300ms per call.
        self._client = httpx.AsyncClient(
            base_url=self._endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.request_timeout, connect=5.0),
        )

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    async def complete(
        self,
        query: str,
        context: ShellContext,
        system_prompt: str | None = None,
    ) -> TerminalAction:
        """
        Non-streaming completion → parse JSON → validate with Pydantic.

        The flow:
          1. Build system prompt with environment context
          2. POST to /v1/chat/completions with response_format=json_object
          3. Extract the 'content' string from the response
          4. Parse JSON → TerminalAction (Pydantic validates)
          5. Apply secondary destructive check

        Args:
            query: Natural language user request.
            context: Shell/OS/Git context snapshot.
            system_prompt: Optional override for the system prompt.

        Returns:
            TerminalAction: Validated structured response.

        Raises:
            ValueError: LLM returned malformed JSON or schema mismatch.
            httpx.HTTPStatusError: API returned 4xx/5xx.
        """
        sys_prompt = system_prompt or self._build_system_prompt(context)
        payload = self._build_payload(query, sys_prompt, stream=False)

        start = time.perf_counter()
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Provider {self._provider_label} returned HTTP {e.response.status_code}: "
                f"{e.response.text[:200]}"
            ) from e

        elapsed_ms = (time.perf_counter() - start) * 1000

        data = response.json()
        raw_content = self._extract_content(data)
        action = self._parse_action(raw_content)

        # Attach timing to the action for telemetry (duck-typed extra field)
        action.__dict__["_latency_ms"] = elapsed_ms
        action.__dict__["_tokens"] = data.get("usage", {})

        return action

    async def stream_complete(
        self,
        query: str,
        context: ShellContext,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming completion — yields tokens as they arrive.

        AI Concept: Server-Sent Events (SSE) Streaming
        ================================================
        OpenAI-compatible streaming uses SSE (Server-Sent Events):
          - Server sends chunks: 'data: {"choices":[{"delta":{"content":"ls"}}]}\n\n'
          - Each line is a partial JSON object
          - Stream ends with: 'data: [DONE]\n\n'

        We collect all chunks, then parse the COMPLETE JSON at the end.
        Why? Because TerminalAction JSON must be complete to validate.
        We stream for TTFT measurement and future partial-render support.

        Yields:
            str: Text chunks as they arrive (for display / TTFT measurement).
        """
        sys_prompt = system_prompt or self._build_system_prompt(context)
        payload = self._build_payload(query, sys_prompt, stream=True)

        full_content = ""
        first_chunk = True
        ttft_ms = 0.0
        start = time.perf_counter()

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break

                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content", "")
                if not text:
                    continue

                if first_chunk:
                    ttft_ms = (time.perf_counter() - start) * 1000
                    first_chunk = False

                full_content += text
                yield text

        # Store TTFT for telemetry access
        self.__dict__["_last_ttft_ms"] = ttft_ms
        self.__dict__["_last_content"] = full_content

    async def complete_remediation(
        self,
        original_command: str,
        error_output: str,
        context: ShellContext,
    ) -> ErrorRemediation:
        """
        Ask the LLM to diagnose and fix a failed command.

        This is called by healer.py in the ReAct self-correction loop.
        The LLM receives the original command + stderr and must return
        an ErrorRemediation JSON object with the corrected command.

        Args:
            original_command: The command that failed.
            error_output: Captured stderr/stdout from the failure.
            context: Current shell context.

        Returns:
            ErrorRemediation: Validated self-correction response.
        """
        remediation_schema = ErrorRemediation.model_json_schema()

        system_prompt = f"""You are aikord-cli's self-healing agent.
A command was executed and failed. Your job: diagnose the error and provide a corrected command.

USER ENVIRONMENT:
{context.to_prompt_string()}

OUTPUT FORMAT — respond with ONLY valid JSON matching this schema:
{remediation_schema}

RULES:
1. Output ONLY raw JSON, no markdown, no prose.
2. The corrected_command must be syntactically valid for Shell: {context.shell}.
3. Set is_destructive=true if the corrected command can delete/overwrite data.
4. Be precise in explanation: identify the exact cause of failure.
"""
        user_message = (
            f"ORIGINAL COMMAND:\n{original_command}\n\n"
            f"ERROR OUTPUT:\n{error_output[:2000]}"  # Cap to avoid token overflow
        )

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,  # Low temp for deterministic fixes
        }

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        raw = self._extract_content(response.json())

        try:
            return ErrorRemediation.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as e:
            raise ValueError(f"LLM returned invalid remediation JSON: {e}") from e

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self, query: str, system_prompt: str, stream: bool = False
    ) -> dict[str, Any]:
        """
        Build the /v1/chat/completions request payload.

        AI Concept: Temperature in LLMs
        =================================
        Temperature controls randomness of the output:
          - temperature=0.0 → most deterministic, always picks highest-prob token
          - temperature=1.0 → creative, sometimes unpredictable
          - temperature=0.1 → our sweet spot: reliable JSON, slight variation

        For structured output (JSON), always use low temperature (0.0-0.2).
        High temperature makes the LLM more likely to produce invalid JSON.
        """
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "response_format": {"type": "json_object"},  # Enforces JSON output
            "temperature": 0.1,
            "max_tokens": 512,  # Commands are short; cap to save cost
            "stream": stream,
        }

    def _extract_content(self, data: dict) -> str:
        """Extract the text content from a non-streaming API response."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected API response structure: {data}") from e

    def _parse_action(self, raw: str) -> TerminalAction:
        """
        Parse raw JSON string into a validated TerminalAction.

        We handle two common LLM formatting mistakes:
          1. JSON wrapped in code fences: ```json {...} ```
          2. Leading/trailing whitespace
        """
        # Strip markdown code fences if the LLM ignored our instructions
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Drop first line (```json or ```) and last line (```)
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            return TerminalAction.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError) as e:
            raise ValueError(
                f"LLM response is not valid TerminalAction JSON.\n"
                f"Raw response: {raw[:300]}\n"
                f"Error: {e}"
            ) from e

    @property
    def provider_name(self) -> str:
        return self._provider_label
