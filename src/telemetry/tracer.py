"""
Telemetry Tracer
=================
Records latency, TTFT, token usage, and cache hit/miss events to a
local JSONL file. Zero-dependency, append-only, grep/jq-friendly.

AI Concept: Production LLM Observability
==========================================
In production AI systems, you can't improve what you don't measure.
The four key metrics for LLM-powered applications:

1. LATENCY (ms)
   Total wall-clock time from user input to response rendered.
   SLO targets: p50 < 1500ms, p95 < 5000ms.
   Split into: embed_ms + cache_lookup_ms + llm_ms + render_ms.

2. TTFT — Time-To-First-Token (ms)
   For streaming responses: time from request sent → first token received.
   This is the PRIMARY UX metric. Users perceive TTFT, not total latency.
   ChatGPT, Claude, Gemini all stream specifically to minimize perceived TTFT.
   Target: TTFT < 500ms.

3. TOKEN USAGE
   input_tokens  = tokens in the prompt (system + user messages)
   output_tokens = tokens in the response (command JSON)
   Total cost = (input_tokens × input_price) + (output_tokens × output_price)
   DeepSeek: $0.27/M input, $1.10/M output (extremely cheap for CLI use)

4. CACHE HIT RATE
   What fraction of queries are served from the semantic cache (no LLM call)?
   50% hit rate = 50% cost reduction + 50% faster responses.
   Track this to know if your cache threshold (0.92) is calibrated correctly.

JSONL Format (one JSON object per line):
  {"query_hash": "a3f2...", "provider": "deepseek", "latency_ms": 1243.5, ...}
  {"query_hash": "b1e9...", "provider": "ollama",   "latency_ms": 432.1, "cache_hit": true, ...}

This format is:
  - Readable by any language (JSON Lines standard)
  - Appendable without rewriting the file
  - Queryable with jq: cat telemetry.jsonl | jq '[.latency_ms] | add/length'
  - Importable into Pandas, ClickHouse, BigQuery for future SaaS analytics
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import TELEMETRY_FILE
from src.core.schema import TelemetryRecord

logger = logging.getLogger(__name__)


class Tracer:
    """
    Append-only telemetry recorder for aikord-cli events.

    Every LLM call, cache hit, and execution is recorded as a
    TelemetryRecord and appended to ~/.config/aikord-cli/telemetry.jsonl.

    Usage:
        tracer = Tracer()

        # Record an LLM response
        tracer.record(
            query="list all docker containers",
            provider="deepseek",
            model="deepseek-chat",
            latency_ms=1243.5,
            ttft_ms=312.0,
            input_tokens=450,
            output_tokens=87,
            cache_hit=False,
            was_destructive=False,
        )
    """

    def __init__(self, output_path: Path | None = None) -> None:
        self.output_path = output_path or TELEMETRY_FILE
        self._session_records: list[TelemetryRecord] = []

    def record(
        self,
        query: str,
        provider: str,
        model: str,
        latency_ms: float,
        ttft_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        was_destructive: bool = False,
        retry_count: int = 0,
    ) -> TelemetryRecord:
        """
        Create and persist a TelemetryRecord.

        The query is hashed (SHA256) before storage for privacy.
        We never store the raw query text in telemetry.

        Args:
            query:          The original natural language query.
            provider:       Provider name (ollama, deepseek, groq).
            model:          Model name (qwen2.5-coder:7b, deepseek-chat).
            latency_ms:     Total end-to-end latency.
            ttft_ms:        Time to first token (streaming only).
            input_tokens:   Prompt token count.
            output_tokens:  Completion token count.
            cache_hit:      True if result came from semantic cache.
            was_destructive: True if the command was flagged destructive.
            retry_count:    Number of self-healing retries used.

        Returns:
            TelemetryRecord: The recorded event.
        """
        record = TelemetryRecord(
            query_hash=self._hash_query(query),
            provider=provider,
            model=model,
            latency_ms=round(latency_ms, 2),
            ttft_ms=round(ttft_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
            was_destructive=was_destructive,
            retry_count=retry_count,
            timestamp=datetime.now(tz=timezone.utc),
        )

        self._session_records.append(record)
        self._append_to_file(record)
        return record

    def session_summary(self) -> dict:
        """
        Compute summary statistics for the current session.

        Returns:
            dict with p50/p95 latency, total cost estimate, hit rate, etc.
        """
        if not self._session_records:
            return {"total_queries": 0}

        latencies = sorted(r.latency_ms for r in self._session_records)
        n = len(latencies)
        hits = sum(1 for r in self._session_records if r.cache_hit)
        total_cost = sum(r.estimated_cost_usd for r in self._session_records)

        def percentile(data: list[float], p: int) -> float:
            idx = max(0, int(len(data) * p / 100) - 1)
            return round(data[idx], 2)

        return {
            "total_queries":     n,
            "cache_hits":        hits,
            "cache_hit_rate":    f"{hits / n * 100:.1f}%" if n else "0%",
            "latency_p50_ms":    percentile(latencies, 50),
            "latency_p95_ms":    percentile(latencies, 95),
            "total_cost_usd":    round(total_cost, 4),
            "total_tokens":      sum(r.total_tokens for r in self._session_records),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_to_file(self, record: TelemetryRecord) -> None:
        """Append a single record as a JSON line to the telemetry file."""
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            line = record.model_dump_json() + "\n"
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            # Never crash the main flow for telemetry failures
            logger.warning(f"Telemetry write failed: {e}")

    @staticmethod
    def _hash_query(query: str) -> str:
        """SHA256 hash the query for privacy-preserving storage."""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
