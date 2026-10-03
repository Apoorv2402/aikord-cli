"""
Tests: cache/vector_cache.py
==============================
Tests for the DuckDB VSS semantic cache: embedding, store, retrieve,
threshold enforcement, and cache stats.

Strategy: use an in-memory DuckDB connection to avoid touching disk.
We mock the sentence-transformers model to avoid slow ML model loading.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.schema import TerminalAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_action(command: str = "ls -la", destructive: bool = False) -> TerminalAction:
    return TerminalAction(
        command=command,
        explanation="Test explanation",
        is_destructive=destructive,
        risk_level="low" if not destructive else "high",
    )


def _make_embedding(seed: int = 42, dim: int = 384) -> np.ndarray:
    """Create a deterministic, unit-normalised test embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.random(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# VectorCache with mocked embedding model
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cache(tmp_path):
    """
    VectorCache backed by a real (temporary) DuckDB file.
    The sentence-transformers model is mocked for fast test execution.
    """
    from src.cache.vector_cache import VectorCache

    db_path = tmp_path / "test_cache.duckdb"
    cache = VectorCache(db_path=db_path, threshold=0.92)

    # Patch the embedding model
    mock_model = MagicMock()
    # Default: return a fixed embedding (seed=42)
    mock_model.encode.return_value = _make_embedding(seed=42)
    cache._model = mock_model

    yield cache
    cache.close()


# ---------------------------------------------------------------------------
# Cache ID generation
# ---------------------------------------------------------------------------

class TestCacheID:
    def test_make_id_is_deterministic(self):
        from src.cache.vector_cache import VectorCache
        id1 = VectorCache._make_id("list files")
        id2 = VectorCache._make_id("list files")
        assert id1 == id2

    def test_make_id_is_case_normalised(self):
        from src.cache.vector_cache import VectorCache
        id1 = VectorCache._make_id("List Files")
        id2 = VectorCache._make_id("list files")
        assert id1 == id2

    def test_different_queries_different_ids(self):
        from src.cache.vector_cache import VectorCache
        id1 = VectorCache._make_id("ls -la")
        id2 = VectorCache._make_id("rm -rf /")
        assert id1 != id2

    def test_id_is_16_hex_chars(self):
        from src.cache.vector_cache import VectorCache
        cache_id = VectorCache._make_id("test query")
        assert len(cache_id) == 16
        assert all(c in "0123456789abcdef" for c in cache_id)


# ---------------------------------------------------------------------------
# Cache miss on empty cache
# ---------------------------------------------------------------------------

class TestCacheMiss:
    def test_empty_cache_returns_none(self, mock_cache):
        action, score = mock_cache.get("list all files")
        assert action is None
        assert score == 0.0

    def test_miss_below_threshold(self, mock_cache):
        """Store one embedding, query with a VERY different embedding → miss."""
        action = _make_action("ls -la")
        # Store with embedding seed=42
        mock_cache._model.encode.return_value = _make_embedding(seed=42)
        mock_cache.set("list files", action)

        # Query with a completely orthogonal embedding
        mock_cache._model.encode.return_value = _make_embedding(seed=999)
        result, score = mock_cache.get("totally different query about databases")
        # Score should be below threshold (0.92) for very different embeddings
        assert result is None or score < mock_cache.threshold


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_exact_same_embedding_is_hit(self, mock_cache):
        """Same query text → same embedding → similarity = 1.0 → hit."""
        action = _make_action("ls -la")
        embedding = _make_embedding(seed=42)

        mock_cache._model.encode.return_value = embedding
        mock_cache.set("list files", action)

        # Query with identical embedding
        mock_cache._model.encode.return_value = embedding
        result, score = mock_cache.get("list files")

        assert result is not None
        assert result.command == "ls -la"
        assert score >= 0.92

    def test_returned_action_matches_stored(self, mock_cache):
        """Retrieved action must match the stored TerminalAction exactly."""
        action = _make_action("docker ps --all", destructive=False)
        embedding = _make_embedding(seed=10)

        mock_cache._model.encode.return_value = embedding
        mock_cache.set("show all docker containers", action)

        mock_cache._model.encode.return_value = embedding
        result, score = mock_cache.get("show all docker containers")

        assert result is not None
        assert result.command == "docker ps --all"
        assert result.is_destructive is False


# ---------------------------------------------------------------------------
# Cache set / overwrite
# ---------------------------------------------------------------------------

class TestCacheSet:
    def test_set_and_retrieve(self, mock_cache):
        action = _make_action("git status")
        embedding = _make_embedding(seed=5)
        mock_cache._model.encode.return_value = embedding
        mock_cache.set("check git status", action)

        mock_cache._model.encode.return_value = embedding
        result, _ = mock_cache.get("check git status")
        assert result is not None
        assert result.command == "git status"

    def test_overwrite_replaces_entry(self, mock_cache):
        """INSERT OR REPLACE should update an existing cache entry."""
        embedding = _make_embedding(seed=7)
        mock_cache._model.encode.return_value = embedding

        action_v1 = _make_action("ls")
        mock_cache.set("list", action_v1)

        action_v2 = _make_action("ls -la")
        mock_cache.set("list", action_v2)

        mock_cache._model.encode.return_value = embedding
        result, _ = mock_cache.get("list")
        assert result is not None
        assert result.command == "ls -la"  # Should be the updated version


# ---------------------------------------------------------------------------
# Cache clear
# ---------------------------------------------------------------------------

class TestCacheClear:
    def test_clear_returns_count(self, mock_cache):
        embedding = _make_embedding(seed=1)
        mock_cache._model.encode.return_value = embedding
        mock_cache.set("query one", _make_action("ls"))
        mock_cache._model.encode.return_value = _make_embedding(seed=2)
        mock_cache.set("query two", _make_action("pwd"))

        count = mock_cache.clear()
        assert count == 2

    def test_clear_empties_cache(self, mock_cache):
        embedding = _make_embedding(seed=1)
        mock_cache._model.encode.return_value = embedding
        mock_cache.set("list files", _make_action("ls"))

        mock_cache.clear()

        mock_cache._model.encode.return_value = embedding
        result, score = mock_cache.get("list files")
        assert result is None


# ---------------------------------------------------------------------------
# Cache stats
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_stats_on_empty_cache(self, mock_cache):
        stats = mock_cache.stats()
        assert stats["total_entries"] == 0
        assert stats["total_hits"] == 0

    def test_stats_after_set(self, mock_cache):
        embedding = _make_embedding(seed=3)
        mock_cache._model.encode.return_value = embedding
        mock_cache.set("list files", _make_action("ls"))

        stats = mock_cache.stats()
        assert stats["total_entries"] == 1

    def test_stats_threshold_matches_config(self, mock_cache):
        stats = mock_cache.stats()
        assert stats["threshold"] == 0.92


# ---------------------------------------------------------------------------
# Pydantic serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_terminal_action_json_roundtrip(self):
        """TerminalAction must survive JSON serialise → deserialise intact."""
        original = TerminalAction(
            command="find . -name '*.py' -mtime -7",
            explanation="Find Python files modified in last 7 days.",
            is_destructive=False,
            risk_level="low",
            alternatives=["git diff --name-only HEAD~7"],
        )
        json_str = original.model_dump_json()
        restored = TerminalAction.model_validate_json(json_str)

        assert restored.command == original.command
        assert restored.is_destructive == original.is_destructive
        assert restored.risk_level == original.risk_level
        assert restored.alternatives == original.alternatives
