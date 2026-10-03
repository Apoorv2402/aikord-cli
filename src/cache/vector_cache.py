"""
Semantic Vector Cache — DuckDB VSS
=====================================
An embedded, persistent vector cache using DuckDB's VSS extension
with an HNSW (Hierarchical Navigable Small World) index for approximate
nearest-neighbour search at sub-2ms latency.

AI Concept: Semantic Caching vs. Exact-String Caching
=======================================================
Traditional caches store the exact string key. A semantic cache stores
the EMBEDDING of the key — a vector of numbers capturing its meaning.

  String cache hit:  "list files" == "list files"          (exact match only)
  Semantic cache hit: "list files" ≈ "show directory files" (meaning match)

This is possible because embedding models (like all-MiniLM-L6-v2) map
semantically similar sentences to nearby points in vector space.

AI Concept: Embeddings
========================
An embedding is a fixed-length numerical representation of text.
Sentence "list files in current directory" becomes a vector like:
  [0.12, -0.45, 0.78, ..., 0.33]  ← 384 numbers (all-MiniLM-L6-v2)

The key property: similar sentences → similar vectors.
Mathematically: cos(θ) between two embedding vectors measures similarity.
  cos(θ) = 1.0 → identical meaning
  cos(θ) = 0.92 → almost same meaning (our hit threshold)
  cos(θ) = 0.0 → completely unrelated

AI Concept: HNSW Index (Hierarchical Navigable Small World)
===========================================================
A brute-force nearest-neighbour search over 10,000 cached vectors would
compare the query vector against all 10,000 entries. Slow!

HNSW builds a hierarchical graph structure that lets you find the nearest
neighbours in O(log N) time instead of O(N). It's the standard algorithm
used in production vector databases (Pinecone, Weaviate, Chroma, pgvector).

DuckDB's VSS extension implements HNSW natively, giving us production-grade
vector search with zero external dependencies.

AI Concept: RAG (Retrieval-Augmented Generation)
==================================================
This cache IS a simplified RAG system:
  1. Embed the query (retrieval step)
  2. Search vector DB for similar past queries (retrieval step)
  3. Return the cached result (generation step — but instant, no LLM needed!)

Full RAG (used in ChatPDF, Perplexity) adds step 2b: retrieve relevant
documents from the DB and feed them to the LLM for context. We skip 2b
because we cache the FINAL LLM output, not source documents.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np

from src.core.schema import TerminalAction

logger = logging.getLogger(__name__)


class VectorCache:
    """
    Semantic cache backed by DuckDB VSS with an HNSW cosine index.

    Lifecycle:
        cache = VectorCache(Path("~/.config/aikord-cli/cache.duckdb").expanduser())
        # On cache miss:
        action, score = cache.get("list all docker containers")  # (None, 0.0)
        # After LLM responds:
        cache.set("list all docker containers", action)
        # On next similar query:
        action, score = cache.get("show running docker containers")  # (action, 0.94)
    """

    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors

    def __init__(
        self,
        db_path: Path,
        threshold: float = 0.92,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.db_path = db_path
        self.threshold = threshold
        self._model_name = embedding_model
        self._model = None  # Lazy-loaded (slow import, ~2s first time)

        # Connect to DuckDB (creates file if not exists)
        self.conn = duckdb.connect(str(db_path))
        self._setup_db()

    def _setup_db(self) -> None:
        """
        Bootstrap the DuckDB database with the VSS extension and schema.

        DuckDB Extension System:
          DuckDB supports installable extensions (similar to Postgres extensions).
          'vss' adds vector similarity search capabilities including HNSW indexing.
          INSTALL downloads it once; LOAD activates it for this session.
        """
        try:
            self.conn.execute("INSTALL vss;")
            self.conn.execute("LOAD vss;")
        except Exception as e:
            logger.warning(f"VSS extension setup issue: {e}. Cache may be slower.")

        # Create the cache table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS query_cache (
                id          VARCHAR PRIMARY KEY,
                query_text  TEXT NOT NULL,
                embedding   FLOAT[{self.EMBEDDING_DIM}] NOT NULL,
                action_json JSON NOT NULL,
                hit_count   INTEGER DEFAULT 0,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_hit_at TIMESTAMP
            )
        """)

        # Create HNSW index for fast cosine similarity search
        # HNSW with cosine metric: ORDER BY array_cosine_distance will use this index
        try:
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS cache_hnsw_idx
                ON query_cache
                USING HNSW (embedding)
                WITH (metric = 'cosine')
            """)
        except Exception as e:
            logger.warning(f"HNSW index creation failed (will use brute force): {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> tuple[Optional[TerminalAction], float]:
        """
        Look up a query in the semantic cache.

        Steps:
          1. Embed the query into a 384-dim vector   (~5ms, local model)
          2. Run HNSW ANN search in DuckDB           (~1ms with index)
          3. Compute cosine similarity of top result
          4. If similarity ≥ threshold → cache HIT   (return action + score)
          5. Else                      → cache MISS  (return None, 0.0)

        Args:
            query: The user's natural language query.

        Returns:
            Tuple of (TerminalAction or None, similarity_score).
            If None, the caller should query the LLM.

        Performance target: < 10ms total for cache hits.
        """
        start = time.perf_counter()
        embedding = self._embed(query)
        embed_ms = (time.perf_counter() - start) * 1000

        try:
            # The HNSW index is used automatically when:
            #   ORDER BY array_cosine_distance(col, vec) LIMIT k
            result = self.conn.execute(f"""
                SELECT
                    action_json,
                    (1.0 - array_cosine_distance(
                        embedding,
                        $1::FLOAT[{self.EMBEDDING_DIM}]
                    )) AS similarity
                FROM query_cache
                ORDER BY array_cosine_distance(
                    embedding,
                    $1::FLOAT[{self.EMBEDDING_DIM}]
                )
                LIMIT 1
            """, [embedding.tolist()]).fetchone()

        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            return None, 0.0

        total_ms = (time.perf_counter() - start) * 1000

        if result is None:
            logger.debug(f"Cache MISS (empty cache) | {total_ms:.1f}ms")
            return None, 0.0

        action_json, similarity = result[0], float(result[1])

        if similarity >= self.threshold:
            # Increment hit counter asynchronously-friendly way
            cache_id = self._make_id(query)
            self.conn.execute("""
                UPDATE query_cache
                SET hit_count = hit_count + 1, last_hit_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, [cache_id])

            action = TerminalAction.model_validate_json(action_json)
            logger.info(
                f"Cache HIT | similarity={similarity:.4f} | "
                f"embed={embed_ms:.1f}ms | total={total_ms:.1f}ms"
            )
            return action, similarity

        logger.debug(
            f"Cache MISS | best_sim={similarity:.4f} < {self.threshold} | {total_ms:.1f}ms"
        )
        return None, 0.0

    def set(self, query: str, action: TerminalAction) -> None:
        """
        Store a validated command in the semantic cache.

        Only called after a command succeeds (exit code 0).
        We never cache failed commands — that would poison the cache.

        Args:
            query:  The original natural language query.
            action: The validated TerminalAction to store.
        """
        embedding = self._embed(query)
        cache_id = self._make_id(query)

        try:
            self.conn.execute(f"""
                INSERT OR REPLACE INTO query_cache (id, query_text, embedding, action_json)
                VALUES (?, ?, ?::FLOAT[{self.EMBEDDING_DIM}], ?)
            """, [
                cache_id,
                query,
                embedding.tolist(),
                action.model_dump_json(),
            ])
            logger.info(f"Cache SET | id={cache_id} | query='{query[:50]}'")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def clear(self) -> int:
        """
        Delete all entries from the cache.

        Returns:
            int: Number of entries deleted.
        """
        count = self.conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        self.conn.execute("DELETE FROM query_cache")
        logger.info(f"Cache CLEARED | {count} entries removed")
        return count

    def stats(self) -> dict:
        """Return cache statistics for the `aikord cache stats` command."""
        row = self.conn.execute("""
            SELECT
                COUNT(*)                        AS total_entries,
                COALESCE(SUM(hit_count), 0)     AS total_hits,
                COALESCE(AVG(hit_count), 0)     AS avg_hits_per_entry,
                MIN(created_at)                 AS oldest_entry,
                MAX(last_hit_at)                AS most_recent_hit
            FROM query_cache
        """).fetchone()

        return {
            "total_entries": row[0],
            "total_hits": row[1],
            "avg_hits_per_entry": round(row[2], 2),
            "oldest_entry": str(row[3]) if row[3] else None,
            "most_recent_hit": str(row[4]) if row[4] else None,
            "db_path": str(self.db_path),
            "threshold": self.threshold,
        }

    def close(self) -> None:
        """Close the DuckDB connection cleanly."""
        self.conn.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """
        Convert text into a 384-dimensional embedding vector.

        Lazy-loads the sentence-transformers model on first call.
        Subsequent calls are fast (~1-5ms on CPU for short texts).

        AI Concept: Sentence Transformers
        ====================================
        all-MiniLM-L6-v2 is a distilled version of BERT fine-tuned specifically
        for semantic similarity tasks. 'MiniLM' = a smaller, faster BERT.
        'L6' = 6 transformer layers (BERT-base has 12 → this is 2x faster).

        normalize_embeddings=True: makes every vector unit-length.
        This is required for cosine similarity to work correctly:
          cos_sim(a, b) = dot(a, b) / (|a| × |b|)
        If both are unit-length: |a| = |b| = 1, so cos_sim = dot(a, b).
        This is faster to compute and numerically stable.

        Args:
            text: Input text to embed.

        Returns:
            np.ndarray: Shape (384,), dtype float32, unit-normalized.
        """
        if self._model is None:
            # Import here to avoid slow startup when cache isn't used
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.debug(f"Loaded embedding model: {self._model_name}")

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,  # Required for cosine similarity
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    @staticmethod
    def _make_id(query: str) -> str:
        """
        Create a stable, short cache key from the query text.

        We use SHA256 and take the first 16 hex chars (8 bytes = 2^64 space).
        Collision probability with 1M entries: ~1 in 10^14. Acceptable.
        """
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
