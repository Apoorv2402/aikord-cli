"""
Tests: engine/router.py
=========================
Tests for the complexity scorer and provider routing logic.
No real HTTP calls are made — providers are not actually invoked.
"""

import pytest

from src.core.schema import AppConfig, ShellContext
from src.engine.router import Router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_no_cloud():
    """Config with no cloud API key — forces local routing even for complex queries."""
    return AppConfig(
        provider="ollama",
        model="qwen2.5-coder:7b",
        endpoint="http://localhost:11434/v1",
        api_key="ollama",
        cloud_api_key="",  # No cloud key
        routing_threshold=0.4,
    )


@pytest.fixture
def config_with_cloud():
    """Config with a cloud API key — allows cloud routing."""
    return AppConfig(
        provider="ollama",
        model="qwen2.5-coder:7b",
        endpoint="http://localhost:11434/v1",
        api_key="ollama",
        cloud_api_key="sk-test-key",
        cloud_endpoint="https://api.deepseek.com/v1",
        cloud_model="deepseek-chat",
        routing_threshold=0.4,
    )


@pytest.fixture
def git_context():
    return ShellContext(
        os_name="linux", shell="bash", cwd="/project",
        is_git_repo=True, git_branch="main",
    )


@pytest.fixture
def no_git_context():
    return ShellContext(
        os_name="linux", shell="bash", cwd="/tmp",
        is_git_repo=False,
    )


# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------

class TestComplexityScorer:
    def test_simple_query_scores_low(self, config_no_cloud, no_git_context):
        router = Router(config_no_cloud)
        score = router.score_complexity("list files", no_git_context)
        assert score < 0.4, f"Expected score < 0.4, got {score:.3f}"

    def test_long_query_scores_higher(self, config_no_cloud, no_git_context):
        router = Router(config_no_cloud)
        short = router.score_complexity("ls", no_git_context)
        long = router.score_complexity(
            "find all python files modified in the last 7 days and count lines of code", no_git_context
        )
        assert long > short

    def test_pipe_operators_increase_score(self, config_no_cloud, no_git_context):
        router = Router(config_no_cloud)
        no_pipe = router.score_complexity("ls -la", no_git_context)
        with_pipes = router.score_complexity("ls -la | grep .py | wc -l", no_git_context)
        assert with_pipes > no_pipe

    def test_destructive_keyword_always_high(self, config_no_cloud, no_git_context):
        router = Router(config_no_cloud)
        score = router.score_complexity("rm -rf node_modules", no_git_context)
        assert score >= 0.3, "Destructive command must score ≥ 0.3"

    def test_git_context_increases_score(self, config_no_cloud, git_context, no_git_context):
        router = Router(config_no_cloud)
        score_git = router.score_complexity("git rebase main", git_context)
        score_no_git = router.score_complexity("git rebase main", no_git_context)
        assert score_git >= score_no_git

    def test_score_is_clamped_to_1(self, config_no_cloud, git_context):
        router = Router(config_no_cloud)
        # Very long + many pipes + destructive + git = should clamp at 1.0
        score = router.score_complexity(
            "rm -rf | kill | drop | delete | truncate | git reset | shred " * 5,
            git_context,
        )
        assert score <= 1.0

    def test_score_is_non_negative(self, config_no_cloud, no_git_context):
        router = Router(config_no_cloud)
        score = router.score_complexity("", no_git_context)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Provider routing decisions
# ---------------------------------------------------------------------------

class TestProviderRouting:
    def test_simple_query_routes_local(self, config_no_cloud, no_git_context):
        router = Router(config_no_cloud)
        provider = router.resolve("ls -la", no_git_context)
        assert provider.provider_name == "ollama"

    def test_no_cloud_key_forces_local(self, config_no_cloud, no_git_context):
        """Even a high-complexity query must route local if no cloud key is set."""
        router = Router(config_no_cloud)
        provider = router.resolve(
            "rm -rf everything and rebuild the entire system", no_git_context
        )
        assert provider.provider_name == "ollama"

    def test_cloud_key_allows_cloud_routing(self, config_with_cloud, no_git_context):
        router = Router(config_with_cloud)
        # Destructive query should score >= 0.4 and route to cloud
        provider = router.resolve("rm -rf /var/log", no_git_context)
        assert provider.provider_name == "deepseek"

    def test_force_local_override(self, config_with_cloud, no_git_context):
        router = Router(config_with_cloud)
        provider = router.resolve(
            "rm -rf /everything", no_git_context, force_provider="local"
        )
        assert provider.provider_name == "ollama"

    def test_force_cloud_override(self, config_with_cloud, no_git_context):
        router = Router(config_with_cloud)
        provider = router.resolve("ls", no_git_context, force_provider="cloud")
        assert provider.provider_name == "deepseek"

    def test_force_groq_override(self, config_with_cloud, no_git_context):
        router = Router(config_with_cloud)
        provider = router.resolve("ls", no_git_context, force_provider="groq")
        assert provider.provider_name == "groq"

    def test_invalid_provider_raises(self, config_with_cloud, no_git_context):
        router = Router(config_with_cloud)
        with pytest.raises(ValueError, match="Unknown provider"):
            router.resolve("ls", no_git_context, force_provider="anthropic")


# ---------------------------------------------------------------------------
# Routing threshold boundary conditions
# ---------------------------------------------------------------------------

class TestRoutingThreshold:
    def test_exactly_at_threshold_routes_cloud(self):
        """Score exactly at threshold should go to cloud."""
        config = AppConfig(
            cloud_api_key="sk-test",
            cloud_endpoint="https://api.deepseek.com/v1",
            routing_threshold=0.4,
        )
        router = Router(config)
        # Mock score_complexity to return exactly 0.4
        import unittest.mock as mock
        with mock.patch.object(router, "score_complexity", return_value=0.4):
            provider = router.resolve(
                "any query",
                ShellContext(os_name="linux", shell="bash", cwd="/"),
            )
        assert provider.provider_name == "deepseek"

    def test_just_below_threshold_routes_local(self):
        """Score just below threshold should stay local."""
        config = AppConfig(
            cloud_api_key="sk-test",
            routing_threshold=0.4,
        )
        router = Router(config)
        import unittest.mock as mock
        with mock.patch.object(router, "score_complexity", return_value=0.39):
            provider = router.resolve(
                "any query",
                ShellContext(os_name="linux", shell="bash", cwd="/"),
            )
        assert provider.provider_name == "ollama"
