"""
Configuration Loader
=====================
Loads application config from ~/.config/aikord-cli/config.json and applies
environment variable overrides.

Priority (highest to lowest):
  1. Environment variables (AIKORD_*)
  2. ~/.config/aikord-cli/config.json
  3. AppConfig field defaults (in schema.py)

AI Engineer Note:
==================
Always design AI tooling with environment variable overrides. This lets you:
  - Switch providers in CI/CD without touching files  (AIKORD_PROVIDER=groq)
  - Inject secrets safely                             (AIKORD_CLOUD_API_KEY=sk-...)
  - Override models per-session                       (AIKORD_MODEL=llama3.2:3b)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.core.schema import AppConfig

# ---------------------------------------------------------------------------
# Config file location
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".config" / "aikord-cli"
CONFIG_FILE = CONFIG_DIR / "config.json"
TELEMETRY_FILE = CONFIG_DIR / "telemetry.jsonl"
CACHE_DIR = CONFIG_DIR / "cache"


def get_config() -> AppConfig:
    """
    Load AppConfig with full priority chain: env vars > file > defaults.

    Returns:
        AppConfig: Fully resolved configuration object.
    """
    # Step 1: Load from file (if it exists)
    file_data: dict = {}
    if CONFIG_FILE.exists():
        try:
            file_data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            file_data = {}

    # Step 2: Build initial config from file data
    config = AppConfig(**file_data)

    # Step 3: Apply environment variable overrides
    env_overrides = {
        # Local provider
        "provider":       os.getenv("AIKORD_PROVIDER"),
        "model":          os.getenv("AIKORD_MODEL"),
        "endpoint":       os.getenv("AIKORD_ENDPOINT"),
        "api_key":        os.getenv("AIKORD_API_KEY"),
        # Cloud provider
        "cloud_provider": os.getenv("AIKORD_CLOUD_PROVIDER"),
        "cloud_model":    os.getenv("AIKORD_CLOUD_MODEL"),
        "cloud_endpoint": os.getenv("AIKORD_CLOUD_ENDPOINT"),
        "cloud_api_key":  os.getenv("AIKORD_CLOUD_API_KEY"),
        # Groq fallback
        "groq_api_key":   os.getenv("AIKORD_GROQ_API_KEY"),
        "groq_model":     os.getenv("AIKORD_GROQ_MODEL"),
        # Behaviour
        "auto_fix":       _parse_bool(os.getenv("AIKORD_AUTO_FIX")),
        "max_retries":    _parse_int(os.getenv("AIKORD_MAX_RETRIES")),
        "plan":           os.getenv("AIKORD_PLAN"),
    }

    # Only override fields that were actually set in the environment
    non_null = {k: v for k, v in env_overrides.items() if v is not None}
    if non_null:
        # Re-validate with merged data
        merged = {**config.model_dump(), **non_null}
        config = AppConfig(**merged)

    return config


def save_config(config: AppConfig) -> None:
    """Persist the current config to disk as JSON."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def ensure_dirs() -> None:
    """Create all required config directories if they don't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes", "on")


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
