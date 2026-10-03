"""
Plugin Loader
==============
Discovers, imports, and manages PluginBase subclasses from configured directories.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from src.core.schema import AppConfig
from src.plugins.base_plugin import PluginBase

logger = logging.getLogger(__name__)

_DEFAULT_PLUGIN_DIR = Path.home() / ".config" / "aikord-cli" / "plugins"


def load_plugins(config: AppConfig) -> list[PluginBase]:
    """
    Discover and load all plugins from configured directories.

    Plugin discovery order:
      1. ~/.config/aikord-cli/plugins/  (user plugins, always scanned)
      2. config.plugin_dirs             (extra dirs from config.json)

    A plugin file must:
      - Be a .py file
      - Contain exactly one class that subclasses PluginBase
      - Have __manifest__ set at the class level

    Args:
        config: Application config with plugin_dirs list.

    Returns:
        List of instantiated plugin objects, sorted by manifest name.
    """
    dirs_to_scan: list[Path] = [_DEFAULT_PLUGIN_DIR]
    for extra in config.plugin_dirs:
        dirs_to_scan.append(Path(extra).expanduser())

    plugins: list[PluginBase] = []

    for plugin_dir in dirs_to_scan:
        if not plugin_dir.exists():
            continue
        for py_file in sorted(plugin_dir.glob("*.py")):
            plugin = _load_plugin_file(py_file, config)
            if plugin:
                plugins.append(plugin)
                logger.info(
                    f"Loaded plugin: {plugin.__manifest__.name} "
                    f"v{plugin.__manifest__.version}"
                )

    return plugins


def _load_plugin_file(path: Path, config: AppConfig) -> PluginBase | None:
    """Load a single plugin .py file and return an instantiated PluginBase."""
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]

        # Find the PluginBase subclass in the module
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, PluginBase)
                and obj is not PluginBase
                and hasattr(obj, "__manifest__")
            ):
                return obj(config)

    except Exception as e:
        logger.warning(f"Failed to load plugin from {path}: {e}")

    return None
