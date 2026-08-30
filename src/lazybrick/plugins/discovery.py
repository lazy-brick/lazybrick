"""Filesystem discovery for external plugin manifests."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from lazybrick.plugins.errors import PluginError, PluginFailure
from lazybrick.plugins.manifest import PluginManifest, load_manifest


def discover_plugins(search_paths: Iterable[str | Path]) -> dict[str, PluginManifest]:
    """Discover plugin-manifest.json files without importing plugin packages."""

    discovered: dict[str, PluginManifest] = {}
    for root_value in search_paths:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        candidates = [root] if root.name == "plugin-manifest.json" else sorted(
            root.glob("*/plugin-manifest.json")
        )
        for candidate in candidates:
            manifest = load_manifest(candidate)
            if manifest.name in discovered:
                raise PluginError(
                    PluginFailure(
                        "duplicate_plugin",
                        f"plugin {manifest.name!r} was discovered more than once",
                        {"manifest": str(candidate)},
                    )
                )
            discovered[manifest.name] = manifest
    return discovered
