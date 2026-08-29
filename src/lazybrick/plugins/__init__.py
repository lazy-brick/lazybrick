"""Public plugin discovery, protocol, and execution boundary."""

from lazybrick.plugins.discovery import discover_plugins
from lazybrick.plugins.errors import PluginError, PluginFailure
from lazybrick.plugins.manifest import (
    PLUGIN_API_VERSION,
    ImplementationRef,
    PluginManifest,
    load_manifest,
)
from lazybrick.plugins.protocol import PluginRequest, PluginResponse
from lazybrick.plugins.runner import PluginInvocation, PluginRunResult, PluginRunner

__all__ = [
    "PLUGIN_API_VERSION",
    "ImplementationRef",
    "PluginError",
    "PluginFailure",
    "PluginInvocation",
    "PluginManifest",
    "PluginRequest",
    "PluginResponse",
    "PluginRunResult",
    "PluginRunner",
    "discover_plugins",
    "load_manifest",
]
