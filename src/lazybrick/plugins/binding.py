"""Bind planner declarations to an exact subprocess manifest and invocation.

This is metadata agreement, not verification that installed code matches a Git
commit. Package/environment provenance and output evidence remain necessary.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import sys

from lazybrick.canonical import digest
from lazybrick.records import StageSpec, PluginManifest as CapabilityManifest
from lazybrick.plugins.manifest import PluginManifest
from lazybrick.plugins.errors import PluginError, PluginFailure


def _reject(message: str) -> None:
    raise PluginError(PluginFailure("plugin_binding_mismatch", message))


@dataclass(frozen=True)
class ExecutionBinding:
    stage_digest: str
    capability_digest: str
    transport_digest: str
    command: tuple[str, ...]
    settings_digest: str

    @classmethod
    def create(cls, stage: StageSpec, capability: CapabilityManifest,
               transport: PluginManifest) -> ExecutionBinding:
        if getattr(stage, "semantics", None) is not None:
            _reject("numerical semantics requires a tested adapter mapping")
        upstream = transport.upstream
        if transport.manifest_version != "0.2" or upstream is None:
            _reject("execution binding requires a v0.2 transport manifest")
        plugin_name = stage.plugin.rsplit("/", 1)[-1]
        accepted_names = {stage.plugin, plugin_name}
        if capability.name not in accepted_names or transport.name not in accepted_names:
            _reject("stage, capability, and transport plugin names differ")
        if capability.kind != "transformation":
            _reject("stage requires a transformation capability")
        if stage.plugin_version != capability.version or capability.version != upstream["version"]:
            _reject("stage and upstream implementation versions differ")
        if not stage.implementation.is_pinned or stage.implementation.to_json() != capability.implementation.to_json():
            _reject("stage and capability implementation pins differ")
        if capability.implementation.to_json() != upstream["implementation"]:
            _reject("transport upstream implementation differs from planned implementation")
        if capability.plugin_api != transport.plugin_api_version:
            _reject("capability and transport protocol versions differ")
        try:
            installed = version(transport.package)
        except Exception as error:
            _reject(f"cannot identify installed adapter package: {type(error).__name__}")
        if installed != transport.package_version:
            _reject("installed adapter package version differs from transport manifest")
        # v0.2 supports Python modules only. Resolve the interpreter once and
        # bind it, rather than silently replacing the manifest in the launcher.
        command = transport.command
        if len(command) != 3 or command[:2] != ("python", "-m"):
            _reject("v0.2 transport requires an explicit python -m module command")
        effective = (sys.executable, *command[1:])
        return cls(digest(stage.to_json()), digest(capability.to_json()),
                   digest(transport.to_dict()), effective, digest(dict(stage.parameters)))

    def validate_transport(self, transport: PluginManifest) -> None:
        if digest(transport.to_dict()) != self.transport_digest:
            _reject("transport changed after binding")
        if self.command != (sys.executable, *transport.command[1:]):
            _reject("effective interpreter or command changed after binding")

    def validate_payload(self, transport: PluginManifest, payload: dict[str, object]) -> None:
        if "semantics" in payload:
            _reject("numerical semantics requires a tested adapter mapping")
        if digest(payload.get("settings")) != self.settings_digest:
            _reject("execution settings differ from the planned stage")
        try:
            installed = version(transport.upstream["package"])
        except Exception as error:
            _reject(f"cannot identify upstream package: {type(error).__name__}")
        if installed != transport.upstream["version"]:
            _reject("installed upstream package version differs from the planned version")

    def to_dict(self) -> dict[str, object]:
        return {"binding_version": "0.1", "stage_digest": self.stage_digest,
                "capability_digest": self.capability_digest,
                "transport_digest": self.transport_digest,
                "settings_digest": self.settings_digest,
                "effective_command": list(self.command)}
