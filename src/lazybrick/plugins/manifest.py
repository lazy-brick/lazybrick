"""Plugin manifest parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from lazybrick.plugins.errors import PluginError, PluginFailure


PLUGIN_API_VERSION = "0.1"
PLUGIN_MANIFEST_VERSION = "0.1"
SUPPORTED_OPERATIONS = frozenset({"inspect", "plan", "execute", "validate"})
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


def _fail(code: str, message: str, **details: Any) -> PluginError:
    return PluginError(PluginFailure(code, message, details))


@dataclass(frozen=True, slots=True)
class ImplementationRef:
    source: str
    commit: str | None = None
    container_digest: str | None = None

    @classmethod
    def from_dict(cls, value: object) -> ImplementationRef:
        if not isinstance(value, dict):
            raise _fail("invalid_plugin_manifest", "implementation must be an object")
        unknown = set(value) - {"source", "commit", "container_digest"}
        if unknown:
            raise _fail(
                "invalid_plugin_manifest",
                "implementation contains unknown fields",
                fields=sorted(unknown),
            )
        source = value.get("source")
        commit = value.get("commit")
        digest = value.get("container_digest")
        if not isinstance(source, str) or not source.strip():
            raise _fail("invalid_plugin_manifest", "implementation.source is required")
        if commit is None and digest is None:
            raise _fail(
                "mutable_plugin_implementation",
                "implementation requires an immutable commit or container digest",
            )
        if commit is not None and (
            not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None
        ):
            raise _fail(
                "mutable_plugin_implementation",
                "implementation.commit must be a full lowercase 40-character Git commit",
            )
        if digest is not None and (
            not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None
        ):
            raise _fail(
                "mutable_plugin_implementation",
                "implementation.container_digest must be a sha256 digest",
            )
        return cls(source=source, commit=commit, container_digest=digest)

    def to_dict(self) -> dict[str, str]:
        result = {"source": self.source}
        if self.commit is not None:
            result["commit"] = self.commit
        if self.container_digest is not None:
            result["container_digest"] = self.container_digest
        return result


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Immutable description of a subprocess plugin."""

    name: str
    package: str
    package_version: str
    implementation: ImplementationRef
    command: tuple[str, ...]
    operations: tuple[str, ...]
    runtime_dependencies: tuple[str, ...]
    manifest_version: str = PLUGIN_MANIFEST_VERSION
    plugin_api_version: str = PLUGIN_API_VERSION

    @classmethod
    def from_dict(cls, value: object) -> PluginManifest:
        if not isinstance(value, dict):
            raise _fail("invalid_plugin_manifest", "manifest root must be an object")
        allowed = {
            "manifest_version",
            "plugin_api_version",
            "name",
            "package",
            "package_version",
            "implementation",
            "command",
            "operations",
            "runtime_dependencies",
        }
        unknown = set(value) - allowed
        if unknown:
            raise _fail(
                "invalid_plugin_manifest",
                "manifest contains unknown fields",
                fields=sorted(unknown),
            )
        required_strings = ("name", "package", "package_version")
        for field in required_strings:
            if not isinstance(value.get(field), str) or not value[field].strip():
                raise _fail("invalid_plugin_manifest", f"{field} must be a non-empty string")

        manifest_version = value.get("manifest_version")
        if manifest_version != PLUGIN_MANIFEST_VERSION:
            raise _fail(
                "incompatible_manifest_version",
                f"manifest version {manifest_version!r} is unsupported",
                supported=PLUGIN_MANIFEST_VERSION,
            )
        api_version = value.get("plugin_api_version")
        if api_version != PLUGIN_API_VERSION:
            raise _fail(
                "incompatible_plugin_api",
                f"plugin API version {api_version!r} is incompatible",
                supported=PLUGIN_API_VERSION,
            )

        command = value.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise _fail("invalid_plugin_manifest", "command must be a non-empty string list")
        operations = value.get("operations")
        if (
            not isinstance(operations, list)
            or not operations
            or any(not isinstance(item, str) for item in operations)
        ):
            raise _fail("invalid_plugin_manifest", "operations must be a non-empty string list")
        unsupported = set(operations) - SUPPORTED_OPERATIONS
        if unsupported:
            raise _fail(
                "invalid_plugin_manifest",
                "manifest declares unsupported operations",
                operations=sorted(unsupported),
            )
        dependencies = value.get("runtime_dependencies", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) or not item.strip() for item in dependencies
        ):
            raise _fail(
                "invalid_plugin_manifest",
                "runtime_dependencies must be a string list",
            )
        return cls(
            name=value["name"],
            package=value["package"],
            package_version=value["package_version"],
            implementation=ImplementationRef.from_dict(value.get("implementation")),
            command=tuple(command),
            operations=tuple(operations),
            runtime_dependencies=tuple(dependencies),
            manifest_version=manifest_version,
            plugin_api_version=api_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "plugin_api_version": self.plugin_api_version,
            "name": self.name,
            "package": self.package,
            "package_version": self.package_version,
            "implementation": self.implementation.to_dict(),
            "command": list(self.command),
            "operations": list(self.operations),
            "runtime_dependencies": list(self.runtime_dependencies),
        }


def load_manifest(path: str | Path) -> PluginManifest:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise _fail(
            "plugin_manifest_unreadable",
            f"cannot read plugin manifest: {manifest_path}",
            error=str(error),
        ) from error
    except json.JSONDecodeError as error:
        raise _fail(
            "invalid_plugin_manifest",
            f"plugin manifest is not valid JSON: {manifest_path}",
            line=error.lineno,
            column=error.colno,
        ) from error
    return PluginManifest.from_dict(value)
