"""Isolated local subprocess execution for LazyBrick plugins."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping

from lazybrick.plugins.errors import PluginError, PluginFailure
from lazybrick.plugins.manifest import PluginManifest
from lazybrick.plugins.protocol import PluginRequest, PluginResponse


_BASE_ENV_KEYS = ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP")


@dataclass(frozen=True, slots=True)
class PluginInvocation:
    manifest: PluginManifest
    command: tuple[str, ...]
    runtime_dependencies: dict[str, str]
    duration_ms: int
    stdout: str
    stderr: str
    returncode: int

    def provenance(self) -> dict[str, object]:
        return {
            "plugin": {
                "name": self.manifest.name,
                "package": self.manifest.package,
                "package_version": self.manifest.package_version,
                "implementation": self.manifest.implementation.to_dict(),
                "plugin_api_version": self.manifest.plugin_api_version,
            },
            "command": list(self.command),
            "runtime_dependencies": self.runtime_dependencies,
            "duration_ms": self.duration_ms,
            "returncode": self.returncode,
        }


@dataclass(frozen=True, slots=True)
class PluginRunResult:
    response: PluginResponse
    invocation: PluginInvocation


def _resolved_dependencies(manifest: PluginManifest) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for package in manifest.runtime_dependencies:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def _bounded_environment(extra: Mapping[str, str] | None) -> dict[str, str]:
    result = {key: os.environ[key] for key in _BASE_ENV_KEYS if key in os.environ}
    result.update({"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"})
    if extra:
        for key, value in extra.items():
            if not isinstance(key, str) or not key or "=" in key or "\x00" in key:
                raise PluginError(
                    PluginFailure("invalid_plugin_environment", f"invalid environment key: {key!r}")
                )
            if not isinstance(value, str) or "\x00" in value:
                raise PluginError(
                    PluginFailure("invalid_plugin_environment", f"invalid value for {key}")
                )
            result[key] = value
    return result


class PluginRunner:
    """Run one validated plugin operation with no shell and a bounded environment."""

    def __init__(self, manifest: PluginManifest, *, timeout_seconds: int = 300) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.manifest = manifest
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        operation: str,
        *,
        input_dir: str | Path,
        output_dir: str | Path,
        payload: dict[str, object] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> PluginRunResult:
        if operation not in self.manifest.operations:
            raise PluginError(
                PluginFailure(
                    "unsupported_plugin_operation",
                    f"plugin {self.manifest.name} does not support {operation}",
                )
            )
        input_path = Path(input_dir).resolve()
        output_path = Path(output_dir).resolve()
        if not input_path.is_dir():
            raise PluginError(
                PluginFailure("plugin_input_missing", f"input directory does not exist: {input_path}")
            )
        output_path.mkdir(parents=True, exist_ok=True)
        request = PluginRequest(
            operation=operation,
            input_dir=str(input_path),
            output_dir=str(output_path),
            payload=dict(payload or {}),
        )
        command = self.manifest.command
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(request.to_dict(), separators=(",", ":")),
                text=True,
                capture_output=True,
                shell=False,
                cwd=str(input_path),
                env=_bounded_environment(environment),
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            duration_ms = round((time.monotonic() - started) * 1000)
            raise PluginError(
                PluginFailure(
                    "plugin_timeout",
                    f"plugin exceeded {self.timeout_seconds}s timeout",
                    {
                        "duration_ms": duration_ms,
                        "stdout": error.stdout or "",
                        "stderr": error.stderr or "",
                    },
                )
            ) from error
        except OSError as error:
            raise PluginError(
                PluginFailure(
                    "plugin_start_failed",
                    "plugin process could not be started",
                    {"error": str(error), "command": list(command)},
                )
            ) from error

        duration_ms = round((time.monotonic() - started) * 1000)
        invocation = PluginInvocation(
            manifest=self.manifest,
            command=command,
            runtime_dependencies=_resolved_dependencies(self.manifest),
            duration_ms=duration_ms,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
        if completed.returncode != 0:
            raise PluginError(
                PluginFailure(
                    "plugin_crashed",
                    f"plugin exited with status {completed.returncode}",
                    {
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "provenance": invocation.provenance(),
                    },
                )
            )
        try:
            raw_response = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise PluginError(
                PluginFailure(
                    "plugin_protocol_error",
                    "plugin stdout is not one JSON response",
                    {
                        "line": error.lineno,
                        "column": error.colno,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    },
                )
            ) from error
        response = PluginResponse.from_dict(raw_response, request)
        if response.error is not None:
            raise PluginError(response.error)
        return PluginRunResult(response=response, invocation=invocation)
