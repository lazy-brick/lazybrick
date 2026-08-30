"""Software, hardware, command, seed, and redacted environment provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import os
import platform
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


_SECRET_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in sorted(environment.items()):
        upper = key.upper()
        result[key] = "<redacted>" if any(marker in upper for marker in _SECRET_MARKERS) else value
    return result


def _package_versions(packages: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _nvidia_inventory() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "command": command, "gpus": []}
    if completed.returncode != 0:
        return {
            "available": False,
            "command": command,
            "gpus": [],
            "error": completed.stderr.strip(),
        }
    fields = ("name", "uuid", "memory_mib", "compute_capability", "driver")
    gpus = [
        dict(zip(fields, (part.strip() for part in line.split(",")), strict=True))
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    return {"available": True, "command": command, "gpus": gpus}


def collect_provenance(
    *,
    commands: Sequence[Sequence[str]],
    seeds: Mapping[str, int],
    environment: Mapping[str, str] | None = None,
    packages: Iterable[str] = (
        "torch",
        "transformers",
        "llmcompressor",
        "compressed-tensors",
        "vllm",
    ),
) -> dict[str, object]:
    return {
        "recorded_at": utc_now(),
        "software": {
            "python": sys.version,
            "packages": _package_versions(packages),
        },
        "system": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "nvidia": _nvidia_inventory(),
        "commands": [list(command) for command in commands],
        "seeds": dict(sorted(seeds.items())),
        "environment": redact_environment(environment or os.environ),
    }
