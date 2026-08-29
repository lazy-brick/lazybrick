"""Atomic immutable run bundles and a content-addressed successful-artifact index."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from lazybrick.runs.identity import RunIdentity, canonical_json


class RunStorageError(RuntimeError):
    """Raised when a run bundle would be incomplete, mutable, or inconsistent."""


_REQUIRED_RECORDS = frozenset(
    {
        "recipe.yaml",
        "resolved_recipe.json",
        "plan.json",
        "artifact.json",
        "provenance.json",
        "results.json",
    }
)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def hash_files(root: str | Path) -> dict[str, str]:
    base = Path(root)
    result: dict[str, str] = {}
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[path.relative_to(base).as_posix()] = digest.hexdigest()
    return result


def verify_hashes(root: str | Path, expected: Mapping[str, str]) -> None:
    actual = hash_files(root)
    if actual != dict(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(
            key for key in set(expected) & set(actual) if expected[key] != actual[key]
        )
        raise RunStorageError(
            f"artifact hash verification failed; missing={missing}, unexpected={unexpected}, changed={changed}"
        )


class AttemptBundle:
    def __init__(self, store: RunStore, identity: RunIdentity, staging: Path) -> None:
        self.store = store
        self.identity = identity
        self.staging = staging
        self.logs_dir = staging / "logs"
        self.artifact_dir = staging / "artifact"
        self.logs_dir.mkdir()
        self.artifact_dir.mkdir()
        self.write_json("identity.json", identity.to_dict())

    def write_json(self, relative_path: str, value: Mapping[str, Any]) -> Path:
        path = self.staging / relative_path
        if path.exists():
            raise RunStorageError(f"run record already exists: {relative_path}")
        _atomic_bytes(path, canonical_json(value) + b"\n")
        return path

    def write_text(self, relative_path: str, value: str) -> Path:
        path = self.staging / relative_path
        if path.exists():
            raise RunStorageError(f"run record already exists: {relative_path}")
        _atomic_bytes(path, value.encode("utf-8"))
        return path

    def write_log(self, name: str, value: str) -> Path:
        if Path(name).name != name:
            raise RunStorageError("log name must not contain a path")
        return self.write_text(f"logs/{name}", value)

    def finalize_success(self, artifact_metadata: Mapping[str, Any]) -> Path:
        if not any(self.artifact_dir.iterdir()):
            raise RunStorageError("cannot succeed without artifact files")
        hashes = hash_files(self.artifact_dir)
        verify_hashes(self.artifact_dir, hashes)
        complete_artifact = {
            **dict(artifact_metadata),
            "artifact_id": self.identity.artifact_id,
            "files": hashes,
        }
        self.write_json("artifact.json", complete_artifact)
        self._require_records()
        self.write_json("status.json", {"state": "SUCCEEDED"})
        destination = self._promote_attempt()
        artifact_index = (
            self.store.root
            / "artifacts"
            / self.identity.artifact_id
            / f"{self.identity.attempt_id}.json"
        )
        _atomic_bytes(
            artifact_index,
            canonical_json(
                {
                    "artifact_id": self.identity.artifact_id,
                    "run_id": self.identity.run_id,
                    "attempt_id": self.identity.attempt_id,
                    "bundle": str(destination.relative_to(self.store.root)),
                    "files": hashes,
                }
            )
            + b"\n",
        )
        return destination

    def finalize_failure(self, state: str, failure: Mapping[str, Any]) -> Path:
        if state == "SUCCEEDED":
            raise RunStorageError("failure state cannot be SUCCEEDED")
        self.write_json("status.json", {"state": state, "failure": dict(failure)})
        return self._promote_attempt()

    def _require_records(self) -> None:
        missing = sorted(name for name in _REQUIRED_RECORDS if not (self.staging / name).is_file())
        if missing:
            raise RunStorageError(f"successful run is missing records: {missing}")

    def _promote_attempt(self) -> Path:
        destination = (
            self.store.root
            / "runs"
            / self.identity.run_id
            / "attempts"
            / self.identity.attempt_id
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RunStorageError(f"attempt already exists: {self.identity.attempt_id}")
        os.replace(self.staging, destination)
        return destination


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        (self.root / ".staging").mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)

    def begin(self, identity: RunIdentity) -> AttemptBundle:
        staging = self.root / ".staging" / identity.attempt_id
        try:
            staging.mkdir()
        except FileExistsError as error:
            raise RunStorageError(f"attempt staging already exists: {identity.attempt_id}") from error
        return AttemptBundle(self, identity, staging)
