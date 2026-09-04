"""Atomic immutable run bundles and a content-addressed successful-artifact index."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from lazybrick.runs.bundle import (
    BundleIntegrityError,
    bundle_digest,
    read_manifest,
    write_manifest,
)
from lazybrick.runs.identity import RunIdentity, canonical_json
from lazybrick.runs.state import FAILURE_STATES, RunState


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


def _record_json(value: Mapping[str, Any]) -> bytes:
    """Serialize run records; evidence may contain finite measured floats."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RunStorageError("run record is not finite JSON") from error


def hash_files(root: str | Path) -> dict[str, str]:
    base = Path(root)
    result: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise RunStorageError(f"artifact must not contain symlinks: {path.relative_to(base)}")
        if not path.is_file():
            continue
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
        path = self._record_path(relative_path)
        if path.exists():
            raise RunStorageError(f"run record already exists: {relative_path}")
        _atomic_bytes(path, _record_json(value) + b"\n")
        return path

    def write_text(self, relative_path: str, value: str) -> Path:
        path = self._record_path(relative_path)
        if path.exists():
            raise RunStorageError(f"run record already exists: {relative_path}")
        _atomic_bytes(path, value.encode("utf-8"))
        return path

    def _record_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if (
            not relative_path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate == Path(".")
        ):
            raise RunStorageError("run record path must stay inside the attempt bundle")
        path = self.staging.joinpath(candidate)
        try:
            path.relative_to(self.staging)
        except ValueError as error:
            raise RunStorageError("run record path must stay inside the attempt bundle") from error
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
        # Last write before promotion: the manifest must cover every other record.
        write_manifest(self.staging)
        destination = self._promote_attempt()
        # Promotion succeeded, so the bundle is real evidence even if indexing
        # now fails. Report that precisely instead of losing the bundle.
        try:
            self.store.index_attempt(destination)
        except (OSError, ValueError, BundleIntegrityError) as error:
            raise RunStorageError(
                f"attempt {self.identity.attempt_id} was promoted but could not be "
                f"indexed: {error}; re-run RunStore.reindex() to repair the index"
            ) from error
        return destination

    def finalize_failure(self, state: str, failure: Mapping[str, Any]) -> Path:
        try:
            run_state = RunState(state)
        except ValueError as error:
            raise RunStorageError(f"unknown failure state: {state}") from error
        if run_state not in FAILURE_STATES:
            raise RunStorageError(f"state is not a terminal failure: {state}")
        code = failure.get("code")
        message = failure.get("message")
        if not isinstance(code, str) or not code or not isinstance(message, str) or not message:
            raise RunStorageError("failure requires non-empty code and message")
        self.write_json(
            "status.json", {"state": run_state.value, "failure": dict(failure)}
        )
        # Failed attempts are evidence too: a doctored failure must not be
        # promotable into an apparently successful record.
        write_manifest(self.staging)
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

    def index_attempt(self, bundle: Path) -> Path | None:
        """Write the artifact index entry for a promoted successful bundle.

        Idempotent: the entry is derived entirely from the bundle on disk, so
        repeating it after a partial failure rewrites the same bytes. Returns
        ``None`` for a bundle that did not succeed, which is not indexed.
        """

        status = json.loads((bundle / "status.json").read_text(encoding="utf-8"))
        if status.get("state") != RunState.SUCCEEDED.value:
            return None
        identity = json.loads((bundle / "identity.json").read_text(encoding="utf-8"))
        artifact = json.loads((bundle / "artifact.json").read_text(encoding="utf-8"))
        entry = (
            self.root
            / "artifacts"
            / identity["artifact_id"]
            / f"{identity['attempt_id']}.json"
        )
        _atomic_bytes(
            entry,
            canonical_json(
                {
                    "artifact_id": identity["artifact_id"],
                    "run_id": identity["run_id"],
                    "attempt_id": identity["attempt_id"],
                    "bundle": bundle.relative_to(self.root).as_posix(),
                    "bundle_digest": bundle_digest(read_manifest(bundle)),
                    "files": artifact["files"],
                }
            )
            + b"\n",
        )
        return entry

    def reindex(self) -> list[Path]:
        """Rebuild artifact index entries for every promoted successful bundle.

        Repairs the window where an attempt was promoted but its index write
        failed. Safe to run at any time.
        """

        written: list[Path] = []
        for bundle in sorted((self.root / "runs").glob("*/attempts/*")):
            if not bundle.is_dir() or bundle.is_symlink():
                continue
            entry = self.index_attempt(bundle)
            if entry is not None:
                written.append(entry)
        return written

    def begin(self, identity: RunIdentity) -> AttemptBundle:
        for name, value in (
            ("run_id", identity.run_id),
            ("attempt_id", identity.attempt_id),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 128
                or Path(value).name != value
                or value in {".", ".."}
            ):
                raise RunStorageError(f"{name} must be a safe single path component")
        staging = self.root / ".staging" / identity.attempt_id
        try:
            staging.mkdir()
        except FileExistsError as error:
            raise RunStorageError(f"attempt staging already exists: {identity.attempt_id}") from error
        return AttemptBundle(self, identity, staging)
