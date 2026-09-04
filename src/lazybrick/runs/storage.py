"""Atomic immutable run bundles and a content-addressed successful-artifact index."""

from __future__ import annotations

from hashlib import sha256
import json
import os
import stat
from pathlib import Path
import tempfile
from typing import Any, Mapping

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


def _snapshot(info: os.stat_result) -> tuple[int, ...]:
    # atime may change because we read the file. Identity and mutation fields
    # must not change between inspection, opening, reading and name recheck.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


_DESCRIPTOR_WALK_SUPPORTED = (
    hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NONBLOCK") and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd and os.stat in os.supports_follow_symlinks
    and os.listdir in os.supports_fd
)


def hash_files(root: str | Path) -> dict[str, str]:
    """Hash descriptor-anchored regular files, rejecting observed replacement.

    This is not a filesystem snapshot: callers must quiesce artifact writers.
    No path-based fallback is safe on platforms lacking descriptor operations.
    """
    if not _DESCRIPTOR_WALK_SUPPORTED:
        raise RunStorageError("safe artifact hashing requires descriptor-relative no-follow operations")
    base = Path(root)
    try:
        root_info = base.lstat()
    except OSError as error:
        raise RunStorageError("artifact root must be a real directory") from error
    if not stat.S_ISDIR(root_info.st_mode):
        raise RunStorageError("artifact root must be a real directory")
    result: dict[str, str] = {}
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)

    def unchanged(before: os.stat_result, after: os.stat_result) -> None:
        if _snapshot(before) != _snapshot(after):
            raise RunStorageError("artifact entry changed while hashing")

    def visit(name: str | Path, parent: int | None, info: os.stat_result,
              relative: str) -> None:
        is_directory = stat.S_ISDIR(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            raise RunStorageError(f"artifact must not contain symlinks: {relative}")
        if not is_directory and not stat.S_ISREG(info.st_mode):
            raise RunStorageError(f"artifact must contain only regular files: {relative}")
        # NOFOLLOW closes the symlink race; NONBLOCK prevents a substituted
        # FIFO from hanging open. Verify the actual descriptor before any read.
        fd = os.open(name, flags | (os.O_DIRECTORY if is_directory else 0), dir_fd=parent)
        try:
            opened = os.fstat(fd)
            unchanged(info, opened)
            if is_directory:
                for child in sorted(os.listdir(fd)):
                    child_info = os.stat(child, dir_fd=fd, follow_symlinks=False)
                    child_relative = f"{relative}/{child}" if relative else child
                    visit(child, fd, child_info, child_relative)
            else:
                if not stat.S_ISREG(opened.st_mode):
                    raise RunStorageError("opened artifact entry is not a regular file")
                hasher = sha256()
                # Bound reads to the inspected size, even if a writer appends
                # continuously. Truncation/growth also fails the final fstat.
                remaining = opened.st_size
                while remaining:
                    block = os.read(fd, min(1024 * 1024, remaining))
                    if not block:
                        raise RunStorageError("artifact entry truncated while hashing")
                    hasher.update(block)
                    remaining -= len(block)
                result[relative] = hasher.hexdigest()
            unchanged(opened, os.fstat(fd))
            unchanged(opened, os.stat(name, dir_fd=parent, follow_symlinks=False))
        finally:
            os.close(fd)

    try:
        visit(base, None, root_info, "")
    except OSError as error:
        raise RunStorageError("artifact entry could not be safely opened or inspected") from error
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
