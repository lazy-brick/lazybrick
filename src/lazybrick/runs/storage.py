"""Atomic immutable run bundles and a content-addressed successful-artifact index."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
import re
import stat
from pathlib import Path
import tempfile
from uuid import uuid4
from typing import Any, Mapping

from lazybrick.runs.bundle import (
    BundleIntegrityError,
    verify_bundle,
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


def _safe_component(value: object, name: str) -> str:
    if (not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None
            or value in {".", ".."}):
        raise RunStorageError(f"{name} must be a safe single path component")
    return value


def _safe_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RunStorageError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _checked_identity(value: object) -> dict[str, str]:
    keys = {"run_id", "attempt_id", "recipe_digest", "plan_digest", "artifact_id"}
    if not isinstance(value, dict) or set(value) != keys:
        raise RunStorageError("run identity has unknown or missing fields")
    for key in ("run_id", "attempt_id"):
        _safe_component(value[key], key)
    for key in ("recipe_digest", "plan_digest", "artifact_id"):
        _safe_digest(value[key], key)
    return value


def _strict_object(data: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RunStorageError("duplicate store record key")
            result[key] = value
        return result
    def constant(value):
        raise RunStorageError("nonfinite store record value")
    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise RunStorageError("store record is not valid JSON") from error
    if not isinstance(value, dict):
        raise RunStorageError("store record must be a JSON object")
    return value


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
        expected_digest = write_manifest(self.staging)
        checked = verify_bundle(self.staging, expected_digest=expected_digest)
        if checked["identity"] != self.identity.to_dict():
            raise RunStorageError("staged bundle identity differs from the attempt")
        # Persist the producer's digest outside the bundle before promotion.
        # Recovery must use this receipt, never a digest supplied by the bundle.
        self.store._record_promotion(self.identity, expected_digest)
        destination = self._promote_attempt()
        # Promotion succeeded, so the bundle is real evidence even if indexing
        # now fails. Report that precisely instead of losing the bundle.
        try:
            self.store.index_attempt(destination)
        except (OSError, ValueError, BundleIntegrityError, RunStorageError) as error:
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

    @contextmanager
    def _directory(self, parts: tuple[str, ...], *, create: bool = False):
        """Pin each store directory; never follow an index/receipt symlink."""
        if not _DESCRIPTOR_WALK_SUPPORTED:
            raise RunStorageError("safe store records require descriptor operations")
        fd = None
        try:
            fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            for part in parts:
                _safe_component(part, "directory")
                if create:
                    try:
                        os.mkdir(part, dir_fd=fd)
                        os.fsync(fd)
                    except FileExistsError:
                        pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd
        except OSError as error:
            raise RunStorageError("store directory could not be safely accessed") from error
        finally:
            if fd is not None:
                os.close(fd)

    @staticmethod
    def _read_at(fd: int, name: str) -> bytes | None:
        _safe_component(name.removesuffix(".json"), "record")
        try:
            record = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        except FileNotFoundError:
            return None
        try:
            before = os.fstat(record)
            if not stat.S_ISREG(before.st_mode):
                raise RunStorageError("store record must be a regular file")
            remaining = before.st_size
            blocks = []
            while remaining:
                block = os.read(record, min(remaining, 1024 * 1024))
                if not block:
                    raise RunStorageError("store record truncated during read")
                blocks.append(block)
                remaining -= len(block)
            if (_snapshot(before) != _snapshot(os.fstat(record))
                    or _snapshot(before) != _snapshot(os.stat(name, dir_fd=fd, follow_symlinks=False))):
                raise RunStorageError("store record changed during read")
            return b"".join(blocks)
        finally:
            os.close(record)

    def _create_record(self, parts: tuple[str, ...], name: str, content: bytes) -> None:
        """Publish durable bytes atomically without replacing an existing anchor."""
        with self._directory(parts, create=True) as fd:
            existing = self._read_at(fd, name)
            if existing is not None:
                if existing != content:
                    raise RunStorageError("existing store record conflicts; refusing to overwrite")
                return
            temporary = f".pending-{uuid4().hex}"
            record = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            try:
                with os.fdopen(record, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
                except FileExistsError:
                    if self._read_at(fd, name) != content:
                        raise RunStorageError("concurrent store record conflicts; refusing to overwrite")
                os.fsync(fd)
            finally:
                os.unlink(temporary, dir_fd=fd)

    def _record_promotion(self, identity: RunIdentity, expected_digest: str) -> None:
        value = _checked_identity(identity.to_dict())
        _safe_digest(expected_digest, "bundle_digest")
        receipt = {"receipt_version": "1", "identity": value, "bundle_digest": expected_digest}
        self._create_record(("anchors", value["run_id"]), f"{value['attempt_id']}.json",
                            canonical_json(receipt) + b"\n")

    def index_attempt(self, bundle: Path) -> Path | None:
        """Verify against a pre-promotion receipt and create a missing index.

        Existing entries are immutable. Missing receipts cannot be regenerated
        from a promoted bundle. Failed attempts are never indexed.
        """
        bundle = Path(bundle)
        try:
            parts = bundle.relative_to(self.root).parts
        except ValueError as error:
            raise RunStorageError("bundle path must stay inside the store") from error
        if len(parts) != 4 or parts[0] != "runs" or parts[2] != "attempts":
            raise RunStorageError("bundle path must name a promoted attempt")
        run_id = _safe_component(parts[1], "run_id")
        attempt_id = _safe_component(parts[3], "attempt_id")
        with self._directory(parts) as fd:
            raw_status = self._read_at(fd, "status.json")
            if raw_status is None:
                raise RunStorageError("bundle status is missing")
            status = _strict_object(raw_status)
            # An unverified failure can only prevent indexing, never grant it.
            if not isinstance(status.get("state"), str):
                raise RunStorageError("bundle status must contain a state string")
            if status["state"] in FAILURE_STATES:
                return None
            if status.get("state") != RunState.SUCCEEDED.value:
                raise RunStorageError("bundle has no terminal success or failure state")
        with self._directory(("anchors", run_id)) as fd:
            raw_receipt = self._read_at(fd, f"{attempt_id}.json")
        if raw_receipt is None:
            raise RunStorageError("trusted promotion receipt is missing; cannot recover from bundle alone")
        receipt = _strict_object(raw_receipt)
        if set(receipt) != {"receipt_version", "identity", "bundle_digest"} or receipt["receipt_version"] != "1":
            raise RunStorageError("invalid promotion receipt")
        identity = _checked_identity(receipt["identity"])
        expected_digest = _safe_digest(receipt["bundle_digest"], "bundle_digest")
        if identity["run_id"] != run_id or identity["attempt_id"] != attempt_id:
            raise RunStorageError("promotion receipt identity differs from bundle location")
        checked = verify_bundle(bundle, expected_digest=expected_digest)
        bundle_identity = _checked_identity(checked["identity"])
        if bundle_identity != identity:
            raise RunStorageError("bundle identity differs from trusted promotion receipt")
        value = {**identity, "bundle": "/".join(parts), "bundle_digest": expected_digest,
                 "files": checked["artifact"]["files"]}
        # Keep the existing index schema: recipe/plan digests live in the receipt.
        del value["recipe_digest"], value["plan_digest"]
        content = canonical_json(value) + b"\n"
        self._create_record(("artifacts", identity["artifact_id"]), f"{attempt_id}.json", content)
        return self.root / "artifacts" / identity["artifact_id"] / f"{attempt_id}.json"

    def reindex(self) -> list[Path]:
        """Repair indexes only with original external receipts; never re-anchor."""
        bundles = []
        # Do not let glob traversal follow substituted store/run directories.
        with self._directory(("runs",)) as runs_fd:
            for run in sorted(os.listdir(runs_fd)):
                _safe_component(run, "run_id")
                with self._directory(("runs", run, "attempts")) as attempts_fd:
                    for attempt in sorted(os.listdir(attempts_fd)):
                        _safe_component(attempt, "attempt_id")
                        bundles.append(self.root / "runs" / run / "attempts" / attempt)
        written = []
        for bundle in bundles:
            entry = self.index_attempt(bundle)
            if entry is not None:
                written.append(entry)
        return written

    def begin(self, identity: RunIdentity) -> AttemptBundle:
        _checked_identity(identity.to_dict())
        staging = self.root / ".staging" / identity.attempt_id
        try:
            staging.mkdir()
        except FileExistsError as error:
            raise RunStorageError(f"attempt staging already exists: {identity.attempt_id}") from error
        return AttemptBundle(self, identity, staging)
