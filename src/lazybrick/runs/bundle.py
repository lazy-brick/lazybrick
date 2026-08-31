"""Bundle-level integrity for a complete attempt, not only its artifact weights.

Artifact file hashes prove that the produced weights are intact. They say nothing
about the recipe, resolved plan, provenance, results, state history, or retained
logs stored beside them, so a successful attempt could keep apparently valid
weight hashes while the surrounding evidence was changed. The bundle manifest
closes that gap by covering *every* regular file in the bundle.

Scope, stated plainly: this detects missing, added, modified, and swapped files.
It is an integrity manifest, not a signature. Anyone who can write to the store
can rewrite a record and rebuild the manifest, so `bundle_digest` is recorded
outside the bundle -- in the artifact index -- and evidence published anywhere
must carry that digest from a source the bundle itself does not control.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from lazybrick.runs.identity import IdentityError, canonical_json

#: Versioned public contract. Bump when the manifest shape or coverage changes.
BUNDLE_MANIFEST_VERSION = "0.1"

MANIFEST_NAME = "bundle-manifest.json"

#: Records every promoted attempt must carry, successful or failed.
COMMON_RECORDS = frozenset({"identity.json", "status.json"})

#: Additional records a SUCCEEDED attempt must carry before it is evidence.
SUCCESS_RECORDS = frozenset(
    {
        "recipe.yaml",
        "resolved_recipe.json",
        "plan.json",
        "artifact.json",
        "provenance.json",
        "results.json",
        "state-history.json",
    }
)

_READ_BLOCK = 1024 * 1024


class BundleIntegrityError(RuntimeError):
    """Raised when a bundle is incomplete, mutable, or internally inconsistent."""


def _require_real_directory(root: Path) -> None:
    if root.is_symlink():
        raise BundleIntegrityError(f"bundle root must not be a symlink: {root}")
    if not root.is_dir():
        raise BundleIntegrityError(f"bundle directory does not exist: {root}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_bundle_files(bundle: Path) -> list[Path]:
    """Every regular file in the bundle, manifest excluded, in stable order.

    Fails closed on symlinks and on anything that is not a regular file. A FIFO,
    socket, or device node must not be silently skipped: skipping would leave it
    outside the manifest and therefore outside verification.
    """

    found: list[Path] = []
    for directory, subdirectories, filenames in os.walk(bundle, followlinks=False):
        current = Path(directory)
        for name in sorted(subdirectories):
            candidate = current / name
            if candidate.is_symlink():
                raise BundleIntegrityError(
                    f"bundle must not contain symlinks: {candidate.relative_to(bundle).as_posix()}"
                )
        for name in sorted(filenames):
            candidate = current / name
            relative = candidate.relative_to(bundle).as_posix()
            if candidate.is_symlink():
                raise BundleIntegrityError(f"bundle must not contain symlinks: {relative}")
            mode = candidate.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise BundleIntegrityError(
                    f"bundle must contain only regular files: {relative}"
                )
            if relative == MANIFEST_NAME:
                continue
            found.append(candidate)
    return sorted(found)


def build_manifest(bundle: str | Path) -> dict[str, Any]:
    """Hash every file in the bundle except the manifest itself."""

    root = Path(bundle)
    _require_real_directory(root)
    files: dict[str, Any] = {}
    for path in _iter_bundle_files(root):
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "sha256": _sha256_file(path),
            "size": path.lstat().st_size,
        }
    return {"manifest_version": BUNDLE_MANIFEST_VERSION, "files": files}


def bundle_digest(manifest: Mapping[str, Any]) -> str:
    """Identity of the manifest itself, for recording outside the bundle."""

    try:
        return sha256(canonical_json(manifest)).hexdigest()
    except IdentityError as error:
        raise BundleIntegrityError(f"bundle manifest is not canonical: {error}") from error


def write_manifest(bundle: str | Path) -> str:
    """Write the manifest as the final record of a staged attempt.

    Returns the bundle digest so the caller can record it outside the bundle.
    """

    root = Path(bundle)
    path = root / MANIFEST_NAME
    if path.exists():
        raise BundleIntegrityError("bundle manifest already exists")
    manifest = build_manifest(root)
    path.write_bytes(canonical_json(manifest) + b"\n")
    return bundle_digest(manifest)


def read_manifest(bundle: str | Path) -> dict[str, Any]:
    root = Path(bundle)
    path = root / MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        raise BundleIntegrityError(f"bundle is missing {MANIFEST_NAME}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise BundleIntegrityError(f"{MANIFEST_NAME} is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise BundleIntegrityError(f"{MANIFEST_NAME} must be a JSON object")
    version = manifest.get("manifest_version")
    if version != BUNDLE_MANIFEST_VERSION:
        raise BundleIntegrityError(
            f"unsupported bundle manifest version: {version!r}; "
            f"this build understands {BUNDLE_MANIFEST_VERSION!r}"
        )
    if not isinstance(manifest.get("files"), dict):
        raise BundleIntegrityError(f"{MANIFEST_NAME} has no files mapping")
    return manifest


def verify_manifest(bundle: str | Path, *, expected_digest: str | None = None) -> dict[str, Any]:
    """Compare the bundle on disk against its manifest.

    A swapped pair of records shows up as two `modified` entries, because the
    manifest is keyed by path: content moving between paths changes both.
    """

    root = Path(bundle)
    manifest = read_manifest(root)
    if expected_digest is not None and bundle_digest(manifest) != expected_digest:
        raise BundleIntegrityError(
            "bundle manifest does not match the expected bundle digest; "
            "the manifest itself was replaced"
        )

    recorded: Mapping[str, Any] = manifest["files"]
    actual = build_manifest(root)["files"]

    missing = sorted(set(recorded) - set(actual))
    added = sorted(set(actual) - set(recorded))
    modified = sorted(
        key
        for key in set(recorded) & set(actual)
        if recorded[key].get("sha256") != actual[key]["sha256"]
        or recorded[key].get("size") != actual[key]["size"]
    )
    if missing or added or modified:
        raise BundleIntegrityError(
            "bundle integrity verification failed; "
            f"missing={missing}, added={added}, modified={modified}"
        )
    return manifest


def _load_record(bundle: Path, name: str) -> Any:
    path = bundle / name
    if path.is_symlink() or not path.is_file():
        raise BundleIntegrityError(f"bundle is missing {name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise BundleIntegrityError(f"{name} is not valid JSON") from error


def verify_links(bundle: str | Path) -> dict[str, Any]:
    """Check that the records inside the bundle agree with each other.

    Hashes prove each record is unchanged. They do not prove the records belong
    together, so a bundle assembled from records of two different attempts would
    hash cleanly. These checks reject that.
    """

    root = Path(bundle)
    status = _load_record(root, "status.json")
    identity = _load_record(root, "identity.json")

    state = status.get("state")
    if state != "SUCCEEDED":
        return {"state": state, "identity": identity}

    plan = _load_record(root, "plan.json")
    artifact = _load_record(root, "artifact.json")
    history = _load_record(root, "state-history.json")

    for key in ("recipe_digest", "plan_digest", "artifact_id"):
        if identity.get(key) != plan.get(key):
            raise BundleIntegrityError(
                f"identity and plan disagree on {key}: "
                f"{identity.get(key)!r} != {plan.get(key)!r}"
            )
    if artifact.get("artifact_id") != identity.get("artifact_id"):
        raise BundleIntegrityError("artifact and identity disagree on artifact_id")
    if history.get("state") != "SUCCEEDED":
        raise BundleIntegrityError("state history is not terminal SUCCEEDED")

    artifact_files = artifact.get("files")
    if not isinstance(artifact_files, dict) or not artifact_files:
        raise BundleIntegrityError("artifact.json records no files")

    # The artifact hashes and the manifest must agree about the same bytes.
    manifest_files = read_manifest(root)["files"]
    for relative, digest in sorted(artifact_files.items()):
        key = f"artifact/{relative}"
        entry = manifest_files.get(key)
        if entry is None:
            raise BundleIntegrityError(f"bundle manifest does not cover {key}")
        if entry.get("sha256") != digest:
            raise BundleIntegrityError(
                f"artifact.json and bundle manifest disagree on {key}"
            )
    return {"state": state, "identity": identity, "artifact": artifact, "plan": plan}


def verify_bundle(
    bundle: str | Path,
    *,
    expected_digest: str | None = None,
    require_success: bool = True,
) -> dict[str, Any]:
    """Full gate: run this before inference on, or publication of, a bundle.

    Verifies file coverage, then record agreement. Pass `expected_digest` from
    the artifact index (or any source outside the bundle) to also detect a
    wholesale manifest replacement.
    """

    root = Path(bundle)
    _require_real_directory(root)

    verify_manifest(root, expected_digest=expected_digest)
    status = _load_record(root, "status.json")
    state = status.get("state")

    required = set(COMMON_RECORDS)
    if state == "SUCCEEDED":
        required |= SUCCESS_RECORDS
    elif require_success:
        raise BundleIntegrityError(f"bundle did not succeed: state={state!r}")
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise BundleIntegrityError(f"bundle is missing records: {missing}")

    return verify_links(root)
