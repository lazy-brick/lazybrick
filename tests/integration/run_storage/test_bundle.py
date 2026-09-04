"""A bundle that hashes clean must actually be the evidence it claims to be.

The defect these cover: artifact weights were hashed, but the records beside
them were not, so results.json could be rewritten while the artifact still
verified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lazybrick.runs import (
    BUNDLE_MANIFEST_VERSION,
    MANIFEST_NAME,
    BundleIntegrityError,
    RunIdentity,
    RunStore,
    build_manifest,
    bundle_digest,
    read_manifest,
    verify_bundle,
    verify_manifest,
    write_manifest,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _identity() -> RunIdentity:
    return RunIdentity.create(
        recipe_digest=DIGEST_A, plan_digest=DIGEST_B, artifact_id=DIGEST_C
    )


def _succeeded_bundle(root: Path) -> tuple[RunStore, Path]:
    """A minimal but complete successful attempt, promoted and indexed."""

    store = RunStore(root)
    identity = _identity()
    attempt = store.begin(identity)
    attempt.write_text("recipe.yaml", "schema_version: '0.1'\n")
    attempt.write_json("resolved_recipe.json", {"schema_version": "0.1"})
    attempt.write_json(
        "plan.json",
        {
            "recipe_digest": identity.recipe_digest,
            "plan_digest": identity.plan_digest,
            "artifact_id": identity.artifact_id,
        },
    )
    attempt.write_json("provenance.json", {"tool": "test"})
    attempt.write_json("results.json", {"quality": {"baseline": 1, "quantized": 1}})
    attempt.write_json("state-history.json", {"state": "SUCCEEDED"})
    attempt.write_log("run.log", "started\nfinished\n")
    (attempt.artifact_dir / "model.safetensors").write_bytes(b"weights")
    bundle = attempt.finalize_success({"format": "compressed-tensors"})
    return store, bundle


def test_manifest_covers_records_logs_and_artifact(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")
    covered = set(read_manifest(bundle)["files"])

    # The point of the issue: records and logs are covered, not only weights.
    assert "results.json" in covered
    assert "provenance.json" in covered
    assert "plan.json" in covered
    assert "state-history.json" in covered
    assert "logs/run.log" in covered
    assert "artifact/model.safetensors" in covered
    assert MANIFEST_NAME not in covered


def test_complete_bundle_verifies(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")

    assert verify_bundle(bundle)["state"] == "SUCCEEDED"


def test_modified_record_fails(tmp_path: Path) -> None:
    """The exact regression: valid weights, rewritten quality numbers."""

    _, bundle = _succeeded_bundle(tmp_path / "store")
    (bundle / "results.json").write_text(
        json.dumps({"quality": {"baseline": 1, "quantized": 99}}), encoding="utf-8"
    )

    with pytest.raises(BundleIntegrityError, match="modified=..results.json"):
        verify_bundle(bundle)


def test_modified_log_fails(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")
    (bundle / "logs" / "run.log").write_text("nothing to see\n", encoding="utf-8")

    with pytest.raises(BundleIntegrityError, match="modified=..logs/run.log"):
        verify_bundle(bundle)


def test_missing_record_fails(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")
    (bundle / "provenance.json").unlink()

    with pytest.raises(BundleIntegrityError, match="missing=..provenance.json"):
        verify_bundle(bundle)


def test_added_file_fails(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")
    (bundle / "logs" / "extra.log").write_text("smuggled\n", encoding="utf-8")

    with pytest.raises(BundleIntegrityError, match="added=..logs/extra.log"):
        verify_bundle(bundle)


def test_swapped_records_fail(tmp_path: Path) -> None:
    """Swapping two records preserves the multiset of hashes, not the mapping."""

    _, bundle = _succeeded_bundle(tmp_path / "store")
    provenance = (bundle / "provenance.json").read_bytes()
    results = (bundle / "results.json").read_bytes()
    (bundle / "provenance.json").write_bytes(results)
    (bundle / "results.json").write_bytes(provenance)

    with pytest.raises(BundleIntegrityError) as error:
        verify_bundle(bundle)
    assert "provenance.json" in str(error.value)
    assert "results.json" in str(error.value)


def test_replaced_manifest_is_caught_by_external_digest(tmp_path: Path) -> None:
    """Rebuilding the manifest hides the edit only until the index is consulted."""

    store, bundle = _succeeded_bundle(tmp_path / "store")
    entry = next((store.root / "artifacts").glob("*/*.json"))
    expected = json.loads(entry.read_text(encoding="utf-8"))["bundle_digest"]

    (bundle / "results.json").write_text(
        json.dumps({"quality": {"baseline": 1, "quantized": 99}}), encoding="utf-8"
    )
    (bundle / MANIFEST_NAME).unlink()
    write_manifest(bundle)

    # Self-consistent now, so the manifest alone no longer detects it.
    verify_manifest(bundle)
    with pytest.raises(BundleIntegrityError, match="manifest itself was replaced"):
        verify_bundle(bundle, expected_digest=expected)


def test_symlinked_record_is_rejected(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (bundle / "results.json").unlink()
    (bundle / "results.json").symlink_to(outside)

    with pytest.raises(BundleIntegrityError, match="must not contain symlinks"):
        verify_bundle(bundle)


def test_symlinked_directory_is_rejected(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "planted.json").write_text("{}", encoding="utf-8")
    (bundle / "extra").symlink_to(outside, target_is_directory=True)

    with pytest.raises(BundleIntegrityError, match="must not contain symlinks"):
        verify_bundle(bundle)


def test_non_regular_file_is_rejected_not_skipped(tmp_path: Path) -> None:
    """A FIFO must fail closed; silently skipping it would evade the manifest."""

    _, bundle = _succeeded_bundle(tmp_path / "store")
    os.mkfifo(bundle / "logs" / "pipe")

    with pytest.raises(BundleIntegrityError, match="only regular files"):
        verify_bundle(bundle)


def test_unknown_manifest_version_is_rejected(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")
    manifest = read_manifest(bundle)
    manifest["manifest_version"] = "9.9"
    (bundle / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleIntegrityError, match="unsupported bundle manifest version"):
        verify_bundle(bundle)


def test_manifest_version_is_recorded(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")

    assert read_manifest(bundle)["manifest_version"] == BUNDLE_MANIFEST_VERSION


def test_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")
    first = bundle_digest(build_manifest(bundle))

    assert first == bundle_digest(build_manifest(bundle))

    (bundle / "results.json").write_text("{}", encoding="utf-8")
    assert bundle_digest(build_manifest(bundle)) != first


def test_mismatched_identity_and_plan_fail(tmp_path: Path) -> None:
    """Records from two attempts hash cleanly; the link check must reject them."""

    _, bundle = _succeeded_bundle(tmp_path / "store")
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    plan["plan_digest"] = "d" * 64
    (bundle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (bundle / MANIFEST_NAME).unlink()
    write_manifest(bundle)

    with pytest.raises(BundleIntegrityError, match="disagree on plan_digest"):
        verify_bundle(bundle)


def test_artifact_hashes_must_match_the_manifest(tmp_path: Path) -> None:
    _, bundle = _succeeded_bundle(tmp_path / "store")
    artifact = json.loads((bundle / "artifact.json").read_text(encoding="utf-8"))
    artifact["files"]["model.safetensors"] = "0" * 64
    (bundle / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    (bundle / MANIFEST_NAME).unlink()
    write_manifest(bundle)

    with pytest.raises(BundleIntegrityError, match="disagree on artifact/model"):
        verify_bundle(bundle)


def test_failed_attempt_is_also_covered(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    attempt = store.begin(_identity())
    attempt.write_log("run.log", "boom\n")
    bundle = attempt.finalize_failure(
        "EXECUTION_FAILED", {"code": "plugin_crashed", "message": "boom"}
    )

    assert "logs/run.log" in read_manifest(bundle)["files"]
    with pytest.raises(BundleIntegrityError, match="did not succeed"):
        verify_bundle(bundle)
    assert verify_bundle(bundle, require_success=False)["state"] == "EXECUTION_FAILED"


def test_failed_attempt_is_not_indexed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    attempt = store.begin(_identity())
    attempt.write_log("run.log", "boom\n")
    attempt.finalize_failure(
        "EXECUTION_FAILED", {"code": "plugin_crashed", "message": "boom"}
    )

    assert list((store.root / "artifacts").glob("*/*.json")) == []


def test_index_records_the_bundle_digest(tmp_path: Path) -> None:
    store, bundle = _succeeded_bundle(tmp_path / "store")
    entry = json.loads(
        next((store.root / "artifacts").glob("*/*.json")).read_text(encoding="utf-8")
    )

    assert entry["bundle_digest"] == bundle_digest(read_manifest(bundle))
    assert entry["bundle"] == bundle.relative_to(store.root).as_posix()


def test_reindex_repairs_a_lost_index_entry(tmp_path: Path) -> None:
    """Promotion succeeded but indexing did not: the bundle must be recoverable."""

    store, bundle = _succeeded_bundle(tmp_path / "store")
    entry = next((store.root / "artifacts").glob("*/*.json"))
    original = entry.read_bytes()
    entry.unlink()

    written = store.reindex()

    assert written == [entry]
    assert entry.read_bytes() == original


def test_reindex_is_idempotent(tmp_path: Path) -> None:
    store, _ = _succeeded_bundle(tmp_path / "store")
    entry = next((store.root / "artifacts").glob("*/*.json"))
    before = entry.read_bytes()

    store.reindex()
    store.reindex()

    assert entry.read_bytes() == before
    assert len(list((store.root / "artifacts").glob("*/*.json"))) == 1


def test_symlinked_bundle_root_is_rejected(tmp_path: Path) -> None:
    """A symlinked attempt directory would let one bundle impersonate another."""

    _, bundle = _succeeded_bundle(tmp_path / "store")
    alias = tmp_path / "alias"
    alias.symlink_to(bundle, target_is_directory=True)

    with pytest.raises(BundleIntegrityError, match="root must not be a symlink"):
        verify_bundle(alias)


def test_non_canonical_manifest_is_an_integrity_error(tmp_path: Path) -> None:
    """A float size is not a canonical identity input; fail as an integrity problem."""

    _, bundle = _succeeded_bundle(tmp_path / "store")
    manifest = read_manifest(bundle)
    manifest["files"]["results.json"]["size"] = 1.5
    (bundle / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleIntegrityError, match="not canonical"):
        verify_bundle(bundle, expected_digest="0" * 64)


def _rewrite_manifest(bundle):
    (bundle / MANIFEST_NAME).unlink()
    write_manifest(bundle)


@pytest.mark.parametrize("delete_index", [False, True])
def test_reindex_cannot_bless_rewritten_bundle(tmp_path, delete_index):
    store, bundle = _succeeded_bundle(tmp_path / "store")
    entry = next((store.root / "artifacts").glob("*/*.json"))
    original = entry.read_bytes()
    receipt = next((store.root / "anchors").glob("*/*.json"))
    anchor = receipt.read_bytes()
    if delete_index:
        entry.unlink()
    (bundle / "results.json").write_text('{"quality":"doctored"}')
    _rewrite_manifest(bundle)
    with pytest.raises(BundleIntegrityError, match="manifest itself was replaced"):
        store.reindex()
    assert receipt.read_bytes() == anchor
    if delete_index:
        assert not entry.exists()
    else:
        assert entry.read_bytes() == original


def test_missing_receipt_cannot_be_rebuilt_from_bundle(tmp_path):
    from lazybrick.runs import RunStorageError
    store, bundle = _succeeded_bundle(tmp_path / "store")
    entry = next((store.root / "artifacts").glob("*/*.json"))
    entry.unlink()
    next((store.root / "anchors").glob("*/*.json")).unlink()
    with pytest.raises(RunStorageError, match="trusted promotion receipt"):
        store.reindex()
    assert not entry.exists()


def test_reindex_preserves_conflicting_existing_index(tmp_path):
    from lazybrick.runs import RunStorageError
    store, bundle = _succeeded_bundle(tmp_path / "store")
    entry = next((store.root / "artifacts").glob("*/*.json"))
    changed = json.loads(entry.read_bytes())
    changed["bundle_digest"] = "f" * 64
    entry.write_text(json.dumps(changed))
    before = entry.read_bytes()
    with pytest.raises(RunStorageError, match="refusing to overwrite"):
        store.index_attempt(bundle)
    assert entry.read_bytes() == before


@pytest.mark.parametrize("field", ["artifact_id", "run_id", "attempt_id"])
@pytest.mark.parametrize("bad", ["../../escape", "/absolute", "..\\escape", ".", "", "bad\x00value", "bad:value"])
def test_bundle_controlled_identity_never_selects_index_path(tmp_path, field, bad):
    from lazybrick.runs import RunStorageError
    store, bundle = _succeeded_bundle(tmp_path / "store")
    entry = next((store.root / "artifacts").glob("*/*.json"))
    before = entry.read_bytes()
    identity = json.loads((bundle / "identity.json").read_bytes())
    identity[field] = bad
    (bundle / "identity.json").write_text(json.dumps(identity))
    _rewrite_manifest(bundle)
    with pytest.raises((RunStorageError, BundleIntegrityError)):
        store.reindex()
    assert entry.read_bytes() == before
    assert list((store.root / "artifacts").glob("*/*.json")) == [entry]
    assert not (tmp_path / "escape.json").exists()


@pytest.mark.parametrize("field", ["artifact_id", "run_id", "attempt_id"])
def test_invalid_receipt_identity_is_rejected_before_directory_creation(tmp_path, monkeypatch, field):
    from lazybrick.runs import RunStorageError
    store, bundle = _succeeded_bundle(tmp_path / "store")
    receipt = next((store.root / "anchors").glob("*/*.json"))
    value = json.loads(receipt.read_bytes())
    value["identity"][field] = "../../escape"
    receipt.write_text(json.dumps(value))
    def no_write(*args, **kwargs):
        pytest.fail("invalid identity reached index creation")
    monkeypatch.setattr(store, "_create_record", no_write)
    with pytest.raises(RunStorageError):
        store.index_attempt(bundle)


@pytest.mark.parametrize("location", ["artifacts", "artifact_directory", "entry", "anchors", "run_directory", "bundle"])
def test_reindex_rejects_symlinked_store_paths(tmp_path, location):
    from lazybrick.runs import RunStorageError
    store, bundle = _succeeded_bundle(tmp_path / "store")
    entry = next((store.root / "artifacts").glob("*/*.json"))
    target = {"artifacts": store.root / "artifacts", "artifact_directory": entry.parent,
              "entry": entry, "anchors": store.root / "anchors",
              "run_directory": bundle.parent.parent, "bundle": bundle}[location]
    retired = tmp_path / "retired"
    target.rename(retired)
    target.symlink_to(retired, target_is_directory=retired.is_dir())
    before = entry.read_bytes()
    with pytest.raises(RunStorageError):
        store.reindex()
    assert entry.read_bytes() == before


def test_index_is_create_only_even_if_another_writer_wins(tmp_path, monkeypatch):
    from lazybrick.runs import RunStorageError
    store, bundle = _succeeded_bundle(tmp_path / "store")
    entry = next((store.root / "artifacts").glob("*/*.json"))
    entry.unlink()
    real_link = os.link
    def racing_link(src, dst, *args, **kwargs):
        entry.write_bytes(b"independent winner")
        return real_link(src, dst, *args, **kwargs)
    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(RunStorageError, match="refusing to overwrite"):
        store.reindex()
    assert entry.read_bytes() == b"independent winner"
    assert list(entry.parent.glob(".pending-*")) == []


def test_promotion_receipt_survives_index_write_failure(tmp_path, monkeypatch):
    from lazybrick.runs import RunStorageError
    original = RunStore.index_attempt
    def fail(*args, **kwargs):
        raise OSError("simulated index failure")
    monkeypatch.setattr(RunStore, "index_attempt", fail)
    with pytest.raises(RunStorageError, match="promoted but could not be indexed"):
        _succeeded_bundle(tmp_path / "store")
    store = RunStore(tmp_path / "store")
    assert len(list((store.root / "runs").glob("*/attempts/*"))) == 1
    receipt = next((store.root / "anchors").glob("*/*.json"))
    before = receipt.read_bytes()
    monkeypatch.setattr(RunStore, "index_attempt", original)
    assert len(store.reindex()) == 1
    assert receipt.read_bytes() == before


def test_receipt_write_failure_prevents_promotion(tmp_path, monkeypatch):
    from lazybrick.runs import RunStorageError
    def fail(*args, **kwargs):
        raise RunStorageError("receipt write failed")
    monkeypatch.setattr(RunStore, "_record_promotion", fail)
    with pytest.raises(RunStorageError, match="receipt write failed"):
        _succeeded_bundle(tmp_path / "store")
    assert list((tmp_path / "store" / "runs").glob("*/attempts/*")) == []
    assert list((tmp_path / "store" / "artifacts").glob("*/*.json")) == []


@pytest.mark.parametrize("entry", ["text", [], None, 1, True, {},
    {"sha256":"a"*64}, {"size":1}, {"sha256":"a"*64,"size":True},
    {"sha256":"a"*64,"size":-1}, {"sha256":"a"*64,"size":1.5},
    {"sha256":"A"*64,"size":1}, {"sha256":42,"size":1},
    {"sha256":"short","size":1}, {"sha256":"a"*64,"size":1,"extra":0}])
@pytest.mark.parametrize("anchored", [False, True])
def test_malformed_manifest_entries_are_integrity_errors(tmp_path, entry, anchored):
    manifest = {"manifest_version":"0.1", "files":{"record.json":entry}}
    (tmp_path / "record.json").write_text("{}")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(BundleIntegrityError):
        read_manifest(tmp_path)
    with pytest.raises(BundleIntegrityError):
        verify_manifest(tmp_path, expected_digest="0"*64 if anchored else None)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a//b", "a/./b", "a/../b", "a\\b", "C:thing", "", MANIFEST_NAME])
def test_manifest_rejects_unsafe_or_self_paths(tmp_path, path):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"manifest_version":"0.1", "files":{
        path:{"sha256":"a"*64,"size":0}}}))
    with pytest.raises(BundleIntegrityError):
        verify_manifest(tmp_path)


@pytest.mark.parametrize("data", [
    b'{"manifest_version":"0.1","files":{},"files":{}}',
    b'{"manifest_version":"0.1","files":{"x":{"sha256":"a","size":NaN}}}',
    b'[]', b'null', b'\xff', b'{',
    b'{"manifest_version":"0.1","files":{},"extra":1}',
])
def test_invalid_manifest_json_is_an_integrity_error(tmp_path, data):
    (tmp_path / MANIFEST_NAME).write_bytes(data)
    with pytest.raises(BundleIntegrityError):
        read_manifest(tmp_path)


def test_idempotent_reindex_does_not_replace_existing_anchor_bytes(tmp_path):
    store, _ = _succeeded_bundle(tmp_path / "store")
    paths = [next((store.root / "artifacts").glob("*/*.json")),
             next((store.root / "anchors").glob("*/*.json"))]
    before = []
    for path in paths:
        os.utime(path, ns=(1234567890000000000, 1234567890000000000))
        before.append((path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()))
    store.reindex()
    assert [(p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in paths] == before


def test_tampering_between_promotion_and_indexing_is_rejected(tmp_path, monkeypatch):
    from lazybrick.runs import AttemptBundle, RunStorageError
    promote = AttemptBundle._promote_attempt
    def tampering_promote(self):
        destination = promote(self)
        (destination / "results.json").write_text('{"tampered":true}')
        _rewrite_manifest(destination)
        return destination
    monkeypatch.setattr(AttemptBundle, "_promote_attempt", tampering_promote)
    with pytest.raises(RunStorageError, match="promoted but could not be indexed"):
        _succeeded_bundle(tmp_path / "store")
    assert list((tmp_path / "store" / "artifacts").glob("*/*.json")) == []
    store = RunStore(tmp_path / "store")
    with pytest.raises(BundleIntegrityError, match="manifest itself was replaced"):
        store.reindex()
