from __future__ import annotations

import json
from pathlib import Path

import pytest

from lazybrick.runs import RunIdentity, RunStorageError, RunStore, hash_files, verify_hashes


def identity() -> RunIdentity:
    return RunIdentity.create(
        recipe_digest="a" * 64,
        plan_digest="b" * 64,
        artifact_id="c" * 64,
    )


def complete_records(bundle: object) -> None:
    bundle.write_text("recipe.yaml", "schema_version: '0.1'\n")
    bundle.write_json("resolved_recipe.json", {"resolved": True})
    bundle.write_json("plan.json", {"accepted": True})
    bundle.write_json("provenance.json", {"python": "test"})
    bundle.write_json("results.json", {"quality": {"baseline": 1, "quantized": 1}})


def test_success_is_atomically_promoted_and_indexed(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    current = identity()
    bundle = store.begin(current)
    complete_records(bundle)
    (bundle.artifact_dir / "model.safetensors").write_bytes(b"weights")

    destination = bundle.finalize_success({"format": "compressed-tensors/safetensors"})

    assert destination.is_dir()
    assert not bundle.staging.exists()
    assert json.loads((destination / "status.json").read_text())["state"] == "SUCCEEDED"
    manifest = json.loads((destination / "artifact.json").read_text())
    verify_hashes(destination / "artifact", manifest["files"])
    index = tmp_path / "artifacts" / current.artifact_id / f"{current.attempt_id}.json"
    assert index.is_file()
    assert json.loads(index.read_text())["attempt_id"] == current.attempt_id


def test_success_requires_complete_records(tmp_path: Path) -> None:
    bundle = RunStore(tmp_path).begin(identity())
    (bundle.artifact_dir / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(RunStorageError, match="missing records"):
        bundle.finalize_success({"format": "compressed-tensors/safetensors"})

    assert bundle.staging.exists()


def test_failed_attempt_is_retained_but_never_indexed_as_artifact(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    current = identity()
    bundle = store.begin(current)
    destination = bundle.finalize_failure(
        "BUILD_FAILED", {"code": "upstream", "message": "build failed"}
    )

    assert json.loads((destination / "status.json").read_text())["state"] == "BUILD_FAILED"
    assert not (tmp_path / "artifacts" / current.artifact_id).exists()


def test_retry_never_overwrites_prior_attempt(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    first = identity()
    first_path = store.begin(first).finalize_failure(
        "BUILD_FAILED", {"code": "first", "message": "first failure"}
    )
    retry = first.retry()
    retry_path = store.begin(retry).finalize_failure(
        "CANCELLED", {"code": "second", "message": "second failure"}
    )

    assert first_path != retry_path
    assert json.loads((first_path / "status.json").read_text())["failure"]["code"] == "first"
    assert json.loads((retry_path / "status.json").read_text())["failure"]["code"] == "second"


def test_hash_verification_detects_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    weight = artifact / "model.safetensors"
    weight.write_bytes(b"first")
    expected = hash_files(artifact)
    weight.write_bytes(b"second")

    with pytest.raises(RunStorageError, match="changed"):
        verify_hashes(artifact, expected)


def test_evidence_records_accept_finite_measured_floats(tmp_path: Path) -> None:
    bundle = RunStore(tmp_path).begin(identity())

    path = bundle.write_json("results.json", {"perplexity": 7.25})

    assert json.loads(path.read_text()) == {"perplexity": 7.25}


def test_run_records_reject_non_finite_floats(tmp_path: Path) -> None:
    bundle = RunStore(tmp_path).begin(identity())

    with pytest.raises(RunStorageError, match="finite JSON"):
        bundle.write_json("results.json", {"perplexity": float("nan")})


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json", "a/../../outside.json"])
def test_record_paths_cannot_escape_the_attempt(tmp_path: Path, path: str) -> None:
    bundle = RunStore(tmp_path).begin(identity())

    with pytest.raises(RunStorageError, match="inside the attempt bundle"):
        bundle.write_json(path, {"unsafe": True})


def test_unsafe_run_identifier_is_rejected(tmp_path: Path) -> None:
    current = identity()
    unsafe = RunIdentity(
        run_id="../outside",
        attempt_id=current.attempt_id,
        recipe_digest=current.recipe_digest,
        plan_digest=current.plan_digest,
        artifact_id=current.artifact_id,
    )

    with pytest.raises(RunStorageError, match="run_id"):
        RunStore(tmp_path).begin(unsafe)


def test_artifact_symlinks_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "linked.bin").symlink_to(outside)

    with pytest.raises(RunStorageError, match="symlinks"):
        hash_files(artifact)


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="requires FIFO support")
def test_hash_files_rejects_fifo_immediately(tmp_path):
    import os
    from lazybrick.runs.storage import hash_files
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(RunStorageError, match="only regular files"):
        hash_files(tmp_path)


def test_hash_files_rejects_symlinked_root(tmp_path):
    from lazybrick.runs.storage import hash_files
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RunStorageError, match="real directory"):
        hash_files(link)


def test_hash_files_rejects_missing_root(tmp_path):
    from lazybrick.runs.storage import hash_files
    with pytest.raises(RunStorageError, match="real directory"):
        hash_files(tmp_path / "missing")


@pytest.mark.parametrize("replacement", ["symlink", "fifo", "regular", "directory"])
def test_entry_replacement_between_inspection_and_open_is_rejected(tmp_path, monkeypatch, replacement):
    import os
    import lazybrick.runs.storage as storage
    root = tmp_path / "artifact"
    root.mkdir()
    victim = root / "weights"
    victim.write_bytes(b"original")
    outside = tmp_path / "outside"
    outside.write_bytes(b"must never be read")
    real_open, real_read = os.open, os.read
    opened, reads = [], []
    replaced = False

    def swapping_open(name, flags, *args, **kwargs):
        nonlocal replaced
        if name == "weights":
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_NONBLOCK  # a regression cannot hang on the FIFO
            victim.rename(root / "retired")
            if replacement == "symlink":
                victim.symlink_to(outside)
            elif replacement == "fifo":
                os.mkfifo(victim)
            elif replacement == "directory":
                victim.mkdir()
            else:
                victim.write_bytes(b"replacement")
            replaced = True
        fd = real_open(name, flags, *args, **kwargs)
        opened.append(fd)
        return fd

    def recording_read(fd, size):
        reads.append(fd)
        return real_read(fd, size)

    monkeypatch.setattr(os, "open", swapping_open)
    monkeypatch.setattr(os, "read", recording_read)
    with pytest.raises(RunStorageError):
        storage.hash_files(root)
    assert replaced
    assert not reads
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("swap_at", ["directory_open", "file_open", "root_open"])
def test_directory_replacement_cannot_redirect_descendant_reads(tmp_path, monkeypatch, swap_at):
    import os
    root = tmp_path / "artifact"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "weights").write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "weights").write_bytes(b"outside")
    (outside / "nested").mkdir()
    (outside / "nested" / "weights").write_bytes(b"outside")
    real_open, real_read = os.open, os.read
    replaced = False
    reads = []

    def swapping_open(name, flags, *args, **kwargs):
        nonlocal replaced
        trigger = {"directory_open":"nested", "file_open":"weights", "root_open":root}[swap_at]
        if not replaced and name == trigger:
            target = root if swap_at == "root_open" else nested
            target.rename(tmp_path / "retired")
            target.symlink_to(outside, target_is_directory=True)
            replaced = True
        return real_open(name, flags, *args, **kwargs)

    def recording_read(fd, size):
        block = real_read(fd, size)
        reads.append(block)
        return block

    monkeypatch.setattr(os, "open", swapping_open)
    monkeypatch.setattr(os, "read", recording_read)
    with pytest.raises(RunStorageError):
        hash_files(root)
    assert replaced
    assert b"outside" not in reads


@pytest.mark.parametrize("mutation", ["replace", "append", "truncate"])
def test_mutation_during_descriptor_read_is_rejected(tmp_path, monkeypatch, mutation):
    import os
    weight = tmp_path / "weights"
    weight.write_bytes(b"original")
    real_read = os.read
    changed = False
    def changing_read(fd, size):
        nonlocal changed
        block = real_read(fd, size)
        if not changed:
            if mutation == "replace":
                weight.rename(tmp_path / "retired")
                weight.write_bytes(b"different")
            elif mutation == "append":
                with weight.open("ab") as handle:
                    handle.write(b"additional")
            else:
                weight.write_bytes(b"")
            changed = True
        return block
    monkeypatch.setattr(os, "read", changing_read)
    with pytest.raises(RunStorageError, match="changed"):
        hash_files(tmp_path)
    assert changed


def test_safe_hashing_requires_descriptor_support(tmp_path, monkeypatch):
    import lazybrick.runs.storage as storage
    monkeypatch.setattr(storage, "_DESCRIPTOR_WALK_SUPPORTED", False)
    with pytest.raises(RunStorageError, match="requires descriptor"):
        hash_files(tmp_path)


def test_nested_files_and_empty_files_keep_byte_hashes(tmp_path):
    from hashlib import sha256
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "weights").write_bytes(b"unchanged")
    (tmp_path / "empty").touch()
    assert hash_files(tmp_path) == {
        "empty": sha256(b"").hexdigest(), "nested/weights": sha256(b"unchanged").hexdigest(),
    }
