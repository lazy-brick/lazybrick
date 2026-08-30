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
