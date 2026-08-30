from __future__ import annotations

import pytest

from lazybrick.runs import IdentityError, RunIdentity, artifact_id


def inputs() -> dict[str, object]:
    return {
        "model": {"uri": "hf://Qwen/Qwen3-4B", "revision": "a" * 40},
        "plugin": {"package": "lazybrick", "version": "0.0.1", "commit": "b" * 40},
        "calibration": {"revision": "c" * 40, "seed": 42, "samples": 512},
        "quantization": {"algorithm": "awq", "weight_bits": 4, "group_size": 128},
        "export": {"format": "compressed-tensors/safetensors", "runtime": "vllm"},
    }


def test_artifact_id_is_stable_and_not_output_hash() -> None:
    first = artifact_id(inputs())
    reordered = dict(reversed(list(inputs().items())))

    assert first == artifact_id(reordered)
    assert len(first) == 64


def test_artifact_id_requires_exact_resolved_tuple() -> None:
    incomplete = inputs()
    incomplete.pop("calibration")

    with pytest.raises(IdentityError, match="missing"):
        artifact_id(incomplete)


def test_floating_point_identity_is_rejected() -> None:
    value = inputs()
    value["quantization"]["ratio"] = 0.5

    with pytest.raises(IdentityError, match="floating-point"):
        artifact_id(value)


def test_retry_preserves_run_and_content_identities() -> None:
    digest = "d" * 64
    identity = RunIdentity.create(
        recipe_digest=digest, plan_digest="e" * 64, artifact_id="f" * 64
    )
    retry = identity.retry()

    assert retry.run_id == identity.run_id
    assert retry.attempt_id != identity.attempt_id
    assert retry.recipe_digest == identity.recipe_digest
    assert retry.plan_digest == identity.plan_digest
    assert retry.artifact_id == identity.artifact_id
    assert len({identity.run_id, identity.attempt_id, digest}) == 3
