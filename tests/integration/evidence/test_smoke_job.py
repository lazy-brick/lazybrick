from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

from lazybrick.runs import RunIdentity, RunStore


ROOT = Path(__file__).parents[3]


def load_script(name: str) -> object:
    path = ROOT / "gpu" / "smoke" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lazybrick_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_recipe_uses_exact_pins() -> None:
    recipe = (ROOT / "gpu" / "smoke" / "qwen3-0.6b-awq.yaml").read_text()

    assert "c1899de289a04d12100db370d81485cdf75e47ca" in recipe
    assert "8049631c405ae6576f93f445c6b8166f76f5505a" in recipe
    assert "de46bfd53513aa87571a8b056a06aeaa5da1c69c" in recipe
    assert "weight_bits: 4" in recipe
    assert "group_size: 128" in recipe
    assert "symmetric: false" in recipe


def test_plan_contract_fails_closed() -> None:
    module = load_script("job")

    with pytest.raises(module.SmokeJobError, match="plan_digest"):
        module.require_plan_fields(
            {
                "accepted": True,
                "recipe_digest": "a" * 64,
                "resolved_recipe": {},
                "artifact_id": "c" * 64,
                "compatibility": {"accepted": True},
            }
        )


def test_plan_contract_accepts_planner_output() -> None:
    module = load_script("job")
    resolved = {"schema_version": "0.1"}

    fields = module.require_plan_fields(
        {
            "recipe_digest": "a" * 64,
            "plan_digest": "b" * 64,
            "artifact_id": "c" * 64,
            "resolved_recipe": resolved,
            "compatibility": {"accepted": True},
        }
    )

    assert fields == ("a" * 64, "b" * 64, "c" * 64, resolved)


def test_bundle_verifier_requires_success_and_raw_evidence(tmp_path: Path) -> None:
    module = load_script("verify_bundle")
    store = RunStore(tmp_path)
    identity = RunIdentity.create(
        recipe_digest="a" * 64, plan_digest="b" * 64, artifact_id="c" * 64
    )
    bundle = store.begin(identity)
    bundle.write_text("recipe.yaml", "schema_version: '0.1'\n")
    bundle.write_json("resolved_recipe.json", {"resolved": True})
    bundle.write_json(
        "plan.json",
        {
            "accepted": True,
            "recipe_digest": identity.recipe_digest,
            "plan_digest": identity.plan_digest,
            "artifact_id": identity.artifact_id,
        },
    )
    bundle.write_json("provenance.json", {"python": "test"})
    bundle.write_json(
        "results.json",
        {
            "generations": [{"prompt": "a", "output": "b"}],
            "quality": {"baseline": {"score": 1}, "quantized": {"score": 1}},
            "performance": {
                "baseline_raw_samples": [{"latency": 1}],
                "quantized_raw_samples": [{"latency": 1}],
            },
        },
    )
    bundle.write_json("state-history.json", {"state": "SUCCEEDED"})
    (bundle.artifact_dir / "model.safetensors").write_bytes(b"weights")
    bundle.finalize_success({"format": "compressed-tensors/safetensors"})

    verified = module.verify(tmp_path)

    assert verified.name == identity.attempt_id


def test_verifier_rejects_failed_attempt(tmp_path: Path) -> None:
    module = load_script("verify_bundle")
    store = RunStore(tmp_path)
    identity = RunIdentity.create(
        recipe_digest="a" * 64, plan_digest="b" * 64, artifact_id="c" * 64
    )
    store.begin(identity).finalize_failure(
        "BUILD_FAILED", {"code": "test", "message": "failed"}
    )

    with pytest.raises(RuntimeError, match="did not succeed"):
        module.verify(tmp_path)


@pytest.mark.gpu
def test_qwen3_0_6b_awq_smoke_job(tmp_path: Path) -> None:
    if os.environ.get("LAZYBRICK_RUN_GPU_TESTS") != "1":
        pytest.skip("set LAZYBRICK_RUN_GPU_TESTS=1 after explicit GPU-run approval")
    environment = os.environ.copy()
    environment["LAZYBRICK_SMOKE_ROOT"] = str(tmp_path / "smoke")

    subprocess.run(
        ["bash", str(ROOT / "gpu" / "smoke" / "run.sh")],
        check=True,
        cwd=ROOT,
        env=environment,
    )
