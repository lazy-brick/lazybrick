"""Compatibility checking.

The negative cases here are the M0 exit gate: Qwen-VL and MoE must be rejected
with the *correct component-level reason*, not a generic "unsupported".
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lazybrick import ExecutionPlan, ModelRef, recipe_digest
from lazybrick.capabilities import (
    HardwareProfile,
    check_compatibility,
    scheme_for_stage,
)


def resolve(resolver, recorded_models, repo: str):
    return resolver.resolve_model(ModelRef(f"hf://{repo}", recorded_models[repo]))


def plan_for(recipe: dict) -> ExecutionPlan:
    return ExecutionPlan.from_recipe(recipe, recipe_digest(recipe))


@pytest.fixture
def check(resolver, recorded_models, valid_recipe, awq_manifest, a100_40gb):
    """Check the reference recipe against a repo, with optional overrides."""

    def run(repo="Qwen/Qwen3-4B", hardware=..., recipe=None, manifest=None):
        recipe = recipe if recipe is not None else valid_recipe
        return check_compatibility(
            plan_for(recipe),
            resolve(resolver, recorded_models, repo),
            [manifest or awq_manifest],
            a100_40gb if hardware is ... else hardware,
        )

    return run


class TestAcceptance:
    def test_the_reference_recipe_is_accepted(self, check) -> None:
        result = check()

        assert result.accepted, result.reasons
        assert result.reasons == ()

    def test_the_smoke_model_is_accepted(self, check) -> None:
        assert check(repo="Qwen/Qwen3-0.6B").accepted

    def test_reports_name_every_participant(self, check) -> None:
        subjects = {report.subject for report in check().reports}

        assert subjects == {"model", "hardware", "plugin:awq", "runtime:vllm"}


class TestArchitectureRejections:
    """The two the exit gate names explicitly."""

    def test_qwen_vl_is_rejected_at_the_vision_encoder(self, check) -> None:
        result = check(repo="Qwen/Qwen2.5-VL-7B-Instruct")

        assert not result.accepted
        assert "unsupported_component" in result.codes
        reason = next(r for r in result.reasons if r.code == "unsupported_component")
        assert reason.required == ("vision_encoder",)
        assert "vision encoder" in reason.detail
        assert reason.subject == "plugin:awq"

    def test_qwen_moe_is_rejected_at_the_experts(self, check) -> None:
        result = check(repo="Qwen/Qwen3-30B-A3B")

        assert not result.accepted
        reason = next(r for r in result.reasons if r.code == "unsupported_component")
        assert reason.required == ("moe_experts",)
        assert "moe experts" in reason.detail

    def test_profile_and_component_are_both_reported(self, check) -> None:
        # The profile says "not this kind of model"; the component says which
        # part. A user needs both to know whether the recipe is salvageable.
        result = check(repo="Qwen/Qwen2.5-VL-7B-Instruct")

        assert "unsupported_model_profile" in result.codes
        assert "unsupported_component" in result.codes


class TestHardwareRejections:
    def test_missing_accelerator(self, check) -> None:
        result = check(hardware=None)

        assert result.codes == ("missing_accelerator",)
        assert "--target" in result.reasons[0].detail

    def test_explicitly_absent_accelerator_is_the_same_answer(self, check) -> None:
        assert check(hardware=HardwareProfile.none()).codes == ("missing_accelerator",)

    def test_insufficient_vram(self, check, a100_40gb) -> None:
        small = HardwareProfile("nvidia", 1, "8.9", 24, "NVIDIA L4")
        result = check(hardware=small)

        assert "insufficient_memory" in result.codes
        assert "40 GiB" in str(result.reasons[0]) or any(
            "40 GiB" in r.detail for r in result.reasons
        )

    def test_insufficient_compute_capability(self, check) -> None:
        old = HardwareProfile("nvidia", 1, "7.5", 48, "NVIDIA T4")
        result = check(hardware=old)

        assert "insufficient_compute_capability" in result.codes

    def test_compute_capability_compares_numerically_not_as_text(self, check) -> None:
        # "10.0" > "8.0" numerically, but sorts before it as a string.
        modern = HardwareProfile("nvidia", 1, "10.0", 96)

        assert "insufficient_compute_capability" not in check(hardware=modern).codes

    def test_wrong_vendor(self, check) -> None:
        amd = HardwareProfile("amd", 1, "9.0", 128, "MI300X")
        result = check(hardware=amd)

        assert "accelerator_vendor_mismatch" in result.codes

    def test_insufficient_devices(self, check, valid_recipe) -> None:
        valid_recipe["target"]["device_count"] = 2
        result = check(recipe=valid_recipe)

        assert "insufficient_devices" in result.codes


class TestRecipeRejections:
    def test_unsupported_runtime(self, check, valid_recipe) -> None:
        valid_recipe["export"]["runtime"] = "tensorrt-llm"
        valid_recipe["target"]["runtime"] = "tensorrt-llm"
        result = check(recipe=valid_recipe)

        assert "unsupported_runtime" in result.codes
        assert any("vllm" in reason.available for reason in result.reasons)

    def test_export_and_target_runtime_must_agree(self, check, valid_recipe) -> None:
        valid_recipe["target"]["runtime"] = "sglang"
        result = check(recipe=valid_recipe)

        assert "runtime_mismatch" in result.codes

    def test_missing_calibration_data(self, check, valid_recipe) -> None:
        valid_recipe.pop("calibration")
        result = check(recipe=valid_recipe)

        assert "calibration_required" in result.codes

    def test_unsupported_scheme(self, check, valid_recipe) -> None:
        valid_recipe["stages"][0]["parameters"]["weight_bits"] = 8
        result = check(recipe=valid_recipe)

        assert "unsupported_quantization_scheme" in result.codes
        reason = next(
            r for r in result.reasons if r.code == "unsupported_quantization_scheme"
        )
        assert reason.required == ("W8A16",)

    def test_unsupported_export_format(self, check, valid_recipe) -> None:
        valid_recipe["export"]["format"] = "gguf"
        result = check(recipe=valid_recipe)

        assert "unsupported_export_format" in result.codes

    def test_unknown_plugin(self, check, valid_recipe) -> None:
        valid_recipe["stages"][0]["plugin"] = "lazybrick.plugin/gptq"
        result = check(recipe=valid_recipe)

        assert "unknown_plugin" in result.codes

    def test_manifest_version_must_match_stage(self, check, awq_manifest) -> None:
        result = check(manifest=replace(awq_manifest, version="9.9.9"))

        assert "plugin_version_mismatch" in result.codes

    def test_manifest_implementation_must_match_stage(
        self, check, awq_manifest
    ) -> None:
        other = replace(
            awq_manifest,
            implementation=replace(awq_manifest.implementation, commit="f" * 40),
        )

        assert "plugin_implementation_mismatch" in check(manifest=other).codes

    def test_plugin_input_format_is_checked(self, check, awq_manifest) -> None:
        capabilities = dict(awq_manifest.capabilities)
        capabilities["input_format"] = ("pytorch",)

        result = check(manifest=replace(awq_manifest, capabilities=capabilities))

        assert "unsupported_input_format" in result.codes

    def test_mutable_model_revision(
        self, resolver, recorded_models, valid_recipe, awq_manifest, a100_40gb
    ) -> None:
        # A plan built from an *unresolved* recipe: the revision still moves.
        valid_recipe["model"]["revision"] = "main"
        model = resolver.resolve_model(ModelRef("hf://Qwen/Qwen3-4B", "main"))
        unpinned = replace(model, revision="main")
        result = check_compatibility(
            plan_for(valid_recipe), unpinned, [awq_manifest], a100_40gb
        )

        assert "mutable_reference" in result.codes

    def test_remote_code_is_refused_unless_opted_into(
        self, valid_recipe, awq_manifest, a100_40gb, resolver, recorded_models
    ) -> None:
        model = resolve(resolver, recorded_models, "Qwen/Qwen3-4B")
        needs_code = replace(model, requires_remote_code=True)
        result = check_compatibility(
            plan_for(valid_recipe), needs_code, [awq_manifest], a100_40gb
        )

        assert "remote_code_required" in result.codes

        valid_recipe["model"]["trust_remote_code"] = True
        accepted = check_compatibility(
            plan_for(valid_recipe), needs_code, [awq_manifest], a100_40gb
        )
        assert "remote_code_required" not in accepted.codes


class TestReporting:
    def test_every_reason_is_collected(self, check, valid_recipe) -> None:
        valid_recipe.pop("calibration")
        valid_recipe["export"]["format"] = "gguf"
        result = check(recipe=valid_recipe, hardware=HardwareProfile("amd", 1, "9.0", 8))

        assert len(result.codes) >= 4

    def test_result_serializes(self, check) -> None:
        payload = check(repo="Qwen/Qwen3-30B-A3B").to_json()

        assert payload["accepted"] is False
        assert payload["reasons"][0]["code"]
        assert payload["reports"]

    def test_reason_reads_as_a_sentence(self, check) -> None:
        reason = next(
            r
            for r in check(repo="Qwen/Qwen2.5-VL-7B-Instruct").reasons
            if r.code == "unsupported_component"
        )

        assert str(reason).startswith("plugin:awq: awq supports the language backbone")


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ({"weight_bits": 4}, "W4A16"),
        ({"weight_bits": 8}, "W8A16"),
        ({"scheme": "W4A8"}, "W4A8"),
        ({"weight_bits": 4, "scheme": "custom"}, "custom"),
        ({}, "unknown"),
        ({"weight_bits": True}, "unknown"),
    ],
)
def test_scheme_derivation(valid_recipe, parameters: dict, expected: str) -> None:
    valid_recipe["stages"][0]["parameters"] = parameters
    stage = plan_for(valid_recipe).stages[0]

    assert scheme_for_stage(stage) == expected
