"""Record round-tripping, identity separation, and optional-field omission."""

from __future__ import annotations

import pytest

from lazybrick import (
    ArtifactManifest,
    CapabilityReport,
    EvidenceRecord,
    ExecutionPlan,
    ExportSpec,
    FileRef,
    ImplementationRef,
    MeasurementSeries,
    ModelRef,
    PluginManifest,
    ProvenanceRecord,
    canonical_json,
    load_recipe,
    recipe_digest,
)

# Duplicated from conftest rather than imported: `tests` is not a package,
# so `from tests.conftest import ...` only resolves when pytest happens to
# put the repo root on sys.path.
MODEL_SHA = "1cfa9a7208912126459214e8b04321603b3df60c"
PLUGIN_SHA = "de46bfd53513aa87571a8b056a06aeaa5da1c69c"


def plan_from(recipe: dict) -> ExecutionPlan:
    return ExecutionPlan.from_recipe(recipe, recipe_digest(recipe))


class TestRoundTrip:
    def test_plan_round_trips(self, valid_recipe) -> None:
        plan = plan_from(valid_recipe)

        assert ExecutionPlan.from_json(plan.to_json()) == plan

    def test_plan_round_trips_without_optional_sections(self, valid_recipe) -> None:
        valid_recipe.pop("calibration")
        valid_recipe.pop("evaluation")
        plan = plan_from(valid_recipe)

        assert plan.calibration is None
        assert ExecutionPlan.from_json(plan.to_json()) == plan

    @pytest.mark.parametrize(
        "record",
        [
            ModelRef("hf://Qwen/Qwen3-4B", MODEL_SHA),
            ImplementationRef(git="https://example.invalid/x", commit=PLUGIN_SHA),
            ExportSpec("compressed-tensors", "vllm"),
            CapabilityReport("plugin:awq", {"quantization_scheme": ("W4A16",)}),
            FileRef("model.safetensors", 4096, "0" * 64),
            MeasurementSeries("ttft", "ms", ("12.1", "12.4")),
        ],
    )
    def test_record_round_trips(self, record) -> None:
        assert type(record).from_json(record.to_json()) == record

    def test_plugin_manifest_round_trips(self) -> None:
        manifest = PluginManifest(
            name="awq",
            plugin_api="0.1",
            kind="transformation",
            version="0.1.0",
            implementation=ImplementationRef(
                git="https://github.com/vllm-project/llm-compressor",
                commit=PLUGIN_SHA,
            ),
            capabilities={"quantization_scheme": ("W4A16",), "runtime": ("vllm",)},
            requires={"calibration": True},
            licenses={"plugin": "Apache-2.0"},
        )

        assert PluginManifest.from_json(manifest.to_json()) == manifest

    def test_provenance_round_trips(self) -> None:
        record = ProvenanceRecord(
            run_id="r1",
            attempt_id="a1",
            recipe_digest="0" * 64,
            plan_digest="1" * 64,
            status="SUCCEEDED",
            versions={"python": "3.12.1"},
            seeds={"calibration": 42},
        )

        assert ProvenanceRecord.from_json(record.to_json()) == record

    def test_evidence_round_trips(self) -> None:
        record = EvidenceRecord(
            protocol_id="token-loss-v1",
            metric="perplexity",
            baseline_score="8.4213",
            compressed_score="8.6001",
            absolute_delta="0.1788",
            relative_delta="0.0212",
            measurements=(MeasurementSeries("ttft", "ms", ("11.9", "12.2")),),
        )

        assert EvidenceRecord.from_json(record.to_json()) == record


class TestOptionalFields:
    def test_unset_optionals_are_omitted_not_null(self) -> None:
        # An omitted field cannot change the digest of a document not using it.
        assert ImplementationRef(container="x@sha256:" + "a" * 64).to_json() == {
            "container": "x@sha256:" + "a" * 64
        }

    def test_omitted_evidence_note_is_absent(self) -> None:
        record = EvidenceRecord("p", "m", "1", "2", "1", "1")

        assert "notes" not in record.to_json()

    def test_absent_calibration_is_absent_from_plan_json(self, valid_recipe) -> None:
        valid_recipe.pop("calibration")

        assert "calibration" not in plan_from(valid_recipe).to_json()


class TestIdentities:
    """recipe_digest, plan_digest, and artifact_id answer different questions."""

    def test_the_three_identities_differ(self, valid_recipe) -> None:
        plan = plan_from(valid_recipe)

        assert len({plan.recipe_digest, plan.plan_digest, plan.artifact_id}) == 3

    def test_target_changes_the_plan_but_not_the_artifact(self, valid_recipe) -> None:
        before = plan_from(valid_recipe)
        valid_recipe["target"]["min_memory_gib"] = 80
        after = plan_from(valid_recipe)

        assert after.plan_digest != before.plan_digest
        assert after.artifact_id == before.artifact_id

    def test_evaluation_changes_the_plan_but_not_the_artifact(
        self, valid_recipe
    ) -> None:
        before = plan_from(valid_recipe)
        valid_recipe["evaluation"]["max_samples"] = 64
        after = plan_from(valid_recipe)

        assert after.plan_digest != before.plan_digest
        assert after.artifact_id == before.artifact_id

    @pytest.mark.parametrize(
        ("section", "key", "value"),
        [
            ("model", "revision", "b" * 40),
            ("export", "format", "safetensors"),
        ],
    )
    def test_input_changes_move_the_artifact_id(
        self, valid_recipe, section: str, key: str, value: str
    ) -> None:
        before = plan_from(valid_recipe)
        valid_recipe[section][key] = value

        assert plan_from(valid_recipe).artifact_id != before.artifact_id

    def test_quantization_parameters_move_the_artifact_id(self, valid_recipe) -> None:
        before = plan_from(valid_recipe)
        valid_recipe["stages"][0]["parameters"]["group_size"] = 64

        assert plan_from(valid_recipe).artifact_id != before.artifact_id

    def test_calibration_seed_moves_the_artifact_id(self, valid_recipe) -> None:
        before = plan_from(valid_recipe)
        valid_recipe["calibration"]["seed"] = 43

        assert plan_from(valid_recipe).artifact_id != before.artifact_id

    def test_plan_json_is_byte_identical_for_equal_inputs(self, valid_recipe) -> None:
        reordered = dict(reversed(list(valid_recipe.items())))

        assert canonical_json(plan_from(valid_recipe).to_json()) == canonical_json(
            plan_from(reordered).to_json()
        )


class TestPinning:
    def test_unpinned_model_is_detected(self, valid_recipe) -> None:
        valid_recipe["model"]["revision"] = "main"

        assert plan_from(valid_recipe).model.is_pinned is False

    def test_pinned_model_is_detected(self, valid_recipe) -> None:
        assert plan_from(valid_recipe).model.is_pinned is True

    @pytest.mark.parametrize(
        ("ref", "pinned"),
        [
            (ImplementationRef(commit=PLUGIN_SHA), True),
            (ImplementationRef(commit="de46bfd"), False),
            (ImplementationRef(container="ghcr.io/x@sha256:" + "a" * 64), True),
            (ImplementationRef(container="ghcr.io/x:latest"), False),
        ],
    )
    def test_implementation_pinning(self, ref: ImplementationRef, pinned: bool) -> None:
        assert ref.is_pinned is pinned


def test_artifact_manifest_round_trips() -> None:
    manifest = ArtifactManifest(
        artifact_id="a" * 64,
        plan_digest="b" * 64,
        recipe_digest="c" * 64,
        export=ExportSpec("compressed-tensors", "vllm"),
        files=(FileRef("model.safetensors", 10, "d" * 64),),
    )

    assert ArtifactManifest.from_json(manifest.to_json()) == manifest


def test_example_recipe_produces_a_plan() -> None:
    document = load_recipe("examples/qwen3-awq.yaml")
    plan = ExecutionPlan.from_recipe(document.data, document.digest)

    assert plan.model.uri == "hf://Qwen/Qwen3-4B"
    assert plan.model.is_pinned
    assert plan.stages[0].implementation.is_pinned
