"""Reference resolution. Every test runs offline against recorded fixtures."""

from __future__ import annotations

import json

import pytest

from lazybrick import DatasetRef, ModelRef, recipe_digest
from lazybrick.resolve import (
    HttpTransport,
    OfflineTransport,
    RecordedTransport,
    ResolutionError,
    ResolvedModel,
    Resolver,
    ResolverCache,
    classify,
)

QWEN3_4B = "1cfa9a7208912126459214e8b04321603b3df60c"


def model(uri: str = "hf://Qwen/Qwen3-4B", revision: str = QWEN3_4B) -> ModelRef:
    return ModelRef(uri, revision)


class TestModelResolution:
    def test_reads_metadata_without_downloading_weights(self, resolver, hf_transport):
        resolved = resolver.resolve_model(model())

        assert resolved.repo_id == "Qwen/Qwen3-4B"
        assert resolved.revision == QWEN3_4B
        assert resolved.architectures == ("Qwen3ForCausalLM",)
        assert resolved.dtype == "bfloat16"
        assert resolved.weight_format == "safetensors"
        assert resolved.license == "apache-2.0"
        assert resolved.parameter_count == 4022468096
        assert resolved.requires_remote_code is False
        # Two small metadata reads, and not one byte of a .safetensors file.
        assert len(hf_transport.requested) == 2
        assert not any(".safetensors" in url for url in hf_transport.requested)

    def test_branch_resolves_to_an_immutable_sha(self, resolver):
        resolved = resolver.resolve_model(model(revision="main"))

        assert resolved.requested_revision == "main"
        assert resolved.revision == QWEN3_4B

    def test_config_is_fetched_at_the_resolved_sha_not_the_branch(
        self, resolver, hf_transport
    ):
        resolver.resolve_model(model(revision="main"))

        assert hf_transport.requested[1].endswith(f"/resolve/{QWEN3_4B}/config.json")

    def test_unknown_scheme_is_refused(self, resolver):
        with pytest.raises(ResolutionError) as caught:
            resolver.resolve_model(model(uri="s3://bucket/model"))

        assert caught.value.codes == ("unsupported_scheme",)

    def test_unrecorded_model_is_an_error_not_a_network_call(self, resolver):
        with pytest.raises(ResolutionError) as caught:
            resolver.resolve_model(model(uri="hf://Qwen/Does-Not-Exist"))

        assert caught.value.codes == ("not_recorded",)


class TestClassification:
    """Component-level classification is what makes rejections explainable."""

    @pytest.mark.parametrize(
        ("repo", "profile", "components"),
        [
            ("hf://Qwen/Qwen3-4B", "dense-decoder", ("language_backbone",)),
            ("hf://Qwen/Qwen3-0.6B", "dense-decoder", ("language_backbone",)),
            (
                "hf://Qwen/Qwen2.5-VL-7B-Instruct",
                "multimodal-decoder",
                ("language_backbone", "vision_encoder"),
            ),
            (
                "hf://Qwen/Qwen3-30B-A3B",
                "moe-decoder",
                ("language_backbone", "moe_experts"),
            ),
        ],
    )
    def test_real_configs_classify_correctly(
        self, resolver, recorded_models, repo, profile, components
    ):
        sha = recorded_models[repo.removeprefix("hf://")]
        resolved = resolver.resolve_model(ModelRef(repo, sha))

        assert resolved.model_profile == profile
        assert resolved.components == components

    def test_audio_config_adds_an_audio_component(self):
        profile, components = classify({"model_type": "x", "audio_config": {}})

        assert profile == "multimodal-decoder"
        assert "audio_encoder" in components

    def test_moe_is_detected_from_expert_count_alone(self):
        profile, components = classify({"model_type": "llama", "num_local_experts": 8})

        assert profile == "moe-decoder"
        assert "moe_experts" in components

    def test_multimodal_wins_over_moe_for_the_profile(self):
        profile, components = classify(
            {"model_type": "x_moe", "vision_config": {}, "num_experts": 4}
        )

        assert profile == "multimodal-decoder"
        # ...but both components are still reported, so neither is hidden.
        assert {"vision_encoder", "moe_experts"} <= set(components)

    def test_auto_map_means_remote_code(self, resolver, tmp_path):
        transport = RecordedTransport(
            {
                "https://huggingface.co/api/models/x/y/revision/main": json.dumps(
                    {"sha": "f" * 40, "siblings": []}
                ).encode(),
                f"https://huggingface.co/x/y/resolve/{'f' * 40}/config.json": json.dumps(
                    {"model_type": "custom", "auto_map": {"AutoModel": "mod.Cls"}}
                ).encode(),
            }
        )
        resolved = Resolver(transport, ResolverCache(tmp_path)).resolve_model(
            ModelRef("hf://x/y", "main")
        )

        assert resolved.requires_remote_code is True
        assert resolved.weight_format == "pytorch"


class TestCache:
    def test_second_resolution_hits_the_cache(self, hf_transport, tmp_path):
        cache = ResolverCache(tmp_path / "c")
        first = Resolver(hf_transport, cache).resolve_model(model())
        before = len(hf_transport.requested)
        second = Resolver(hf_transport, cache).resolve_model(model())

        assert second == first
        assert len(hf_transport.requested) == before

    def test_cache_is_keyed_by_requested_revision(self, hf_transport, tmp_path):
        # "main" today and "main" tomorrow are different answers.
        cache = ResolverCache(tmp_path / "c")
        resolver = Resolver(hf_transport, cache)
        resolver.resolve_model(model(revision="main"))
        before = len(hf_transport.requested)
        resolver.resolve_model(model(revision=QWEN3_4B))

        assert len(hf_transport.requested) > before

    def test_corrupt_cache_entry_is_a_miss_not_a_crash(self, hf_transport, tmp_path):
        cache = ResolverCache(tmp_path / "c")
        Resolver(hf_transport, cache).resolve_model(model())
        for path in (tmp_path / "c").iterdir():
            path.write_text("{ not json", encoding="utf-8")

        assert Resolver(hf_transport, cache).resolve_model(model()).revision == QWEN3_4B

    def test_offline_uses_the_cache(self, hf_transport, tmp_path):
        cache = ResolverCache(tmp_path / "c")
        Resolver(hf_transport, cache).resolve_model(model())

        offline = Resolver(OfflineTransport(), cache, offline=True)
        assert offline.resolve_model(model()).revision == QWEN3_4B

    def test_offline_without_cache_refuses(self, tmp_path):
        offline = Resolver(OfflineTransport(), ResolverCache(tmp_path), offline=True)

        with pytest.raises(ResolutionError) as caught:
            offline.resolve_model(model())

        assert caught.value.codes == ("offline_unresolved",)


class TestRecipeResolution:
    def test_resolves_every_reference(self, resolver, valid_recipe):
        valid_recipe["model"]["revision"] = "main"
        valid_recipe["calibration"]["dataset"]["uri"] = (
            "hf-dataset://example/calibration-set"
        )
        valid_recipe["evaluation"]["dataset"]["uri"] = (
            "hf-dataset://example/held-out-set"
        )

        resolved = resolver.resolve_recipe(valid_recipe, recipe_digest(valid_recipe))

        assert resolved.recipe["model"]["revision"] == QWEN3_4B
        assert resolved.calibration_dataset.revision == "a" * 40
        assert resolved.evaluation_dataset.revision == "a" * 40

    def test_authored_recipe_is_not_mutated(self, resolver, valid_recipe):
        valid_recipe["model"]["revision"] = "main"
        resolver.resolve_recipe(valid_recipe, recipe_digest(valid_recipe))

        assert valid_recipe["model"]["revision"] == "main"

    def test_resolved_plan_differs_from_the_authored_one(self, resolver, valid_recipe):
        valid_recipe["model"]["revision"] = "main"
        digest_before = recipe_digest(valid_recipe)
        resolved = resolver.resolve_recipe(valid_recipe, digest_before)

        assert resolved.plan.model.is_pinned
        assert resolved.recipe_digest == digest_before

    def test_all_failures_are_reported_at_once(self, resolver, valid_recipe):
        valid_recipe["model"]["uri"] = "hf://Qwen/Nope"
        valid_recipe["calibration"]["dataset"]["uri"] = "hf-dataset://nope/nope"

        with pytest.raises(ResolutionError) as caught:
            resolver.resolve_recipe(valid_recipe, recipe_digest(valid_recipe))

        # Both failures surface together; resolution does not stop at the first.
        assert len(caught.value.issues) == 2
        assert [issue.path for issue in caught.value.issues] == [
            "https://huggingface.co/api/models/Qwen/Nope/revision/"
            "1cfa9a7208912126459214e8b04321603b3df60c",
            "https://huggingface.co/api/datasets/nope/nope/revision/" + "a" * 40,
        ]

    def test_unpinned_plugin_implementation_is_refused(self, resolver, valid_recipe):
        # Bypasses the schema the way a programmatically built plan would.
        valid_recipe["stages"][0]["implementation"] = {"git": "https://x", "commit": "abc"}
        valid_recipe["calibration"]["dataset"]["uri"] = (
            "hf-dataset://example/calibration-set"
        )
        valid_recipe["evaluation"]["dataset"]["uri"] = (
            "hf-dataset://example/held-out-set"
        )

        with pytest.raises(ResolutionError) as caught:
            resolver.resolve_recipe(valid_recipe, "0" * 64)

        assert "mutable_reference" in caught.value.codes

    def test_resolved_json_round_trips(self, resolver, valid_recipe):
        valid_recipe["calibration"]["dataset"]["uri"] = (
            "hf-dataset://example/calibration-set"
        )
        valid_recipe["evaluation"]["dataset"]["uri"] = (
            "hf-dataset://example/held-out-set"
        )
        resolved = resolver.resolve_recipe(valid_recipe, recipe_digest(valid_recipe))
        payload = resolved.to_json()

        assert ResolvedModel.from_json(payload["model"]) == resolved.model
        assert payload["resolved_recipe_version"] == "0.1"


def test_http_transport_is_the_default_when_online() -> None:
    assert isinstance(Resolver(cache=ResolverCache("/nonexistent"))._transport, HttpTransport)


def test_offline_resolver_defaults_to_the_offline_transport() -> None:
    resolver = Resolver(cache=ResolverCache("/nonexistent"), offline=True)

    assert isinstance(resolver._transport, OfflineTransport)
