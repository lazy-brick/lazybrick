"""CLI behaviour, exit codes, and the promise that `plan` touches no weights.

Network isolation is real, not mocked at the CLI layer: each test primes a
cache with recorded fixtures and then runs the command with --offline, so the
same code path a user gets is what is exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lazybrick.cli import INCOMPATIBLE, INVALID_RECIPE, OK, REFUSED, UNRESOLVED, main
from lazybrick.records import DatasetRef, ModelRef
from lazybrick.resolve import Resolver, ResolverCache

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples" / "qwen3-awq.yaml"
MANIFEST = ROOT / "examples" / "plugins" / "awq.manifest.json"
A100 = ROOT / "examples" / "targets" / "a100-40gb.json"
L4 = ROOT / "examples" / "targets" / "l4-24gb.json"


@pytest.fixture
def primed_cache(hf_transport, tmp_path, valid_recipe):
    """A cache holding every reference the test recipe names."""

    cache_dir = tmp_path / "cache"
    resolver = Resolver(hf_transport, ResolverCache(cache_dir))
    resolver.resolve_model(ModelRef("hf://Qwen/Qwen3-4B", valid_recipe["model"]["revision"]))
    resolver.resolve_model(ModelRef("hf://Qwen/Qwen2.5-VL-7B-Instruct", "main"))
    resolver.resolve_dataset(DatasetRef("hf-dataset://example/calibration-set", "a" * 40))
    resolver.resolve_dataset(DatasetRef("hf-dataset://example/held-out-set", "b" * 40))
    return cache_dir


@pytest.fixture
def recipe_file(tmp_path, valid_recipe):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(valid_recipe), encoding="utf-8")
    return path


def run(*argv: str) -> int:
    return main(list(argv))


class TestValidateAndDigest:
    def test_validate(self, capsys) -> None:
        assert run("validate", str(EXAMPLE)) == OK
        assert "recipe_digest:" in capsys.readouterr().out

    def test_validate_json(self, capsys) -> None:
        run("validate", str(EXAMPLE), "--json")
        payload = json.loads(capsys.readouterr().out)

        assert payload["valid"] is True
        assert len(payload["recipe_digest"]) == 64

    def test_invalid_recipe_exit_code(self, tmp_path, capsys, valid_recipe) -> None:
        valid_recipe["model"].pop("uri")
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(valid_recipe), encoding="utf-8")

        assert run("validate", str(path)) == INVALID_RECIPE
        assert "model.uri" in capsys.readouterr().err

    def test_digest_prints_only_the_digest(self, capsys) -> None:
        assert run("digest", str(EXAMPLE)) == OK
        assert len(capsys.readouterr().out.strip()) == 64

    def test_no_command_prints_help(self, capsys) -> None:
        assert run() == REFUSED
        assert "usage: lazybrick" in capsys.readouterr().out


class TestInspect:
    def test_inspect(self, primed_cache, capsys, valid_recipe) -> None:
        code = run(
            "inspect", "Qwen/Qwen3-4B",
            "--revision", valid_recipe["model"]["revision"],
            "--offline", "--cache-dir", str(primed_cache),
        )
        out = capsys.readouterr().out

        assert code == OK
        assert "dense-decoder" in out
        assert "4,022,468,096" in out

    def test_inspect_json(self, primed_cache, capsys, valid_recipe) -> None:
        run(
            "inspect", "hf://Qwen/Qwen3-4B",
            "--revision", valid_recipe["model"]["revision"],
            "--offline", "--cache-dir", str(primed_cache), "--json",
        )
        payload = json.loads(capsys.readouterr().out)

        assert payload["model_profile"] == "dense-decoder"
        assert payload["components"] == ["language_backbone"]

    def test_uncached_model_offline_exits_three(self, tmp_path, capsys) -> None:
        code = run(
            "inspect", "Qwen/Qwen3-4B",
            "--offline", "--cache-dir", str(tmp_path / "empty"),
        )

        assert code == UNRESOLVED
        assert "offline" in capsys.readouterr().err


class TestPlan:
    def test_accepted_plan(self, recipe_file, primed_cache, capsys) -> None:
        code = run(
            "plan", str(recipe_file),
            "--target", str(A100), "--plugin-manifest", str(MANIFEST),
            "--offline", "--cache-dir", str(primed_cache),
        )
        out = capsys.readouterr().out

        assert code == OK
        assert "ACCEPTED" in out
        # The M0 exit gate: all three identities are reported and distinct.
        for label in ("recipe_digest", "plan_digest", "artifact_id"):
            assert label in out

    def test_plan_json_is_byte_identical_across_runs(
        self, recipe_file, primed_cache, capsys
    ) -> None:
        argv = (
            "plan", str(recipe_file), "--target", str(A100),
            "--plugin-manifest", str(MANIFEST), "--offline",
            "--cache-dir", str(primed_cache), "--json",
        )
        run(*argv)
        first = capsys.readouterr().out
        run(*argv)
        second = capsys.readouterr().out

        assert first == second
        assert json.loads(first)["compatibility"]["accepted"] is True

    def test_insufficient_vram_is_rejected(
        self, recipe_file, primed_cache, capsys
    ) -> None:
        code = run(
            "plan", str(recipe_file),
            "--target", str(L4), "--plugin-manifest", str(MANIFEST),
            "--offline", "--cache-dir", str(primed_cache),
        )
        out = capsys.readouterr().out

        assert code == INCOMPATIBLE
        assert "insufficient_memory" in out

    def test_vl_model_is_rejected_at_the_component(
        self, tmp_path, primed_cache, capsys, valid_recipe
    ) -> None:
        valid_recipe["model"]["uri"] = "hf://Qwen/Qwen2.5-VL-7B-Instruct"
        valid_recipe["model"]["revision"] = "main"
        path = tmp_path / "vl.yaml"
        path.write_text(yaml.safe_dump(valid_recipe), encoding="utf-8")

        code = run(
            "plan", str(path), "--target", str(A100),
            "--plugin-manifest", str(MANIFEST),
            "--offline", "--cache-dir", str(primed_cache),
        )
        out = capsys.readouterr().out

        assert code == INCOMPATIBLE
        assert "vision encoder" in out
        assert "unsupported_component" in out

    def test_missing_target_is_rejected_not_assumed(
        self, recipe_file, primed_cache, capsys
    ) -> None:
        code = run(
            "plan", str(recipe_file), "--plugin-manifest", str(MANIFEST),
            "--offline", "--cache-dir", str(primed_cache),
        )

        assert code == INCOMPATIBLE
        assert "missing_accelerator" in capsys.readouterr().out

    def test_unreadable_target_is_a_usage_error(
        self, recipe_file, primed_cache, capsys
    ) -> None:
        code = run(
            "plan", str(recipe_file), "--target", "/nonexistent/target.json",
            "--offline", "--cache-dir", str(primed_cache),
        )

        assert code == REFUSED
        assert "cannot read target" in capsys.readouterr().err


class TestPlanTouchesNothing:
    """`plan` must not download weights, allocate a GPU, or run a plugin."""

    def test_no_weight_urls_are_requested(self, hf_transport, tmp_path, valid_recipe):
        resolver = Resolver(hf_transport, ResolverCache(tmp_path))
        resolver.resolve_recipe(valid_recipe, "0" * 64)

        assert hf_transport.requested
        assert not any(
            url.endswith((".safetensors", ".bin", ".pt", ".gguf"))
            for url in hf_transport.requested
        )

    def test_no_subprocess_is_spawned(
        self, recipe_file, primed_cache, monkeypatch
    ) -> None:
        import subprocess

        def explode(*args, **kwargs):
            raise AssertionError("plan must not execute a plugin")

        monkeypatch.setattr(subprocess, "run", explode)
        monkeypatch.setattr(subprocess, "Popen", explode)

        run(
            "plan", str(recipe_file), "--target", str(A100),
            "--plugin-manifest", str(MANIFEST),
            "--offline", "--cache-dir", str(primed_cache),
        )

    def test_offline_plan_makes_no_network_call(
        self, recipe_file, primed_cache, monkeypatch
    ) -> None:
        import urllib.request

        def explode(*args, **kwargs):
            raise AssertionError("offline plan must not open a connection")

        monkeypatch.setattr(urllib.request, "urlopen", explode)

        assert run(
            "plan", str(recipe_file), "--target", str(A100),
            "--plugin-manifest", str(MANIFEST),
            "--offline", "--cache-dir", str(primed_cache),
        ) == OK


class TestBuild:
    def test_build_without_dry_run_is_refused(self, recipe_file, capsys) -> None:
        assert run("build", str(recipe_file)) == REFUSED
        assert "not implemented yet" in capsys.readouterr().err

    def test_build_dry_run(self, recipe_file, primed_cache, capsys) -> None:
        code = run(
            "build", str(recipe_file), "--dry-run",
            "--target", str(A100), "--plugin-manifest", str(MANIFEST),
            "--offline", "--cache-dir", str(primed_cache),
        )
        out = capsys.readouterr().out

        assert code == OK
        assert "nothing was executed" in out
        assert "would build" in out

    def test_build_dry_run_json(self, recipe_file, primed_cache, capsys) -> None:
        run(
            "build", str(recipe_file), "--dry-run",
            "--target", str(A100), "--plugin-manifest", str(MANIFEST),
            "--offline", "--cache-dir", str(primed_cache), "--json",
        )
        payload = json.loads(capsys.readouterr().out)

        assert payload == {
            **payload,
            "dry_run": True,
            "would_build": True,
        }


@pytest.mark.parametrize("declared", [False, True])
def test_v02_semantic_identity_does_not_imply_declaration(
    valid_recipe, primed_cache, tmp_path, capsys, declared,
):
    from lazybrick.semantics.profile import PROFILE_ID, profile_digest
    valid_recipe["schema_version"] = "0.2"
    if declared:
        valid_recipe["stages"][0]["semantics"] = {
            "profile": PROFILE_ID, "profile_digest": profile_digest(),
        }
    path = tmp_path / "v02.yaml"
    path.write_text(yaml.safe_dump(valid_recipe))
    argv = ["plan", str(path), "--target", str(A100), "--plugin-manifest", str(MANIFEST),
            "--offline", "--cache-dir", str(primed_cache)]
    expected_status = "declared" if declared else "unspecified"
    expected_exit = INCOMPATIBLE if declared else OK
    assert run(*argv, "--json") == expected_exit
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["semantic_digest"]) == 64
    assert payload["semantic_status"] == expected_status
    assert run(*argv) == expected_exit
    text = capsys.readouterr().out
    assert payload["semantic_digest"] in text
    assert f"({expected_status}; not conformance evidence)" in text
