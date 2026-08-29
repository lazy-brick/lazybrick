from __future__ import annotations

from pathlib import Path

import pytest

from lazybrick import (
    RecipeValidationError,
    load_recipe,
    recipe_fingerprint,
    validate_recipe,
)


def valid_recipe() -> dict:
    return {
        "schema_version": "0.1",
        "model": {"uri": "hf://Qwen/Qwen3-4B", "revision": "abc123"},
        "stages": [
            {
                "id": "quantize",
                "plugin": "lazybrick.plugin/awq",
                "parameters": {"weight_bits": 4},
            }
        ],
        "target": {"runtime": "vllm", "device": "cuda:0"},
    }


def test_valid_recipe_passes() -> None:
    validate_recipe(valid_recipe())


def test_missing_stages_fails() -> None:
    recipe = valid_recipe()
    recipe.pop("stages")

    with pytest.raises(RecipeValidationError, match="stages"):
        validate_recipe(recipe)


def test_duplicate_stage_ids_fail() -> None:
    recipe = valid_recipe()
    recipe["stages"].append(dict(recipe["stages"][0]))

    with pytest.raises(RecipeValidationError, match="unique"):
        validate_recipe(recipe)


def test_fingerprint_ignores_mapping_order() -> None:
    recipe = valid_recipe()
    reordered = {
        "target": recipe["target"],
        "stages": recipe["stages"],
        "model": recipe["model"],
        "schema_version": recipe["schema_version"],
    }

    assert recipe_fingerprint(recipe) == recipe_fingerprint(reordered)


def test_example_recipe_loads() -> None:
    example = Path(__file__).parents[1] / "examples" / "qwen3-awq.yaml"
    document = load_recipe(example)

    assert document.source == example.resolve()
    assert len(document.fingerprint) == 64
