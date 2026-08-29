from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lazybrick import RecipeValidationError, load_recipe, recipe_digest

EXAMPLE = Path(__file__).parents[2] / "examples" / "qwen3-awq.yaml"


def test_digest_ignores_mapping_order(valid_recipe) -> None:
    reordered = dict(reversed(list(valid_recipe.items())))

    assert recipe_digest(valid_recipe) == recipe_digest(reordered)


def test_digest_is_stable_across_yaml_and_json(tmp_path, valid_recipe) -> None:
    as_yaml = tmp_path / "recipe.yaml"
    as_json = tmp_path / "recipe.json"
    as_yaml.write_text(yaml.safe_dump(valid_recipe), encoding="utf-8")
    as_json.write_text(json.dumps(valid_recipe), encoding="utf-8")

    assert load_recipe(as_yaml).digest == load_recipe(as_json).digest


def test_digest_changes_with_content(valid_recipe) -> None:
    before = recipe_digest(valid_recipe)
    valid_recipe["calibration"]["seed"] = 43

    assert recipe_digest(valid_recipe) != before


def test_digest_refuses_invalid_recipes(valid_recipe) -> None:
    valid_recipe.pop("export")

    with pytest.raises(RecipeValidationError):
        recipe_digest(valid_recipe)


def test_example_recipe_loads() -> None:
    document = load_recipe(EXAMPLE)

    assert document.source == EXAMPLE.resolve()
    assert document.schema_version == "0.1"
    assert len(document.digest) == 64


def test_missing_file(tmp_path) -> None:
    with pytest.raises(RecipeValidationError) as caught:
        load_recipe(tmp_path / "nope.yaml")

    assert caught.value.codes == ("missing_file",)


def test_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "recipe.txt"
    path.write_text("schema_version: '0.1'\n", encoding="utf-8")

    with pytest.raises(RecipeValidationError) as caught:
        load_recipe(path)

    assert caught.value.codes == ("unsupported_format",)


def test_unparsable_yaml(tmp_path) -> None:
    path = tmp_path / "recipe.yaml"
    path.write_text("stages: [unclosed\n", encoding="utf-8")

    with pytest.raises(RecipeValidationError) as caught:
        load_recipe(path)

    assert caught.value.codes == ("unparsable",)


def test_scalar_document_is_rejected(tmp_path) -> None:
    path = tmp_path / "recipe.yaml"
    path.write_text("just-a-string\n", encoding="utf-8")

    with pytest.raises(RecipeValidationError) as caught:
        load_recipe(path)

    assert caught.value.codes == ("invalid_type",)
