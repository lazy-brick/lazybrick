from __future__ import annotations

from pathlib import Path

import yaml

from lazybrick.cli import main

EXAMPLE = Path(__file__).parents[2] / "examples" / "qwen3-awq.yaml"


def test_validate_command(capsys) -> None:
    exit_code = main(["validate", str(EXAMPLE)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Valid LazyBrick recipe (schema v0.1)" in out
    assert "recipe_digest:" in out


def test_digest_command_prints_only_the_digest(capsys) -> None:
    exit_code = main(["digest", str(EXAMPLE)])

    out = capsys.readouterr().out.strip()
    assert exit_code == 0
    assert len(out) == 64


def test_invalid_recipe_returns_two(tmp_path, capsys, valid_recipe) -> None:
    valid_recipe["model"].pop("uri")
    recipe = tmp_path / "invalid.yaml"
    recipe.write_text(yaml.safe_dump(valid_recipe), encoding="utf-8")

    exit_code = main(["validate", str(recipe)])

    err = capsys.readouterr().err
    assert exit_code == 2
    assert "Invalid LazyBrick recipe:" in err
    assert "model.uri: required field is missing [missing_field]" in err


def test_no_command_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "usage: lazybrick" in capsys.readouterr().out
