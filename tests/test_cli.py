from pathlib import Path

from lazybrick.cli import main


def test_validate_command(capsys) -> None:
    recipe = Path(__file__).parents[1] / "examples" / "qwen3-awq.yaml"

    exit_code = main(["validate", str(recipe)])

    assert exit_code == 0
    assert "Valid LazyBrick recipe:" in capsys.readouterr().out


def test_invalid_recipe_returns_two(tmp_path, capsys) -> None:
    recipe = tmp_path / "invalid.yaml"
    recipe.write_text("schema_version: '0.1'\n", encoding="utf-8")

    exit_code = main(["validate", str(recipe)])

    assert exit_code == 2
    assert "Invalid LazyBrick recipe:" in capsys.readouterr().err
