"""The documented examples must behave the way the documentation says."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lazybrick import RecipeValidationError, load_recipe

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"


def test_pinned_example_validates() -> None:
    document = load_recipe(ROOT / "examples" / "qwen3-awq.yaml")

    assert document.schema_version == "0.1"


def test_pinned_example_pins_model_and_implementation() -> None:
    data = load_recipe(ROOT / "examples" / "qwen3-awq.yaml").data
    sha = re.compile(r"\A[0-9a-f]{40}\Z")

    assert sha.match(data["model"]["revision"])
    assert sha.match(data["stages"][0]["implementation"]["commit"])


def test_template_does_not_validate() -> None:
    # A template that validates is a template somebody ships by accident.
    with pytest.raises(RecipeValidationError):
        load_recipe(ROOT / "examples" / "qwen3-awq.template.yaml")


def test_every_documented_target_loads() -> None:
    from lazybrick.capabilities import HardwareProfile
    import json

    for path in (ROOT / "examples" / "targets").glob("*.json"):
        HardwareProfile.from_json(json.loads(path.read_text(encoding="utf-8")))


def test_example_plugin_manifest_loads() -> None:
    import json

    from lazybrick import PluginManifest

    path = ROOT / "examples" / "plugins" / "awq.manifest.json"
    manifest = PluginManifest.from_json(json.loads(path.read_text(encoding="utf-8")))

    assert manifest.implementation.is_pinned


@pytest.mark.parametrize("page", sorted(p.name for p in DOCS.glob("*.md")))
def test_relative_doc_links_resolve(page: str) -> None:
    """A broken link in documentation about rigour is a bad look."""

    text = (DOCS / page).read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if target.startswith(("http://", "https://")):
            continue
        assert (DOCS / target).resolve().exists(), f"{page} -> {target}"


def test_readme_links_resolve() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if target.startswith(("http://", "https://")):
            continue
        assert (ROOT / target).resolve().exists(), f"README -> {target}"
