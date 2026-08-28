"""Minimal recipe loading, validation, and fingerprinting primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import yaml


class RecipeValidationError(ValueError):
    """Raised when a recipe does not satisfy the current minimal contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid LazyBrick recipe: " + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class RecipeDocument:
    """A validated recipe plus its deterministic content fingerprint."""

    data: Mapping[str, Any]
    fingerprint: str
    source: Path | None = None


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    """Validate the intentionally small v0.1 recipe envelope.

    The schema will grow only after the first end-to-end compression workflow is
    implemented. This validator checks identity and composition fields without
    pretending to validate algorithm-specific parameters.
    """

    if not isinstance(recipe, Mapping):
        raise RecipeValidationError(["the document root must be a mapping"])

    errors: list[str] = []

    if not _nonempty_string(recipe.get("schema_version")):
        errors.append("schema_version must be a non-empty string")

    model = recipe.get("model")
    if not isinstance(model, Mapping):
        errors.append("model must be a mapping")
    elif not _nonempty_string(model.get("uri")):
        errors.append("model.uri must be a non-empty string")

    stages = recipe.get("stages")
    if not isinstance(stages, list) or not stages:
        errors.append("stages must be a non-empty list")
    else:
        stage_ids: set[str] = set()
        for index, stage in enumerate(stages):
            prefix = f"stages[{index}]"
            if not isinstance(stage, Mapping):
                errors.append(f"{prefix} must be a mapping")
                continue

            stage_id = stage.get("id")
            if not _nonempty_string(stage_id):
                errors.append(f"{prefix}.id must be a non-empty string")
            elif stage_id in stage_ids:
                errors.append(f"{prefix}.id must be unique")
            else:
                stage_ids.add(stage_id)

            if not _nonempty_string(stage.get("plugin")):
                errors.append(f"{prefix}.plugin must be a non-empty string")

            parameters = stage.get("parameters")
            if parameters is not None and not isinstance(parameters, Mapping):
                errors.append(f"{prefix}.parameters must be a mapping")

    target = recipe.get("target")
    if target is not None:
        if not isinstance(target, Mapping):
            errors.append("target must be a mapping")
        else:
            for field in ("runtime", "device"):
                value = target.get(field)
                if value is not None and not _nonempty_string(value):
                    errors.append(f"target.{field} must be a non-empty string")

    if errors:
        raise RecipeValidationError(errors)


def recipe_fingerprint(recipe: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a validated recipe."""

    validate_recipe(recipe)
    try:
        canonical = json.dumps(
            recipe,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecipeValidationError(
            ["recipe values must be JSON-compatible for deterministic hashing"]
        ) from error
    return sha256(canonical).hexdigest()


def load_recipe(path: str | Path) -> RecipeDocument:
    """Load a YAML or JSON recipe, validate it, and compute its fingerprint."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RecipeValidationError([f"recipe file does not exist: {source}"])

    suffix = source.suffix.lower()
    try:
        with source.open("r", encoding="utf-8") as handle:
            if suffix == ".json":
                data = json.load(handle)
            elif suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(handle)
            else:
                raise RecipeValidationError(
                    ["recipe file must use a .json, .yaml, or .yml extension"]
                )
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise RecipeValidationError([f"recipe cannot be parsed: {error}"]) from error

    if not isinstance(data, Mapping):
        raise RecipeValidationError(["the document root must be a mapping"])

    validate_recipe(data)
    return RecipeDocument(
        data=data,
        fingerprint=recipe_fingerprint(data),
        source=source,
    )
