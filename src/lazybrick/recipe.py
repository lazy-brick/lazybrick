"""Recipe loading, validation, and content digesting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from lazybrick.canonical import digest as canonical_digest
from lazybrick.errors import (
    CanonicalizationError,
    RecipeValidationError,
    ValidationIssue,
)
from lazybrick.schema import SCHEMA_VERSION, validate_document

__all__ = [
    "RecipeDocument",
    "load_recipe",
    "recipe_digest",
    "validate_recipe",
]


@dataclass(frozen=True, slots=True)
class RecipeDocument:
    """A validated recipe plus the digest of its authored content."""

    data: Mapping[str, Any]
    digest: str
    source: Path | None = None

    @property
    def schema_version(self) -> str:
        return str(self.data["schema_version"])


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    """Validate a recipe against the v0.1 schema.

    Raises :class:`RecipeValidationError` carrying *every* issue found, each with
    a field path and a stable reason code.
    """

    issues: tuple[ValidationIssue, ...] = validate_document(recipe)
    if issues:
        raise RecipeValidationError(issues)


def recipe_digest(recipe: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a validated recipe's authored content.

    This is ``recipe_digest``: the identity of *what the author wrote*. It is
    not an artifact identity and it does not prove reproducibility. Two recipes
    with the same digest were authored identically; they can still resolve to
    different weights if either references a mutable revision.
    """

    validate_recipe(recipe)
    try:
        return canonical_digest(recipe)
    except CanonicalizationError as error:
        # Re-raised as a recipe error so callers handle one exception type.
        raise RecipeValidationError(error.issues) from error


def load_recipe(path: str | Path) -> RecipeDocument:
    """Load a YAML or JSON recipe, validate it, and compute its digest."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RecipeValidationError(
            [ValidationIssue("", "missing_file", f"recipe file does not exist: {source}")]
        )

    suffix = source.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise RecipeValidationError(
            [
                ValidationIssue(
                    "",
                    "unsupported_format",
                    "recipe file must use a .json, .yaml, or .yml extension",
                )
            ]
        )

    try:
        with source.open("r", encoding="utf-8") as handle:
            if suffix == ".json":
                data = json.load(handle)
            else:
                data = yaml.safe_load(handle)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise RecipeValidationError(
            [ValidationIssue("", "unparsable", f"recipe cannot be parsed: {error}")]
        ) from error

    if not isinstance(data, Mapping):
        raise RecipeValidationError(
            [ValidationIssue("", "invalid_type", "the document root must be a mapping")]
        )

    return RecipeDocument(data=data, digest=recipe_digest(data), source=source)
