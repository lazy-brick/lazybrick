"""Public API for LazyBrick."""

from lazybrick.__about__ import __version__
from lazybrick.errors import RecipeValidationError, ValidationIssue
from lazybrick.recipe import (
    RecipeDocument,
    load_recipe,
    recipe_digest,
    validate_recipe,
)
from lazybrick.schema import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    is_immutable_revision,
)

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "RecipeDocument",
    "RecipeValidationError",
    "ValidationIssue",
    "__version__",
    "is_immutable_revision",
    "load_recipe",
    "recipe_digest",
    "validate_recipe",
]
