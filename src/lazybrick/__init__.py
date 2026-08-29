"""Public API for LazyBrick."""

from lazybrick.__about__ import __version__
from lazybrick.canonical import canonical_json, digest, ensure_canonical
from lazybrick.capabilities import (
    CompatibilityResult,
    HardwareProfile,
    Reason,
    check_compatibility,
)
from lazybrick.errors import (
    CanonicalizationError,
    RecipeValidationError,
    ValidationError,
    ValidationIssue,
)
from lazybrick.recipe import (
    RecipeDocument,
    load_recipe,
    recipe_digest,
    validate_recipe,
)
from lazybrick.records import (
    ArtifactManifest,
    CalibrationSpec,
    CapabilityReport,
    DatasetRef,
    EvalSpec,
    EvidenceRecord,
    ExecutionPlan,
    ExportSpec,
    FileRef,
    ImplementationRef,
    MeasurementSeries,
    ModelRef,
    PluginManifest,
    ProvenanceRecord,
    StageSpec,
    TargetSpec,
)
from lazybrick.resolve import (
    ResolutionError,
    ResolvedDataset,
    ResolvedModel,
    ResolvedRecipe,
    Resolver,
    ResolverCache,
)
from lazybrick.schema import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    is_immutable_revision,
)

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ArtifactManifest",
    "CalibrationSpec",
    "CanonicalizationError",
    "CapabilityReport",
    "CompatibilityResult",
    "DatasetRef",
    "EvalSpec",
    "EvidenceRecord",
    "ExecutionPlan",
    "ExportSpec",
    "FileRef",
    "HardwareProfile",
    "ImplementationRef",
    "MeasurementSeries",
    "ModelRef",
    "PluginManifest",
    "ProvenanceRecord",
    "Reason",
    "RecipeDocument",
    "RecipeValidationError",
    "ResolutionError",
    "ResolvedDataset",
    "ResolvedModel",
    "ResolvedRecipe",
    "Resolver",
    "ResolverCache",
    "StageSpec",
    "TargetSpec",
    "ValidationError",
    "ValidationIssue",
    "__version__",
    "canonical_json",
    "check_compatibility",
    "digest",
    "ensure_canonical",
    "is_immutable_revision",
    "load_recipe",
    "recipe_digest",
    "validate_recipe",
]
