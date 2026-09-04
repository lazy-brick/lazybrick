"""Typed public records and their stable JSON serialization.

These are the shapes that cross every LazyBrick boundary: the planner, the
subprocess plugin protocol, the artifact store, and the evidence bundle. They
are the contract, so they live in one module and change in one place.

Two rules hold for all of them:

- ``to_json`` returns plain JSON-compatible data, never a dataclass. Key order
  is irrelevant because :func:`lazybrick.canonical.canonical_json` sorts.
- ``from_json`` accepts exactly what ``to_json`` produced. Round-tripping is
  tested for every record; a field that does not survive the trip is a bug.

Optional fields are *omitted* when unset rather than serialized as ``null``, so
that adding an optional field cannot change the digest of a document that does
not use it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any

from lazybrick.canonical import digest
from lazybrick.errors import RecipeValidationError, ValidationIssue
from lazybrick.schema import is_immutable_revision

__all__ = [
    "ArtifactManifest",
    "CalibrationSpec",
    "CapabilityReport",
    "DatasetRef",
    "EvalSpec",
    "EvidenceRecord",
    "ExecutionPlan",
    "ExportSpec",
    "FileRef",
    "ImplementationRef",
    "MeasurementSeries",
    "ModelRef",
    "PluginManifest",
    "ProvenanceRecord",
    "StageSpec",
    "TargetSpec",
]

PLAN_VERSION = "0.1"

_COMMIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
_CONTAINER_DIGEST = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")


def _require(data: Mapping[str, Any], field_name: str, path: str) -> Any:
    if field_name not in data:
        raise RecipeValidationError(
            [
                ValidationIssue(
                    f"{path}.{field_name}" if path else field_name,
                    "missing_field",
                    "required field is missing",
                )
            ]
        )
    return data[field_name]


def _tuple(values: object) -> tuple[Any, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return (values,)
    return tuple(values)


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecipeValidationError(
            [ValidationIssue(path, "invalid_type", "must be an object")]
        )
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise RecipeValidationError(
            [
                ValidationIssue(
                    path,
                    "unknown_field",
                    "unknown fields: " + ", ".join(unknown),
                )
            ]
        )


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeValidationError(
            [ValidationIssue(path, "invalid_type", "must be a non-empty string")]
        )
    return value


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A base model, ideally pinned to an immutable revision."""

    uri: str
    revision: str
    trust_remote_code: bool = False

    @property
    def is_pinned(self) -> bool:
        return is_immutable_revision(self.revision)

    def to_json(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], path: str = "model") -> ModelRef:
        data = _mapping(data, path)
        _reject_unknown(data, {"uri", "revision", "trust_remote_code"}, path)
        trust_remote_code = data.get("trust_remote_code", False)
        if not isinstance(trust_remote_code, bool):
            raise RecipeValidationError(
                [
                    ValidationIssue(
                        f"{path}.trust_remote_code",
                        "invalid_type",
                        "must be a boolean",
                    )
                ]
            )
        return cls(
            uri=_text(_require(data, "uri", path), f"{path}.uri"),
            revision=_text(_require(data, "revision", path), f"{path}.revision"),
            trust_remote_code=trust_remote_code,
        )


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """A calibration or evaluation dataset."""

    uri: str
    revision: str

    @property
    def is_pinned(self) -> bool:
        return is_immutable_revision(self.revision)

    def to_json(self) -> dict[str, Any]:
        return {"uri": self.uri, "revision": self.revision}

    @classmethod
    def from_json(cls, data: Mapping[str, Any], path: str = "dataset") -> DatasetRef:
        return cls(
            uri=_require(data, "uri", path),
            revision=_require(data, "revision", path),
        )


@dataclass(frozen=True, slots=True)
class ImplementationRef:
    """Where a plugin's algorithm code actually comes from.

    Pinned by a full 40-character commit SHA or a ``sha256`` container digest.
    Abbreviated SHAs are rejected: a 7-character prefix is ambiguous, can become
    ambiguous later as the repository grows, and therefore is not immutable.
    """

    git: str | None = None
    commit: str | None = None
    container: str | None = None

    @property
    def is_pinned(self) -> bool:
        if self.container is not None:
            return bool(_CONTAINER_DIGEST.fullmatch(self.container))
        return bool(self.commit and _COMMIT_SHA.fullmatch(self.commit))

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.git is not None:
            result["git"] = self.git
        if self.commit is not None:
            result["commit"] = self.commit
        if self.container is not None:
            result["container"] = self.container
        return result

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], path: str = "implementation"
    ) -> ImplementationRef:
        data = _mapping(data, path)
        if not data:
            raise RecipeValidationError(
                [ValidationIssue(path, "missing_field", "implementation is empty")]
            )
        _reject_unknown(data, {"git", "commit", "container"}, path)
        has_git = "git" in data or "commit" in data
        has_container = "container" in data
        if has_git == has_container:
            raise RecipeValidationError(
                [
                    ValidationIssue(
                        path,
                        "invalid_value",
                        "must use exactly one of git plus commit, or container",
                    )
                ]
            )
        if has_container:
            return cls(container=_text(data.get("container"), f"{path}.container"))
        return cls(
            git=_text(data.get("git"), f"{path}.git"),
            commit=_text(data.get("commit"), f"{path}.commit"),
        )


# --------------------------------------------------------------------------
# Recipe sections
# --------------------------------------------------------------------------


def _validated_semantics(value: object) -> dict[str, str]:
    from lazybrick.semantics.profile import validate_semantics, SemanticError
    try:
        return validate_semantics(value)
    except SemanticError as error:
        raise RecipeValidationError([ValidationIssue("semantics", error.code, str(error))]) from error


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One transformation step."""

    id: str
    plugin: str
    plugin_version: str
    implementation: ImplementationRef
    parameters: Mapping[str, Any] = field(default_factory=dict)
    semantics: Mapping[str, str] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "plugin": self.plugin,
            "plugin_version": self.plugin_version,
            "implementation": self.implementation.to_json(),
            "parameters": dict(self.parameters),
            **({"semantics": _validated_semantics(self.semantics)} if self.semantics is not None else {}),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], path: str = "stage") -> StageSpec:
        return cls(
            id=_require(data, "id", path),
            plugin=_require(data, "plugin", path),
            plugin_version=_require(data, "plugin_version", path),
            implementation=ImplementationRef.from_json(
                _require(data, "implementation", path), f"{path}.implementation"
            ),
            parameters=dict(data.get("parameters") or {}),
            semantics=_validated_semantics(data["semantics"]) if "semantics" in data else None,
        )


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    dataset: DatasetRef
    samples: int
    seed: int
    preprocessing_id: str
    max_sequence_length: int

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_json(),
            "samples": self.samples,
            "seed": self.seed,
            "preprocessing_id": self.preprocessing_id,
            "max_sequence_length": self.max_sequence_length,
        }

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], path: str = "calibration"
    ) -> CalibrationSpec:
        return cls(
            dataset=DatasetRef.from_json(
                _require(data, "dataset", path), f"{path}.dataset"
            ),
            samples=_require(data, "samples", path),
            seed=_require(data, "seed", path),
            preprocessing_id=_require(data, "preprocessing_id", path),
            max_sequence_length=_require(data, "max_sequence_length", path),
        )


@dataclass(frozen=True, slots=True)
class ExportSpec:
    format: str
    runtime: str

    def to_json(self) -> dict[str, Any]:
        return {"format": self.format, "runtime": self.runtime}

    @classmethod
    def from_json(cls, data: Mapping[str, Any], path: str = "export") -> ExportSpec:
        return cls(
            format=_require(data, "format", path),
            runtime=_require(data, "runtime", path),
        )


@dataclass(frozen=True, slots=True)
class TargetSpec:
    accelerator_family: str
    min_compute_capability: str
    device_count: int
    min_memory_gib: int
    runtime: str

    def to_json(self) -> dict[str, Any]:
        return {
            "accelerator_family": self.accelerator_family,
            "min_compute_capability": self.min_compute_capability,
            "device_count": self.device_count,
            "min_memory_gib": self.min_memory_gib,
            "runtime": self.runtime,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], path: str = "target") -> TargetSpec:
        return cls(
            accelerator_family=_require(data, "accelerator_family", path),
            min_compute_capability=_require(data, "min_compute_capability", path),
            device_count=_require(data, "device_count", path),
            min_memory_gib=_require(data, "min_memory_gib", path),
            runtime=_require(data, "runtime", path),
        )


@dataclass(frozen=True, slots=True)
class EvalSpec:
    protocol_id: str
    dataset: DatasetRef
    max_samples: int
    seed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "dataset": self.dataset.to_json(),
            "max_samples": self.max_samples,
            "seed": self.seed,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], path: str = "evaluation") -> EvalSpec:
        return cls(
            protocol_id=_require(data, "protocol_id", path),
            dataset=DatasetRef.from_json(
                _require(data, "dataset", path), f"{path}.dataset"
            ),
            max_samples=_require(data, "max_samples", path),
            seed=_require(data, "seed", path),
        )


# --------------------------------------------------------------------------
# Plugin and capability records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """What a plugin declares about itself, before it is ever executed.

    ``capabilities`` uses the controlled vocabulary; the planner intersects it
    with the model, exporter, runtime, and hardware reports.
    """

    name: str
    plugin_api: str
    kind: str
    version: str
    implementation: ImplementationRef
    capabilities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    requires: Mapping[str, bool] = field(default_factory=dict)
    licenses: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "plugin_api": self.plugin_api,
            "kind": self.kind,
            "version": self.version,
            "implementation": self.implementation.to_json(),
            "capabilities": {k: list(v) for k, v in self.capabilities.items()},
            "requires": dict(self.requires),
            "licenses": dict(self.licenses),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], path: str = "plugin") -> PluginManifest:
        data = _mapping(data, path)
        _reject_unknown(
            data,
            {
                "name", "plugin_api", "kind", "version", "implementation",
                "capabilities", "requires", "licenses",
            },
            path,
        )
        capabilities = _mapping(data.get("capabilities", {}), f"{path}.capabilities")
        parsed_capabilities: dict[str, tuple[str, ...]] = {}
        for key, value in capabilities.items():
            if not isinstance(key, str) or isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise RecipeValidationError(
                    [ValidationIssue(f"{path}.capabilities", "invalid_type", "values must be string arrays")]
                )
            items = tuple(value)
            if not items or any(not isinstance(item, str) or not item for item in items):
                raise RecipeValidationError(
                    [ValidationIssue(f"{path}.capabilities.{key}", "invalid_type", "must be a non-empty string array")]
                )
            parsed_capabilities[key] = items
        requires = _mapping(data.get("requires", {}), f"{path}.requires")
        if any(not isinstance(key, str) or not isinstance(value, bool) for key, value in requires.items()):
            raise RecipeValidationError(
                [ValidationIssue(f"{path}.requires", "invalid_type", "values must be booleans")]
            )
        licenses = _mapping(data.get("licenses", {}), f"{path}.licenses")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in licenses.items()):
            raise RecipeValidationError(
                [ValidationIssue(f"{path}.licenses", "invalid_type", "values must be strings")]
            )
        return cls(
            name=_text(_require(data, "name", path), f"{path}.name"),
            plugin_api=_text(_require(data, "plugin_api", path), f"{path}.plugin_api"),
            kind=_text(_require(data, "kind", path), f"{path}.kind"),
            version=_text(_require(data, "version", path), f"{path}.version"),
            implementation=ImplementationRef.from_json(
                _require(data, "implementation", path), f"{path}.implementation"
            ),
            capabilities=parsed_capabilities,
            requires=dict(requires),
            licenses=dict(licenses),
        )


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """What one participant can do, in the controlled vocabulary.

    ``subject`` names the participant -- ``"model"``, ``"plugin:awq"``,
    ``"runtime:vllm"``, ``"hardware"`` -- so a rejection can say which side of
    the intersection was empty.
    """

    subject: str
    capabilities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def get(self, key: str) -> tuple[str, ...]:
        return tuple(self.capabilities.get(key, ()))

    def to_json(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "capabilities": {k: list(v) for k, v in self.capabilities.items()},
        }

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], path: str = "capability_report"
    ) -> CapabilityReport:
        return cls(
            subject=_require(data, "subject", path),
            capabilities={
                key: _tuple(value)
                for key, value in (data.get("capabilities") or {}).items()
            },
        )


# --------------------------------------------------------------------------
# Plan, artifact, provenance, evidence
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """A fully resolved recipe: what would be executed, and against what.

    Distinct from the authored recipe. ``recipe_digest`` identifies what the
    author wrote; :attr:`plan_digest` identifies what it resolved to; and
    :attr:`artifact_id` identifies the inputs that determine the output weights.
    All three are different questions and none substitutes for another.
    """

    recipe_digest: str
    model: ModelRef
    stages: tuple[StageSpec, ...]
    export: ExportSpec
    target: TargetSpec
    calibration: CalibrationSpec | None = None
    evaluation: EvalSpec | None = None
    plan_version: str = PLAN_VERSION

    def __post_init__(self) -> None:
        if self.plan_version not in {"0.1", "0.2"}:
            raise RecipeValidationError([ValidationIssue("plan_version", "unknown_plan_version", "unsupported plan version")])
        if self.plan_version == "0.1" and any(stage.semantics is not None for stage in self.stages):
            raise RecipeValidationError([ValidationIssue("stages", "semantic_version_mismatch", "stage semantics requires plan v0.2")])

    @property
    def semantic_digest(self) -> str | None:
        if self.plan_version == "0.1":
            return None
        return digest({"semantic_recipe_version":"1", "stages":[
            {"id":stage.id, "semantics":_validated_semantics(stage.semantics) if stage.semantics is not None else None}
            for stage in self.stages]})

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "plan_version": self.plan_version,
            "recipe_digest": self.recipe_digest,
            "model": self.model.to_json(),
            "stages": [stage.to_json() for stage in self.stages],
            "export": self.export.to_json(),
            "target": self.target.to_json(),
        }
        if self.calibration is not None:
            result["calibration"] = self.calibration.to_json()
        if self.evaluation is not None:
            result["evaluation"] = self.evaluation.to_json()
        return result

    @classmethod
    def from_json(cls, data: Mapping[str, Any], path: str = "plan") -> ExecutionPlan:
        calibration = data.get("calibration")
        evaluation = data.get("evaluation")
        return cls(
            plan_version=data.get("plan_version", PLAN_VERSION),
            recipe_digest=_require(data, "recipe_digest", path),
            model=ModelRef.from_json(_require(data, "model", path), f"{path}.model"),
            stages=tuple(
                StageSpec.from_json(stage, f"{path}.stages[{index}]")
                for index, stage in enumerate(_require(data, "stages", path))
            ),
            export=ExportSpec.from_json(
                _require(data, "export", path), f"{path}.export"
            ),
            target=TargetSpec.from_json(
                _require(data, "target", path), f"{path}.target"
            ),
            calibration=(
                CalibrationSpec.from_json(calibration) if calibration else None
            ),
            evaluation=EvalSpec.from_json(evaluation) if evaluation else None,
        )

    @classmethod
    def from_recipe(
        cls, recipe: Mapping[str, Any], recipe_digest: str
    ) -> ExecutionPlan:
        """Map a *validated* recipe onto plan records.

        This performs no resolution. References are carried across exactly as
        authored, so a plan built this way can still contain mutable revisions;
        the resolver is what makes a plan executable.
        """

        calibration = recipe.get("calibration")
        evaluation = recipe.get("evaluation")
        return cls(
            recipe_digest=recipe_digest,
            plan_version=recipe["schema_version"],
            model=ModelRef.from_json(recipe["model"]),
            stages=tuple(
                StageSpec.from_json(stage, f"stages[{index}]")
                for index, stage in enumerate(recipe["stages"])
            ),
            export=ExportSpec.from_json(recipe["export"]),
            target=TargetSpec.from_json(recipe["target"]),
            calibration=CalibrationSpec.from_json(calibration) if calibration else None,
            evaluation=EvalSpec.from_json(evaluation) if evaluation else None,
        )

    @property
    def plan_digest(self) -> str:
        """Identity of the resolved plan, including the hardware target."""

        return digest(self.to_json())

    @property
    def artifact_id(self) -> str:
        """Identity of the inputs that determine the produced weights.

        Deliberately excludes :attr:`target` and :attr:`evaluation`. Measuring a
        artifact on different hardware, or against a different eval protocol,
        does not make it a different artifact.

        This identifies *inputs*, not output bytes. GPU quantization is not
        bit-reproducible, so two runs sharing an ``artifact_id`` are expected to
        agree within tolerance, never byte-for-byte.
        """

        payload: dict[str, Any] = {
            "artifact_id_version": self.plan_version,
            "model": self.model.to_json(),
            "stages": [stage.to_json() for stage in self.stages],
            "export": self.export.to_json(),
        }
        if self.calibration is not None:
            payload["calibration"] = self.calibration.to_json()
        return digest(payload)


@dataclass(frozen=True, slots=True)
class FileRef:
    """One file in an artifact, with the hash used to verify it before loading."""

    path: str
    size_bytes: int
    sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], path: str = "file") -> FileRef:
        return cls(
            path=_require(data, "path", path),
            size_bytes=_require(data, "size_bytes", path),
            sha256=_require(data, "sha256", path),
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """A produced artifact and the identities it descends from."""

    artifact_id: str
    plan_digest: str
    recipe_digest: str
    export: ExportSpec
    files: tuple[FileRef, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "plan_digest": self.plan_digest,
            "recipe_digest": self.recipe_digest,
            "export": self.export.to_json(),
            "files": [file.to_json() for file in self.files],
        }

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], path: str = "artifact"
    ) -> ArtifactManifest:
        return cls(
            artifact_id=_require(data, "artifact_id", path),
            plan_digest=_require(data, "plan_digest", path),
            recipe_digest=_require(data, "recipe_digest", path),
            export=ExportSpec.from_json(
                _require(data, "export", path), f"{path}.export"
            ),
            files=tuple(
                FileRef.from_json(entry, f"{path}.files[{index}]")
                for index, entry in enumerate(data.get("files") or ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Everything needed to explain how an artifact came to exist.

    ``environment`` and ``versions`` are open mappings on purpose: the set of
    things worth recording grows, and a closed schema would force a version bump
    every time a new dependency matters. ``environment`` must already be
    redacted by the writer -- this record does not redact anything itself.
    """

    run_id: str
    attempt_id: str
    recipe_digest: str
    plan_digest: str
    status: str
    artifact_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    versions: Mapping[str, str] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    commands: tuple[Mapping[str, Any], ...] = ()
    seeds: Mapping[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "recipe_digest": self.recipe_digest,
            "plan_digest": self.plan_digest,
            "status": self.status,
            "versions": dict(self.versions),
            "environment": dict(self.environment),
            "commands": [dict(command) for command in self.commands],
            "seeds": dict(self.seeds),
        }
        for name in ("artifact_id", "started_at", "finished_at"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], path: str = "provenance"
    ) -> ProvenanceRecord:
        return cls(
            run_id=_require(data, "run_id", path),
            attempt_id=_require(data, "attempt_id", path),
            recipe_digest=_require(data, "recipe_digest", path),
            plan_digest=_require(data, "plan_digest", path),
            status=_require(data, "status", path),
            artifact_id=data.get("artifact_id"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            versions=dict(data.get("versions") or {}),
            environment=dict(data.get("environment") or {}),
            commands=tuple(dict(c) for c in (data.get("commands") or ())),
            seeds=dict(data.get("seeds") or {}),
        )


@dataclass(frozen=True, slots=True)
class MeasurementSeries:
    """Raw samples for one measured quantity, plus its unit.

    Samples are decimal *strings*, not floats. Retaining them is required: an
    average alone cannot show variance, and variance is what decides whether a
    regression gate is meaningful yet.
    """

    name: str
    unit: str
    samples: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "unit": self.unit, "samples": list(self.samples)}

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], path: str = "measurement"
    ) -> MeasurementSeries:
        return cls(
            name=_require(data, "name", path),
            unit=_require(data, "unit", path),
            samples=tuple(str(s) for s in (data.get("samples") or ())),
        )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """A baseline-versus-compressed comparison under one named protocol.

    Scores are decimal strings for the same reason digests forbid floats: the
    printed form is the measurement. ``8.42`` and ``8.4200000000001`` are
    different claims, and a float round-trip must not be allowed to blur them.

    There is no pass/fail field. A regression gate is only meaningful once
    repeat runs have established the expected variance, and that has not
    happened yet.
    """

    protocol_id: str
    metric: str
    baseline_score: str
    compressed_score: str
    absolute_delta: str
    relative_delta: str
    measurements: tuple[MeasurementSeries, ...] = ()
    notes: str | None = None

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "protocol_id": self.protocol_id,
            "metric": self.metric,
            "baseline_score": self.baseline_score,
            "compressed_score": self.compressed_score,
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "measurements": [m.to_json() for m in self.measurements],
        }
        if self.notes is not None:
            result["notes"] = self.notes
        return result

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], path: str = "evidence"
    ) -> EvidenceRecord:
        return cls(
            protocol_id=_require(data, "protocol_id", path),
            metric=_require(data, "metric", path),
            baseline_score=_require(data, "baseline_score", path),
            compressed_score=_require(data, "compressed_score", path),
            absolute_delta=_require(data, "absolute_delta", path),
            relative_delta=_require(data, "relative_delta", path),
            measurements=tuple(
                MeasurementSeries.from_json(m, f"{path}.measurements[{index}]")
                for index, m in enumerate(data.get("measurements") or ())
            ),
            notes=data.get("notes"),
        )
