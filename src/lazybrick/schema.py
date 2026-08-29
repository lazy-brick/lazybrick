"""The versioned LazyBrick recipe schema, v0.1.

This module answers exactly one question: *is this document shaped like a v0.1
recipe?* It deliberately does not answer whether the recipe is **runnable**.
Whether a runtime, accelerator, export format, or quantization scheme is
actually supported is a capability question, resolved later against plugin and
hardware manifests. Keeping the two apart stops the supported vocabulary from
being duplicated in two places that can disagree.

Consequently the schema checks presence, type, and shape, and it rejects
unknown fields. It does not enumerate legal runtimes.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Final

from lazybrick.errors import _IssueCollector, ValidationIssue

SCHEMA_VERSION: Final = "0.1"
SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({SCHEMA_VERSION})

_COMMIT_SHA: Final = re.compile(r"\A[0-9a-f]{40}\Z")
_CONTAINER_DIGEST: Final = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")
_MODEL_URI: Final = re.compile(r"\A(?P<scheme>[a-z][a-z0-9+.-]*)://(?P<rest>\S+)\Z")

#: URI schemes the v0.1 resolver knows how to pin. Anything else is refused at
#: authoring time, because an unrecognised scheme can never be made immutable.
KNOWN_MODEL_SCHEMES: Final = frozenset({"hf"})
KNOWN_DATASET_SCHEMES: Final = frozenset({"hf-dataset"})

_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "model",
        "stages",
        "calibration",
        "export",
        "target",
        "evaluation",
    }
)
_REQUIRED_TOP_LEVEL: Final = ("schema_version", "model", "stages", "export", "target")


def is_immutable_revision(revision: object) -> bool:
    """Return True for a revision that cannot move under us.

    Only a full 40-character Git commit SHA qualifies. Branches, tags, and
    ``main`` are all mutable, and pinning them proves nothing.
    """

    return isinstance(revision, str) and bool(_COMMIT_SHA.match(revision))


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_int(value: object) -> bool:
    # bool is a subclass of int; a flag is never a count.
    return isinstance(value, int) and not isinstance(value, bool)


def _check_unknown(
    issues: _IssueCollector, path: str, mapping: Mapping[str, Any], allowed: frozenset[str]
) -> None:
    for key in sorted(k for k in mapping if k not in allowed):
        issues.add(
            _join(path, str(key)),
            "unknown_field",
            f"unknown field; v{SCHEMA_VERSION} accepts "
            + ", ".join(sorted(allowed)),
        )


def _join(prefix: str, field: str) -> str:
    return f"{prefix}.{field}" if prefix else field


def _text_field(
    issues: _IssueCollector, path: str, mapping: Mapping[str, Any], field: str
) -> str | None:
    value = mapping.get(field)
    target = _join(path, field)
    if field not in mapping:
        issues.add(target, "missing_field", "required field is missing")
        return None
    if not _is_text(value):
        issues.add(target, "invalid_type", "must be a non-empty string")
        return None
    return value


def _int_field(
    issues: _IssueCollector,
    path: str,
    mapping: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
) -> int | None:
    value = mapping.get(field)
    target = _join(path, field)
    if field not in mapping:
        issues.add(target, "missing_field", "required field is missing")
        return None
    if not _is_int(value):
        issues.add(
            target,
            "invalid_type",
            "must be an integer; fractional values are not representable in a "
            "canonical recipe",
        )
        return None
    if value < minimum:
        issues.add(target, "invalid_value", f"must be >= {minimum}")
        return None
    return value


def _submapping(
    issues: _IssueCollector,
    path: str,
    mapping: Mapping[str, Any],
    field: str,
    *,
    required: bool,
) -> Mapping[str, Any] | None:
    target = _join(path, field)
    if field not in mapping:
        if required:
            issues.add(target, "missing_field", "required field is missing")
        return None
    value = mapping.get(field)
    if not isinstance(value, Mapping):
        issues.add(target, "invalid_type", "must be a mapping")
        return None
    return value


def _uri_field(
    issues: _IssueCollector,
    path: str,
    mapping: Mapping[str, Any],
    field: str,
    *,
    schemes: frozenset[str],
) -> None:
    value = _text_field(issues, path, mapping, field)
    if value is None:
        return
    target = _join(path, field)
    match = _MODEL_URI.match(value)
    if match is None:
        issues.add(
            target,
            "invalid_value",
            "must be a '<scheme>://<path>' URI",
        )
        return
    scheme = match.group("scheme")
    if scheme not in schemes:
        issues.add(
            target,
            "unsupported_scheme",
            f"unsupported URI scheme '{scheme}'; v{SCHEMA_VERSION} resolves "
            + ", ".join(f"{s}://" for s in sorted(schemes)),
        )


_REFERENCE_FIELDS: Final = frozenset({"uri", "revision"})


def _reference(
    issues: _IssueCollector,
    path: str,
    reference: Mapping[str, Any],
    *,
    schemes: frozenset[str],
) -> None:
    """Validate a ``{uri, revision}`` pair.

    Immutability is *not* enforced here. An author may write a branch name and
    let the resolver pin it; execution is refused later if it is still mutable.
    """

    _check_unknown(issues, path, reference, _REFERENCE_FIELDS)
    _uri_field(issues, path, reference, "uri", schemes=schemes)
    _text_field(issues, path, reference, "revision")


_MODEL_FIELDS: Final = frozenset({"uri", "revision", "trust_remote_code"})


def _model(issues: _IssueCollector, model: Mapping[str, Any]) -> None:
    _check_unknown(issues, "model", model, _MODEL_FIELDS)
    _uri_field(issues, "model", model, "uri", schemes=KNOWN_MODEL_SCHEMES)
    _text_field(issues, "model", model, "revision")

    if "trust_remote_code" in model and not isinstance(
        model["trust_remote_code"], bool
    ):
        issues.add("model.trust_remote_code", "invalid_type", "must be a boolean")


_IMPLEMENTATION_FIELDS: Final = frozenset({"git", "commit", "container"})


def _implementation(issues: _IssueCollector, path: str, impl: Mapping[str, Any]) -> None:
    _check_unknown(issues, path, impl, _IMPLEMENTATION_FIELDS)

    has_git = "git" in impl or "commit" in impl
    has_container = "container" in impl

    if not has_git and not has_container:
        issues.add(
            path,
            "missing_field",
            "must pin the implementation with either 'git' plus 'commit', or "
            "'container'",
        )
        return

    if has_git:
        _text_field(issues, path, impl, "git")
        commit = _text_field(issues, path, impl, "commit")
        if commit is not None and not _COMMIT_SHA.match(commit):
            issues.add(
                _join(path, "commit"),
                "mutable_reference",
                "must be a full 40-character commit SHA; branches and tags move",
            )

    if has_container:
        container = impl.get("container")
        if not _is_text(container):
            issues.add(_join(path, "container"), "invalid_type", "must be a non-empty string")
        elif not _CONTAINER_DIGEST.match(container):
            issues.add(
                _join(path, "container"),
                "mutable_reference",
                "must be pinned by digest, as 'image@sha256:<64 hex characters>'",
            )


_STAGE_FIELDS: Final = frozenset(
    {"id", "plugin", "plugin_version", "implementation", "parameters"}
)


def _stages(issues: _IssueCollector, stages: object) -> None:
    if not isinstance(stages, list) or not stages:
        issues.add("stages", "invalid_type", "stages must be a non-empty list")
        return

    seen: set[str] = set()
    for index, stage in enumerate(stages):
        path = f"stages[{index}]"
        if not isinstance(stage, Mapping):
            issues.add(path, "invalid_type", "must be a mapping")
            continue

        _check_unknown(issues, path, stage, _STAGE_FIELDS)

        stage_id = _text_field(issues, path, stage, "id")
        if stage_id is not None:
            if stage_id in seen:
                issues.add(
                    _join(path, "id"),
                    "duplicate_id",
                    f"stage id '{stage_id}' must be unique within the recipe",
                )
            seen.add(stage_id)

        _text_field(issues, path, stage, "plugin")
        _text_field(issues, path, stage, "plugin_version")

        implementation = _submapping(issues, path, stage, "implementation", required=True)
        if implementation is not None:
            _implementation(issues, _join(path, "implementation"), implementation)

        if "parameters" in stage and not isinstance(stage["parameters"], Mapping):
            issues.add(_join(path, "parameters"), "invalid_type", "must be a mapping")


_CALIBRATION_FIELDS: Final = frozenset(
    {"dataset", "samples", "seed", "preprocessing_id", "max_sequence_length"}
)


def _calibration(issues: _IssueCollector, calibration: Mapping[str, Any]) -> None:
    path = "calibration"
    _check_unknown(issues, path, calibration, _CALIBRATION_FIELDS)

    dataset = _submapping(issues, path, calibration, "dataset", required=True)
    if dataset is not None:
        _reference(
            issues, _join(path, "dataset"), dataset, schemes=KNOWN_DATASET_SCHEMES
        )

    _int_field(issues, path, calibration, "samples", minimum=1)
    _int_field(issues, path, calibration, "seed", minimum=0)
    _text_field(issues, path, calibration, "preprocessing_id")
    _int_field(issues, path, calibration, "max_sequence_length", minimum=1)


_EXPORT_FIELDS: Final = frozenset({"format", "runtime"})


def _export(issues: _IssueCollector, export: Mapping[str, Any]) -> None:
    _check_unknown(issues, "export", export, _EXPORT_FIELDS)
    _text_field(issues, "export", export, "format")
    _text_field(issues, "export", export, "runtime")


_TARGET_FIELDS: Final = frozenset(
    {
        "accelerator_family",
        "min_compute_capability",
        "device_count",
        "min_memory_gib",
        "runtime",
    }
)
_COMPUTE_CAPABILITY: Final = re.compile(r"\A[0-9]+\.[0-9]+\Z")


def _target(issues: _IssueCollector, target: Mapping[str, Any]) -> None:
    _check_unknown(issues, "target", target, _TARGET_FIELDS)
    _text_field(issues, "target", target, "accelerator_family")
    _text_field(issues, "target", target, "runtime")
    _int_field(issues, "target", target, "device_count", minimum=1)
    _int_field(issues, "target", target, "min_memory_gib", minimum=1)

    capability = _text_field(issues, "target", target, "min_compute_capability")
    if capability is not None and not _COMPUTE_CAPABILITY.match(capability):
        issues.add(
            "target.min_compute_capability",
            "invalid_value",
            "must be a 'major.minor' string such as '8.0'; it is a string, not a "
            "number, because 8.0 and 8.00 must not compare equal",
        )


_EVALUATION_FIELDS: Final = frozenset({"protocol_id", "dataset", "max_samples", "seed"})


def _evaluation(issues: _IssueCollector, evaluation: Mapping[str, Any]) -> None:
    path = "evaluation"
    _check_unknown(issues, path, evaluation, _EVALUATION_FIELDS)

    _text_field(issues, path, evaluation, "protocol_id")
    dataset = _submapping(issues, path, evaluation, "dataset", required=True)
    if dataset is not None:
        _reference(
            issues, _join(path, "dataset"), dataset, schemes=KNOWN_DATASET_SCHEMES
        )
    _int_field(issues, path, evaluation, "max_samples", minimum=1)
    _int_field(issues, path, evaluation, "seed", minimum=0)


def validate_document(recipe: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
    """Return every schema issue in ``recipe``; empty means valid."""

    issues = _IssueCollector()

    if not isinstance(recipe, Mapping):
        issues.add("", "invalid_type", "the document root must be a mapping")
        return issues.issues

    version = recipe.get("schema_version")
    if "schema_version" not in recipe:
        issues.add("schema_version", "missing_field", "required field is missing")
        return issues.issues
    if not _is_text(version):
        issues.add("schema_version", "invalid_type", "must be a non-empty string")
        return issues.issues
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        # Do not validate the body against a schema it was not written for.
        issues.add(
            "schema_version",
            "unknown_schema_version",
            f"unsupported schema version '{version}'; this build understands "
            + ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS)),
        )
        return issues.issues

    _check_unknown(issues, "", recipe, _TOP_LEVEL_FIELDS)

    for field in _REQUIRED_TOP_LEVEL:
        if field not in recipe:
            issues.add(field, "missing_field", "required field is missing")

    model = _submapping(issues, "", recipe, "model", required=False)
    if model is not None:
        _model(issues, model)

    if "stages" in recipe:
        _stages(issues, recipe["stages"])

    export = _submapping(issues, "", recipe, "export", required=False)
    if export is not None:
        _export(issues, export)

    target = _submapping(issues, "", recipe, "target", required=False)
    if target is not None:
        _target(issues, target)

    calibration = _submapping(issues, "", recipe, "calibration", required=False)
    if calibration is not None:
        _calibration(issues, calibration)

    evaluation = _submapping(issues, "", recipe, "evaluation", required=False)
    if evaluation is not None:
        _evaluation(issues, evaluation)

    return issues.issues
