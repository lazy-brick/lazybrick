"""Distinct identities for authored inputs, plans, runs, attempts, and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping
from uuid import uuid4


class IdentityError(ValueError):
    """Raised when an identity input is incomplete or non-canonical."""


_ARTIFACT_KEYS = frozenset(
    {"model", "plugin", "calibration", "quantization", "export"}
)


def _validate_value(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise IdentityError(f"{path} contains a floating-point value")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentityError(f"{path} contains a non-string key")
            _validate_value(item, f"{path}.{key}")
        return
    raise IdentityError(f"{path} contains unsupported type {type(value).__name__}")


def canonical_json(value: Mapping[str, Any]) -> bytes:
    _validate_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def artifact_id(resolved_inputs: Mapping[str, Any]) -> str:
    """Hash the resolved input tuple, never the nondeterministic output bytes."""

    missing = _ARTIFACT_KEYS - set(resolved_inputs)
    unknown = set(resolved_inputs) - _ARTIFACT_KEYS
    if missing or unknown:
        raise IdentityError(
            f"artifact identity keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return sha256(canonical_json(resolved_inputs)).hexdigest()


def _digest(value: str, field: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise IdentityError(f"{field} must be a 64-character SHA-256 digest")
    return value.lower()


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: str
    attempt_id: str
    recipe_digest: str
    plan_digest: str
    artifact_id: str

    @classmethod
    def create(
        cls,
        *,
        recipe_digest: str,
        plan_digest: str,
        artifact_id: str,
        run_id: str | None = None,
    ) -> RunIdentity:
        return cls(
            run_id=run_id or f"run-{uuid4().hex}",
            attempt_id=f"attempt-{uuid4().hex}",
            recipe_digest=_digest(recipe_digest, "recipe_digest"),
            plan_digest=_digest(plan_digest, "plan_digest"),
            artifact_id=_digest(artifact_id, "artifact_id"),
        )

    def retry(self) -> RunIdentity:
        return RunIdentity(
            run_id=self.run_id,
            attempt_id=f"attempt-{uuid4().hex}",
            recipe_digest=self.recipe_digest,
            plan_digest=self.plan_digest,
            artifact_id=self.artifact_id,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "recipe_digest": self.recipe_digest,
            "plan_digest": self.plan_digest,
            "artifact_id": self.artifact_id,
        }
