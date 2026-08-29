"""Canonical JSON serialization and content digests.

Every LazyBrick identity -- ``recipe_digest``, ``plan_digest``, ``artifact_id``
-- is the SHA-256 of a canonical byte string. "Canonical" here means one value
has exactly one encoding:

- object keys are sorted by Unicode code point
- no insignificant whitespace
- UTF-8, not escaped ASCII, so the bytes match what a human sees
- **no floating-point numbers at all**

The last rule is the load-bearing one. ``128`` and ``128.0`` are the same
number and different bytes, ``0.1`` is not representable, and repr rules differ
between languages. A digest that changes because an author typed a trailing
``.0`` is not an identity. Every quantity LazyBrick pins -- weight bits, group
size, sample counts, seeds, sequence lengths, device counts -- is a whole
number, and the one genuinely fractional-looking field, compute capability, is
carried as the string ``"8.0"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from typing import Any

from lazybrick.errors import CanonicalizationError, ValidationIssue

__all__ = ["canonical_json", "digest", "ensure_canonical"]


def _describe(value: object) -> str:
    return type(value).__name__


def _walk(value: object, path: str, issues: list[ValidationIssue]) -> None:
    if value is None or isinstance(value, (str, bool)):
        return

    if isinstance(value, float):
        issues.append(
            ValidationIssue(
                path,
                "float_not_allowed",
                f"floating-point values are not canonical; got {value!r}. Use an "
                "integer, or a string if the value is genuinely fractional",
            )
        )
        return

    if isinstance(value, int):
        return

    if isinstance(value, Mapping):
        for key in value:
            child = f"{path}.{key}" if path else str(key)
            if not isinstance(key, str):
                issues.append(
                    ValidationIssue(
                        path,
                        "invalid_key",
                        f"object keys must be strings; got {_describe(key)}",
                    )
                )
                continue
            _walk(value[key], child, issues)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _walk(item, f"{path}[{index}]", issues)
        return

    issues.append(
        ValidationIssue(
            path,
            "not_serializable",
            f"{_describe(value)} has no canonical JSON representation",
        )
    )


def ensure_canonical(value: Any) -> None:
    """Raise :class:`CanonicalizationError` if ``value`` cannot be canonicalized."""

    issues: list[ValidationIssue] = []
    _walk(value, "", issues)
    if issues:
        raise CanonicalizationError(issues)


def canonical_json(value: Any) -> bytes:
    """Return the one canonical UTF-8 encoding of ``value``."""

    ensure_canonical(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """Return the SHA-256 hex digest of ``value``'s canonical encoding."""

    return sha256(canonical_json(value)).hexdigest()
