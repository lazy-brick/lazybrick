"""Stable errors exposed by the LazyBrick plugin boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PluginFailure:
    """A serializable plugin failure with a stable machine-readable code."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class PluginError(RuntimeError):
    """Raised when discovery, validation, or plugin execution fails."""

    def __init__(self, failure: PluginFailure) -> None:
        self.failure = failure
        super().__init__(f"{failure.code}: {failure.message}")
