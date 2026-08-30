"""Structured validation errors.

Every rejection carries a field path and a stable reason code so that machine
consumers (the ``--json`` CLI output, and the planner) can react to *why* a
document was rejected without parsing English.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One field-level problem with an authored document."""

    path: str
    code: str
    message: str

    def __str__(self) -> str:
        location = self.path or "<document>"
        return f"{location}: {self.message} [{self.code}]"

    def to_json(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "message": self.message}


class RecipeValidationError(ValueError):
    """Raised when a recipe does not satisfy the versioned schema."""

    def __init__(self, issues: Iterable[ValidationIssue | str]) -> None:
        normalized: list[ValidationIssue] = []
        for issue in issues:
            if isinstance(issue, str):
                normalized.append(ValidationIssue("", "invalid_recipe", issue))
            else:
                normalized.append(issue)

        self.issues: tuple[ValidationIssue, ...] = tuple(normalized)
        super().__init__(
            "Invalid LazyBrick recipe: "
            + "; ".join(str(issue) for issue in self.issues)
        )

    @property
    def errors(self) -> tuple[str, ...]:
        """Human-readable rendering of every issue, in report order."""

        return tuple(str(issue) for issue in self.issues)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def to_json(self) -> list[dict[str, str]]:
        return [issue.to_json() for issue in self.issues]


class _IssueCollector:
    """Accumulates issues so one pass reports every problem, not just the first."""

    def __init__(self) -> None:
        self._issues: list[ValidationIssue] = []

    def add(self, path: str, code: str, message: str) -> None:
        self._issues.append(ValidationIssue(path, code, message))

    def extend(self, issues: Sequence[ValidationIssue]) -> None:
        self._issues.extend(issues)

    def __bool__(self) -> bool:
        return bool(self._issues)

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        return tuple(self._issues)

    def raise_if_any(self) -> None:
        if self._issues:
            raise RecipeValidationError(self._issues)
