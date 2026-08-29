"""Explicit run-attempt state transitions and stable failure semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RunState(str, Enum):
    CREATED = "CREATED"
    RESOLVING = "RESOLVING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    RUNNING = "RUNNING"
    EXPORTING = "EXPORTING"
    EVALUATING = "EVALUATING"
    SUCCEEDED = "SUCCEEDED"
    INCOMPATIBLE = "INCOMPATIBLE"
    BUILD_FAILED = "BUILD_FAILED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXPORT_FAILED = "EXPORT_FAILED"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset(
    {
        RunState.SUCCEEDED,
        RunState.INCOMPATIBLE,
        RunState.BUILD_FAILED,
        RunState.EXECUTION_FAILED,
        RunState.EXPORT_FAILED,
        RunState.EVALUATION_FAILED,
        RunState.CANCELLED,
    }
)
FAILURE_STATES = TERMINAL_STATES - {RunState.SUCCEEDED}

_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.RESOLVING, RunState.CANCELLED}),
    RunState.RESOLVING: frozenset(
        {RunState.VALIDATING, RunState.EXECUTION_FAILED, RunState.CANCELLED}
    ),
    RunState.VALIDATING: frozenset(
        {
            RunState.READY,
            RunState.INCOMPATIBLE,
            RunState.EXECUTION_FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.READY: frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.RUNNING: frozenset(
        {
            RunState.EXPORTING,
            RunState.BUILD_FAILED,
            RunState.EXECUTION_FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.EXPORTING: frozenset(
        {RunState.EVALUATING, RunState.EXPORT_FAILED, RunState.CANCELLED}
    ),
    RunState.EVALUATING: frozenset(
        {RunState.SUCCEEDED, RunState.EVALUATION_FAILED, RunState.CANCELLED}
    ),
}


class InvalidStateTransition(RuntimeError):
    """Raised before any status record is changed by an illegal transition."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class StateEvent:
    sequence: int
    previous: RunState | None
    state: RunState
    timestamp: str
    reason_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sequence": self.sequence,
            "previous": self.previous.value if self.previous else None,
            "state": self.state.value,
            "timestamp": self.timestamp,
        }
        if self.reason_code is not None:
            result["reason_code"] = self.reason_code
        if self.message is not None:
            result["message"] = self.message
        return result


class RunStateMachine:
    def __init__(self, *, created_at: str | None = None) -> None:
        self._state = RunState.CREATED
        self._events = [
            StateEvent(
                sequence=0,
                previous=None,
                state=RunState.CREATED,
                timestamp=created_at or _timestamp(),
            )
        ]

    @property
    def state(self) -> RunState:
        return self._state

    @property
    def terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    @property
    def events(self) -> tuple[StateEvent, ...]:
        return tuple(self._events)

    def transition(
        self,
        next_state: RunState | str,
        *,
        reason_code: str | None = None,
        message: str | None = None,
        timestamp: str | None = None,
    ) -> StateEvent:
        try:
            target = RunState(next_state)
        except ValueError as error:
            raise InvalidStateTransition(f"unknown run state: {next_state}") from error
        allowed = _TRANSITIONS.get(self._state, frozenset())
        if target not in allowed:
            raise InvalidStateTransition(
                f"cannot transition from {self._state.value} to {target.value}"
            )
        if target in FAILURE_STATES:
            if not isinstance(reason_code, str) or not reason_code:
                raise InvalidStateTransition(
                    f"{target.value} requires a non-empty reason_code"
                )
            if not isinstance(message, str) or not message:
                raise InvalidStateTransition(f"{target.value} requires a non-empty message")
        elif reason_code is not None or message is not None:
            raise InvalidStateTransition("non-failure transitions must not carry failure details")
        event = StateEvent(
            sequence=len(self._events),
            previous=self._state,
            state=target,
            timestamp=timestamp or _timestamp(),
            reason_code=reason_code,
            message=message,
        )
        self._state = target
        self._events.append(event)
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "terminal": self.terminal,
            "history": [event.to_dict() for event in self._events],
        }

    @classmethod
    def from_dict(cls, value: object) -> RunStateMachine:
        if not isinstance(value, dict) or not isinstance(value.get("history"), list):
            raise InvalidStateTransition("state record must contain a history list")
        history = value["history"]
        if not history or not isinstance(history[0], dict):
            raise InvalidStateTransition("state history must begin with CREATED")
        if history[0].get("state") != RunState.CREATED.value:
            raise InvalidStateTransition("state history must begin with CREATED")
        machine = cls(created_at=history[0].get("timestamp"))
        for item in history[1:]:
            if not isinstance(item, dict):
                raise InvalidStateTransition("state history entries must be objects")
            machine.transition(
                item.get("state"),
                reason_code=item.get("reason_code"),
                message=item.get("message"),
                timestamp=item.get("timestamp"),
            )
        if value.get("state") != machine.state.value:
            raise InvalidStateTransition("state record does not match reconstructed history")
        return machine
