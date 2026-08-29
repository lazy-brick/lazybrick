from __future__ import annotations

import pytest

from lazybrick.runs import InvalidStateTransition, RunState, RunStateMachine


SUCCESS_PATH = [
    RunState.RESOLVING,
    RunState.VALIDATING,
    RunState.READY,
    RunState.RUNNING,
    RunState.EXPORTING,
    RunState.EVALUATING,
    RunState.SUCCEEDED,
]


def test_complete_success_path() -> None:
    machine = RunStateMachine(created_at="2026-01-01T00:00:00Z")

    for index, state in enumerate(SUCCESS_PATH, 1):
        machine.transition(state, timestamp=f"2026-01-01T00:00:0{index}Z")

    assert machine.state is RunState.SUCCEEDED
    assert machine.terminal is True
    assert [event.state for event in machine.events] == [RunState.CREATED, *SUCCESS_PATH]


@pytest.mark.parametrize(
    ("path", "failure"),
    [
        ([RunState.RESOLVING], RunState.EXECUTION_FAILED),
        ([RunState.RESOLVING, RunState.VALIDATING], RunState.INCOMPATIBLE),
        (
            [RunState.RESOLVING, RunState.VALIDATING, RunState.READY, RunState.RUNNING],
            RunState.BUILD_FAILED,
        ),
        (
            [
                RunState.RESOLVING,
                RunState.VALIDATING,
                RunState.READY,
                RunState.RUNNING,
                RunState.EXPORTING,
            ],
            RunState.EXPORT_FAILED,
        ),
        (
            [
                RunState.RESOLVING,
                RunState.VALIDATING,
                RunState.READY,
                RunState.RUNNING,
                RunState.EXPORTING,
                RunState.EVALUATING,
            ],
            RunState.EVALUATION_FAILED,
        ),
    ],
)
def test_stage_specific_failures(path: list[RunState], failure: RunState) -> None:
    machine = RunStateMachine()
    for state in path:
        machine.transition(state)

    machine.transition(failure, reason_code="test_failure", message="failed in test")

    assert machine.state is failure
    assert machine.terminal is True


def test_cancellation_is_allowed_from_nonterminal_state() -> None:
    machine = RunStateMachine()
    machine.transition(RunState.RESOLVING)
    machine.transition(
        RunState.CANCELLED,
        reason_code="user_cancelled",
        message="cancel requested",
    )

    assert machine.state is RunState.CANCELLED


def test_illegal_transition_does_not_mutate_history() -> None:
    machine = RunStateMachine()

    with pytest.raises(InvalidStateTransition, match="CREATED to SUCCEEDED"):
        machine.transition(RunState.SUCCEEDED)

    assert machine.state is RunState.CREATED
    assert len(machine.events) == 1


def test_failure_requires_structured_reason() -> None:
    machine = RunStateMachine()
    machine.transition(RunState.RESOLVING)

    with pytest.raises(InvalidStateTransition, match="reason_code"):
        machine.transition(RunState.EXECUTION_FAILED)


def test_terminal_attempt_cannot_transition() -> None:
    machine = RunStateMachine()
    machine.transition(
        RunState.CANCELLED,
        reason_code="user_cancelled",
        message="cancel requested",
    )

    with pytest.raises(InvalidStateTransition, match="CANCELLED to RESOLVING"):
        machine.transition(RunState.RESOLVING)


def test_state_history_round_trip() -> None:
    machine = RunStateMachine(created_at="2026-01-01T00:00:00Z")
    machine.transition(RunState.RESOLVING, timestamp="2026-01-01T00:00:01Z")
    machine.transition(RunState.VALIDATING, timestamp="2026-01-01T00:00:02Z")
    machine.transition(
        RunState.INCOMPATIBLE,
        reason_code="unsupported_component",
        message="multimodal models are unsupported",
        timestamp="2026-01-01T00:00:03Z",
    )

    restored = RunStateMachine.from_dict(machine.to_dict())

    assert restored.to_dict() == machine.to_dict()
