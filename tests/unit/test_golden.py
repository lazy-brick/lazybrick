"""Golden serialization.

These files are the record of what LazyBrick's canonical output *is*. If a diff
here is intentional, regenerate with:

    LAZYBRICK_REGENERATE_GOLDENS=1 pytest tests/unit/test_golden.py

and review the diff. A digest that moves without a deliberate reason means an
identity changed silently, which is the failure this suite exists to catch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lazybrick import ExecutionPlan, canonical_json, load_recipe

GOLDEN = Path(__file__).parents[1] / "fixtures" / "golden"
EXAMPLE = Path(__file__).parents[2] / "examples" / "qwen3-awq.yaml"
REGENERATE = os.environ.get("LAZYBRICK_REGENERATE_GOLDENS") == "1"


def built() -> tuple[bytes, bytes]:
    document = load_recipe(EXAMPLE)
    plan = ExecutionPlan.from_recipe(document.data, document.digest)
    digests = {
        "recipe_digest": document.digest,
        "plan_digest": plan.plan_digest,
        "artifact_id": plan.artifact_id,
    }
    return canonical_json(plan.to_json()) + b"\n", canonical_json(digests) + b"\n"


def check(name: str, actual: bytes) -> None:
    path = GOLDEN / name
    if REGENERATE:
        path.write_bytes(actual)
        pytest.skip(f"regenerated {name}")
    assert actual == path.read_bytes(), (
        f"{name} drifted; if intentional, regenerate with "
        "LAZYBRICK_REGENERATE_GOLDENS=1"
    )


def test_plan_json_matches_golden() -> None:
    plan_json, _ = built()
    check("qwen3-awq.plan.json", plan_json)


def test_digests_match_golden() -> None:
    _, digests = built()
    check("qwen3-awq.digests.json", digests)


def test_plan_json_is_stable_across_repeated_builds() -> None:
    assert built() == built()
