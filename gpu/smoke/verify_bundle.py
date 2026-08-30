"""Fail closed unless exactly one successful smoke evidence bundle is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lazybrick.runs import verify_hashes


REQUIRED = {
    "identity.json",
    "recipe.yaml",
    "resolved_recipe.json",
    "plan.json",
    "artifact.json",
    "provenance.json",
    "results.json",
    "state-history.json",
    "status.json",
}


def verify(store: Path) -> Path:
    attempts = sorted((store / "runs").glob("*/attempts/*"))
    if len(attempts) != 1:
        raise RuntimeError(f"expected one attempt bundle, found {len(attempts)}")
    bundle = attempts[0]
    status_path = bundle / "status.json"
    if not status_path.is_file():
        raise RuntimeError("evidence bundle is missing status.json")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "SUCCEEDED":
        raise RuntimeError(f"smoke attempt did not succeed: {status}")
    missing = sorted(name for name in REQUIRED if not (bundle / name).is_file())
    if missing:
        raise RuntimeError(f"evidence bundle is missing: {missing}")
    artifact = json.loads((bundle / "artifact.json").read_text(encoding="utf-8"))
    identity = json.loads((bundle / "identity.json").read_text(encoding="utf-8"))
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    state = json.loads((bundle / "state-history.json").read_text(encoding="utf-8"))
    for key in ("recipe_digest", "plan_digest", "artifact_id"):
        if identity.get(key) != plan.get(key):
            raise RuntimeError(f"identity and plan disagree on {key}")
    if artifact.get("artifact_id") != identity.get("artifact_id"):
        raise RuntimeError("artifact and identity disagree on artifact_id")
    if state.get("state") != "SUCCEEDED":
        raise RuntimeError("state history is not terminal SUCCEEDED")
    verify_hashes(bundle / "artifact", artifact["files"])
    results = json.loads((bundle / "results.json").read_text(encoding="utf-8"))
    if not results.get("generations"):
        raise RuntimeError("evidence contains no raw generation outputs")
    if results.get("quality", {}).get("baseline") is None:
        raise RuntimeError("evidence contains no BF16 baseline quality score")
    if results.get("quality", {}).get("quantized") is None:
        raise RuntimeError("evidence contains no AWQ quality score")
    performance = results.get("performance", {})
    if not performance.get("baseline_raw_samples") or not performance.get(
        "quantized_raw_samples"
    ):
        raise RuntimeError("evidence contains no raw matched benchmark samples")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    args = parser.parse_args()
    print(verify(args.store.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
