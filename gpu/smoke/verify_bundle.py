"""Fail closed unless exactly one successful smoke evidence bundle is complete.

Structural integrity -- file coverage, symlink rejection, and record agreement --
lives in `lazybrick.runs.bundle` so it is exercised by the CPU/offline suite.
What remains here is what is specific to the smoke workflow: that exactly one
attempt exists, and that its results carry the evidence the workflow promised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lazybrick.runs import verify_bundle


def verify(store: Path) -> Path:
    attempts = sorted((store / "runs").glob("*/attempts/*"))
    if len(attempts) != 1:
        raise RuntimeError(f"expected one attempt bundle, found {len(attempts)}")
    bundle = attempts[0]

    # Cheap early exit with a precise reason. This reads an as-yet unverified
    # record, which is safe in one direction only: a bundle claiming success
    # still goes through full verification below.
    status_path = bundle / "status.json"
    if not status_path.is_file():
        raise RuntimeError("evidence bundle is missing status.json")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "SUCCEEDED":
        raise RuntimeError(f"smoke attempt did not succeed: {status}")

    index_entries = sorted((store / "artifacts").glob("*/*.json"))
    if len(index_entries) != 1:
        raise RuntimeError(
            f"expected one artifact index entry, found {len(index_entries)}"
        )
    index = json.loads(index_entries[0].read_text(encoding="utf-8"))

    # The expected digest comes from the index, outside the bundle, so replacing
    # the manifest along with a record is caught too.
    expected_digest = index.get("bundle_digest")
    if expected_digest is None:
        raise RuntimeError("artifact index is missing bundle_digest")
    verify_bundle(bundle, expected_digest=expected_digest)

    if index.get("bundle") != bundle.relative_to(store).as_posix():
        raise RuntimeError("artifact index does not point at the verified bundle")

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
