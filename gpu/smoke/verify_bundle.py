"""Fail closed unless exactly one successful smoke evidence bundle is complete.

Structural integrity -- file coverage, symlink rejection, and record agreement --
lives in `lazybrick.runs.bundle` so it is exercised by the CPU/offline suite.
What remains here is what is specific to the smoke workflow: that exactly one
attempt exists, and that its results carry the evidence the workflow promised.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

from lazybrick.evidence.assistant_quality import compare_assistant_quality
from lazybrick.runs import verify_bundle


def _require_resource_record(value: object, phase: str) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence contains no {phase} resource record")
    if value.get("schema_version") != "0.1" or value.get("scope") != "allocating_process_tree":
        raise RuntimeError(f"{phase} resource record has an unsupported contract")
    if not isinstance(value.get("method"), str) or not value["method"].strip():
        raise RuntimeError(f"{phase} resource sampling method is missing")
    if type(value.get("interval_ms")) is not int or value["interval_ms"] < 10:
        raise RuntimeError(f"{phase} resource sampling interval is invalid")
    samples = value.get("raw_samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise RuntimeError(f"{phase} resource samples are incomplete")
    for sample in samples:
        if not isinstance(sample, dict):
            raise RuntimeError(f"{phase} resource sample is invalid")
        if (
            not isinstance(sample.get("pids"), list)
            or not sample["pids"]
            or any(type(pid) is not int or pid < 0 for pid in sample["pids"])
        ):
            raise RuntimeError(f"{phase} resource sample has no process identity")
        for key in ("sum_rss_bytes", "gpu_process_bytes", "vanished_processes", "elapsed_ns"):
            if type(sample.get(key)) is not int or sample[key] < 0:
                raise RuntimeError(f"{phase} resource sample has invalid {key}")
    expected_host = max(sample["sum_rss_bytes"] for sample in samples)
    expected_gpu = max(sample["gpu_process_bytes"] for sample in samples)
    if value.get("sampled_peak_sum_rss_bytes") != expected_host:
        raise RuntimeError(f"{phase} host-memory peak disagrees with raw samples")
    if value.get("sampled_peak_gpu_process_bytes") != expected_gpu:
        raise RuntimeError(f"{phase} GPU-memory peak disagrees with raw samples")
    if not isinstance(value.get("limitations"), str) or not value["limitations"].strip():
        raise RuntimeError(f"{phase} resource limitations are missing")


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
    if results.get("schema_version") != "0.2":
        raise RuntimeError("evidence results contract is unsupported")
    if not results.get("generations"):
        raise RuntimeError("evidence contains no raw generation outputs")
    quality = results.get("quality")
    if not isinstance(quality, dict):
        raise RuntimeError("evidence contains no assistant-continuation quality protocol")
    try:
        expected_quality = compare_assistant_quality(quality["baseline"], quality["quantized"])
    except Exception as error:
        raise RuntimeError("assistant-continuation quality evidence is invalid") from error
    if quality != expected_quality:
        raise RuntimeError("assistant-continuation quality summary disagrees with raw evidence")
    if quality.get("baseline") is None:
        raise RuntimeError("evidence contains no BF16 baseline quality score")
    if quality.get("quantized") is None:
        raise RuntimeError("evidence contains no AWQ quality score")
    performance = results.get("performance")
    if not isinstance(performance, dict):
        raise RuntimeError("evidence contains no matched benchmark record")
    if not performance.get("baseline_raw_samples") or not performance.get(
        "quantized_raw_samples"
    ):
        raise RuntimeError("evidence contains no raw matched benchmark samples")
    resources = results.get("resources")
    if not isinstance(resources, dict) or set(resources) != {"build", "baseline", "quantized"}:
        raise RuntimeError("evidence resource phases are incomplete")
    for phase in ("build", "baseline", "quantized"):
        _require_resource_record(resources[phase], phase)
    runtime = results.get("runtime")
    expected_runtime = {
        "name": "vllm",
        "seed": 1234,
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "max_model_len": 2048,
        "gpu_memory_utilization": "0.85",
    }
    if not isinstance(runtime, dict) or any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise RuntimeError("evidence runtime settings are incomplete or mismatched")
    if not isinstance(runtime.get("version"), str) or not runtime["version"].strip():
        raise RuntimeError("evidence runtime version is missing")
    build_time = results.get("build_time_seconds")
    try:
        build_time_value = float(build_time) if isinstance(build_time, str) else float("nan")
    except ValueError as error:
        raise RuntimeError("evidence build duration is invalid") from error
    if not isfinite(build_time_value) or build_time_value <= 0:
        raise RuntimeError("evidence build duration is invalid")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    args = parser.parse_args()
    print(verify(args.store.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
