from __future__ import annotations

from lazybrick.evidence import evidence_record


def test_evidence_records_resources_and_no_universal_gate() -> None:
    record = evidence_record(
        generations=[{"prompt": "hello", "output": "world"}],
        quality={"absolute_delta": 0.1, "relative_delta": 0.02},
        performance={"baseline_raw_samples": [], "quantized_raw_samples": []},
        build_time_seconds=10.5,
        peak_host_memory_bytes=100,
        peak_gpu_memory_bytes=200,
    )

    assert record["resources"] == {
        "build_time_seconds": 10.5,
        "peak_host_memory_bytes": 100,
        "peak_gpu_memory_bytes": 200,
    }
    assert record["claims"]["quality_regression_gate"] is None
