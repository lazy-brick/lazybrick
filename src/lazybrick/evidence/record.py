"""Assembly of complete evidence without inventing an unmeasured quality gate."""

from __future__ import annotations

from typing import Mapping, Sequence


def evidence_record(
    *,
    generations: Sequence[Mapping[str, object]],
    quality: Mapping[str, object],
    performance: Mapping[str, object],
    build_time_seconds: float,
    peak_host_memory_bytes: int,
    peak_gpu_memory_bytes: int,
) -> dict[str, object]:
    if build_time_seconds < 0:
        raise ValueError("build_time_seconds must be non-negative")
    if peak_host_memory_bytes < 0 or peak_gpu_memory_bytes < 0:
        raise ValueError("peak memory must be non-negative")
    return {
        "schema_version": "0.1",
        "generations": [dict(item) for item in generations],
        "quality": dict(quality),
        "performance": dict(performance),
        "resources": {
            "build_time_seconds": build_time_seconds,
            "peak_host_memory_bytes": peak_host_memory_bytes,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
        "claims": {
            "quality_regression_gate": None,
            "note": "Quality deltas are measured; no universal threshold is claimed.",
        },
    }
