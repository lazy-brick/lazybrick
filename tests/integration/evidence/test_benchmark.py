from __future__ import annotations

import pytest

from lazybrick.evidence import BenchmarkProtocol, EvidenceError, benchmark, compare_benchmarks


def protocol() -> BenchmarkProtocol:
    return BenchmarkProtocol(
        input_tokens=128,
        output_tokens=16,
        concurrency=2,
        warmups=1,
        repetitions=2,
        seed=42,
    )


def test_benchmark_retains_raw_samples_and_discards_warmup() -> None:
    calls: list[list[str]] = []
    ticks = iter([1.0, 2.0, 3.0, 5.0])

    def generate(prompts: list[str]) -> list[int]:
        calls.append(prompts)
        return [16, 16]

    result = benchmark(generate, ["a", "b"], protocol(), clock=lambda: next(ticks))

    assert len(calls) == 3
    assert result["protocol"] == protocol().to_dict()
    assert result["raw_samples"] == [
        {
            "repetition": 0,
            "latency_seconds": 1.0,
            "output_tokens": 32,
            "throughput_tokens_per_second": 32.0,
        },
        {
            "repetition": 1,
            "latency_seconds": 2.0,
            "output_tokens": 32,
            "throughput_tokens_per_second": 16.0,
        },
    ]


def test_bf16_and_awq_protocols_must_match() -> None:
    baseline = {"protocol": protocol().to_dict(), "raw_samples": []}
    changed = protocol().to_dict()
    changed["concurrency"] = 4
    quantized = {"protocol": changed, "raw_samples": []}

    with pytest.raises(EvidenceError, match="protocols differ"):
        compare_benchmarks(baseline, quantized)


def test_benchmark_rejects_short_generation() -> None:
    with pytest.raises(EvidenceError, match="expected 32"):
        benchmark(
            lambda prompts: [16, 15],
            ["a", "b"],
            protocol(),
            clock=iter([1.0, 2.0]).__next__,
        )
