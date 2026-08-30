"""Matched BF16/AWQ vLLM workload benchmarking with raw sample retention."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from lazybrick.evidence.vllm import EvidenceError


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    input_tokens: int
    output_tokens: int
    concurrency: int
    warmups: int
    repetitions: int
    seed: int

    def validate(self) -> None:
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("concurrency", self.concurrency),
            ("warmups", self.warmups),
            ("repetitions", self.repetitions),
            ("seed", self.seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise EvidenceError(f"{name} must be an integer")
        if min(self.input_tokens, self.output_tokens, self.concurrency, self.repetitions) <= 0:
            raise EvidenceError("token lengths, concurrency, and repetitions must be positive")
        if self.warmups < 0:
            raise EvidenceError("warmups must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "concurrency": self.concurrency,
            "warmups": self.warmups,
            "repetitions": self.repetitions,
            "seed": self.seed,
        }


def benchmark(
    generate: Callable[[list[str]], list[int]],
    prompts: list[str],
    protocol: BenchmarkProtocol,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Run a preconfigured batch; generate returns output-token counts per prompt."""

    protocol.validate()
    if len(prompts) != protocol.concurrency:
        raise EvidenceError("prompt count must equal configured concurrency")
    for _ in range(protocol.warmups):
        generate(prompts)
    samples: list[dict[str, float | int]] = []
    expected_total = protocol.output_tokens * protocol.concurrency
    for repetition in range(protocol.repetitions):
        started = clock()
        counts = generate(prompts)
        elapsed = clock() - started
        if len(counts) != protocol.concurrency or any(
            isinstance(count, bool) or not isinstance(count, int) for count in counts
        ):
            raise EvidenceError("generate returned invalid output-token counts")
        actual_total = sum(counts)
        if actual_total != expected_total:
            raise EvidenceError(
                f"expected {expected_total} output tokens, observed {actual_total}"
            )
        if elapsed <= 0:
            raise EvidenceError("benchmark clock did not advance")
        samples.append(
            {
                "repetition": repetition,
                "latency_seconds": elapsed,
                "output_tokens": actual_total,
                "throughput_tokens_per_second": actual_total / elapsed,
            }
        )
    return {"protocol": protocol.to_dict(), "raw_samples": samples}


def compare_benchmarks(
    baseline: dict[str, object], quantized: dict[str, object]
) -> dict[str, object]:
    if baseline.get("protocol") != quantized.get("protocol"):
        raise EvidenceError("baseline and quantized benchmark protocols differ")
    return {
        "protocol": baseline["protocol"],
        "baseline_raw_samples": baseline.get("raw_samples"),
        "quantized_raw_samples": quantized.get("raw_samples"),
    }
