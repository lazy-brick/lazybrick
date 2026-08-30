"""Held-out token-loss extraction and baseline-versus-quantized deltas."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Callable, Iterable, Mapping

from lazybrick.evidence.vllm import (
    EvidenceError,
    GenerationProtocol,
    deterministic_sampling_params,
)


EVALUATION_DATASET_URI = "hf-dataset://HuggingFaceH4/ultrachat_200k"
EVALUATION_DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
EVALUATION_SPLIT = "test_sft"
EVALUATION_PROTOCOL_ID = "lazybrick.ultrachat-token-loss.v1"


@dataclass(frozen=True, slots=True)
class TokenLoss:
    total_negative_log_likelihood: float
    token_count: int

    @property
    def mean_negative_log_likelihood(self) -> float:
        if self.token_count <= 0:
            raise EvidenceError("token loss contains no scored tokens")
        return self.total_negative_log_likelihood / self.token_count

    @property
    def perplexity(self) -> float:
        return exp(self.mean_negative_log_likelihood)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_negative_log_likelihood": self.total_negative_log_likelihood,
            "token_count": self.token_count,
            "mean_negative_log_likelihood": self.mean_negative_log_likelihood,
            "perplexity": self.perplexity,
        }


def _logprob(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    candidate = getattr(value, "logprob", None)
    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        return float(candidate)
    raise EvidenceError("prompt logprob entry has no numeric logprob")


def token_loss_from_vllm(outputs: Iterable[object]) -> TokenLoss:
    total = 0.0
    count = 0
    for output in outputs:
        prompt_ids = getattr(output, "prompt_token_ids", None)
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if not isinstance(prompt_ids, list) or not isinstance(prompt_logprobs, list):
            raise EvidenceError("vLLM output is missing prompt token logprobs")
        if len(prompt_ids) != len(prompt_logprobs):
            raise EvidenceError("prompt token and logprob lengths differ")
        for token_id, alternatives in zip(prompt_ids[1:], prompt_logprobs[1:], strict=True):
            if not isinstance(alternatives, Mapping) or token_id not in alternatives:
                raise EvidenceError("chosen prompt token logprob is missing")
            total -= _logprob(alternatives[token_id])
            count += 1
    if count == 0:
        raise EvidenceError("evaluation produced no scored tokens")
    return TokenLoss(total_negative_log_likelihood=total, token_count=count)


def measure_token_loss(
    engine: object,
    prompts: list[str],
    *,
    seed: int,
    sampling_factory: Callable[..., object] | None = None,
) -> TokenLoss:
    """Request chosen-token prompt logprobs from vLLM for a fixed prompt set."""

    if not prompts:
        raise EvidenceError("token-loss prompts must not be empty")
    generate = getattr(engine, "generate", None)
    if not callable(generate):
        raise EvidenceError("vLLM engine does not provide generate")
    sampling = deterministic_sampling_params(
        GenerationProtocol(max_output_tokens=1, seed=seed),
        sampling_factory=sampling_factory,
        prompt_logprobs=1,
    )
    return token_loss_from_vllm(generate(prompts, sampling))


def quality_comparison(baseline: TokenLoss, quantized: TokenLoss) -> dict[str, Any]:
    if baseline.token_count != quantized.token_count:
        raise EvidenceError("baseline and quantized token counts differ")
    baseline_value = baseline.mean_negative_log_likelihood
    quantized_value = quantized.mean_negative_log_likelihood
    absolute = quantized_value - baseline_value
    relative = absolute / baseline_value if baseline_value != 0 else None
    return {
        "protocol": {
            "id": EVALUATION_PROTOCOL_ID,
            "dataset_uri": EVALUATION_DATASET_URI,
            "dataset_revision": EVALUATION_DATASET_REVISION,
            "split": EVALUATION_SPLIT,
            "metric": "mean_token_negative_log_likelihood",
        },
        "baseline": baseline.to_dict(),
        "quantized": quantized.to_dict(),
        "absolute_delta": absolute,
        "relative_delta": relative,
        "regression_gate": None,
    }
