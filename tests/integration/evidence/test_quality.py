from __future__ import annotations

from math import isclose
from types import SimpleNamespace

import pytest

from lazybrick.evidence import (
    EvidenceError,
    TokenLoss,
    measure_token_loss,
    quality_comparison,
    token_loss_from_vllm,
)


def output(logprobs: tuple[float, float]) -> object:
    return SimpleNamespace(
        prompt_token_ids=[10, 11, 12],
        prompt_logprobs=[
            None,
            {11: SimpleNamespace(logprob=logprobs[0])},
            {12: SimpleNamespace(logprob=logprobs[1])},
        ],
    )


def test_token_loss_uses_chosen_prompt_tokens() -> None:
    loss = token_loss_from_vllm([output((-1.0, -2.0))])

    assert loss.total_negative_log_likelihood == 3.0
    assert loss.token_count == 2
    assert loss.mean_negative_log_likelihood == 1.5


def test_quality_records_baseline_quantized_and_deltas_without_gate() -> None:
    comparison = quality_comparison(
        TokenLoss(total_negative_log_likelihood=10.0, token_count=10),
        TokenLoss(total_negative_log_likelihood=11.0, token_count=10),
    )

    assert comparison["baseline"]["mean_negative_log_likelihood"] == 1.0
    assert comparison["quantized"]["mean_negative_log_likelihood"] == 1.1
    assert isclose(comparison["absolute_delta"], 0.1)
    assert isclose(comparison["relative_delta"], 0.1)
    assert comparison["regression_gate"] is None


def test_comparison_rejects_different_token_counts() -> None:
    with pytest.raises(EvidenceError, match="token counts differ"):
        quality_comparison(TokenLoss(1, 1), TokenLoss(1, 2))


def test_measure_token_loss_requests_prompt_logprobs() -> None:
    captured: dict[str, object] = {}

    class Engine:
        def generate(self, prompts: list[str], sampling: object) -> list[object]:
            captured["prompts"] = prompts
            captured["sampling"] = sampling
            return [output((-1.0, -2.0))]

    def sampling_factory(**kwargs: object) -> object:
        captured["sampling_kwargs"] = kwargs
        return kwargs

    loss = measure_token_loss(
        Engine(), ["held-out"], seed=9, sampling_factory=sampling_factory
    )

    assert loss.token_count == 2
    assert captured["sampling_kwargs"] == {
        "temperature": 0,
        "max_tokens": 1,
        "seed": 9,
        "prompt_logprobs": 1,
    }
