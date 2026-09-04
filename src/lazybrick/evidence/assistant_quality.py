"""Final-assistant continuation loss with exact token boundaries (protocol v2)."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isclose
from typing import Any, Callable, Iterable, Mapping

from lazybrick.canonical import digest
from lazybrick.evidence.quality import TokenLoss
from lazybrick.evidence.vllm import EvidenceError, GenerationProtocol, deterministic_sampling_params

PROTOCOL_ID = "lazybrick.ultrachat-assistant-continuation-loss.v2"


def _ids(value: object) -> list[int]:
    if not isinstance(value, list) or not value or any(type(x) is not int or x < 0 for x in value):
        raise EvidenceError("tokenization must return non-empty nonnegative integer token IDs")
    return value


def validate_samples(samples: object) -> list[dict[str, Any]]:
    if not isinstance(samples, list) or not samples:
        raise EvidenceError("assistant evaluation requires samples")
    result = []
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"prompt_token_ids", "score_start"}:
            raise EvidenceError("assistant sample fields are invalid")
        token_ids = _ids(sample["prompt_token_ids"])
        start = sample["score_start"]
        if type(start) is not int or not 0 < start < len(token_ids):
            raise EvidenceError("assistant sample has no scoreable continuation")
        result.append({"prompt_token_ids": list(token_ids), "score_start": start})
    return result


def assistant_samples(conversations: Iterable[list[dict[str, str]]], tokenizer: object,
                      *, max_model_len: int) -> list[dict[str, Any]]:
    render = getattr(tokenizer, "apply_chat_template", None)
    if not callable(render) or type(max_model_len) is not int or max_model_len < 2:
        raise EvidenceError("tokenizer and explicit context bound are required")
    samples = []
    for conversation in conversations:
        if len(conversation) < 2 or conversation[-1].get("role") != "assistant":
            raise EvidenceError("evaluation requires a final assistant continuation")
        if not conversation[-1].get("content"):
            raise EvidenceError("assistant continuation is empty")
        prefix = _ids(render(conversation[:-1], tokenize=True,
                             add_generation_prompt=True, enable_thinking=False))
        full = _ids(render(conversation, tokenize=True,
                           add_generation_prompt=False, enable_thinking=False))
        if full[:len(prefix)] != prefix:
            raise EvidenceError("chat template has ambiguous assistant token boundary")
        if len(full) >= max_model_len:
            raise EvidenceError("evaluation exceeds context bound; truncation is forbidden")
        samples.append({"prompt_token_ids": full, "score_start": len(prefix)})
    return validate_samples(samples)


@dataclass(frozen=True)
class AssistantLoss:
    loss: TokenLoss
    sample_digest: str
    samples: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {**self.loss.to_dict(), "sample_digest": self.sample_digest,
                "raw_samples": list(self.samples)}


def measure_assistant_loss(engine: object, samples: list[dict[str, Any]], *, seed: int,
                           sampling_factory: Callable[..., object] | None = None) -> AssistantLoss:
    samples = validate_samples(samples)
    sampling = deterministic_sampling_params(GenerationProtocol(1, seed),
        sampling_factory=sampling_factory, prompt_logprobs=1)
    outputs = list(engine.generate([{"prompt_token_ids": s["prompt_token_ids"]} for s in samples], sampling))
    if len(outputs) != len(samples):
        raise EvidenceError("assistant evaluation output count differs")
    total = 0.0
    count = 0
    raw = []
    for sample, output in zip(samples, outputs, strict=True):
        token_ids = sample["prompt_token_ids"]
        actual_ids = getattr(output, "prompt_token_ids", None)
        logprobs = getattr(output, "prompt_logprobs", None)
        if actual_ids != token_ids or not isinstance(logprobs, list) or len(logprobs) != len(token_ids):
            raise EvidenceError("evaluation tokenization or logprob length changed")
        values = []
        for index in range(sample["score_start"], len(token_ids)):
            alternatives = logprobs[index]
            if not isinstance(alternatives, Mapping) or token_ids[index] not in alternatives:
                raise EvidenceError("chosen assistant token logprob is missing")
            value = alternatives[token_ids[index]]
            value = getattr(value, "logprob", value)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or value > 0:
                raise EvidenceError("assistant token logprob must be finite and nonpositive")
            total -= value
            count += 1
            values.append(str(value))
        raw.append({**sample, "selected_logprobs": values})
    if not isfinite(total) or not count:
        raise EvidenceError("invalid aggregate assistant token loss")
    return AssistantLoss(TokenLoss(total, count), digest(samples), tuple(raw))


def _verify_loss(record: dict[str, Any]) -> None:
    raw = record.get("raw_samples")
    if not isinstance(raw, list) or not raw:
        raise EvidenceError("assistant raw samples are missing")
    samples = validate_samples([{k: s[k] for k in ("prompt_token_ids", "score_start")} for s in raw])
    if digest(samples) != record.get("sample_digest"):
        raise EvidenceError("assistant evaluation samples differ from their digest")
    total = 0.0
    count = 0
    for sample, result in zip(samples, raw, strict=True):
        values = result.get("selected_logprobs")
        if not isinstance(values, list) or len(values) != len(sample["prompt_token_ids"])-sample["score_start"]:
            raise EvidenceError("assistant raw logprobs are incomplete")
        for value in values:
            if not isinstance(value, str):
                raise EvidenceError("raw logprobs must be decimal strings")
            try:
                number = float(value)
            except ValueError as error:
                raise EvidenceError("raw logprob is invalid") from error
            if not isfinite(number) or number > 0:
                raise EvidenceError("raw logprob must be finite and nonpositive")
            total -= number
            count += 1
    if count != record.get("token_count") or not isclose(total, record.get("total_negative_log_likelihood", -1), rel_tol=1e-12, abs_tol=1e-12):
        raise EvidenceError("assistant loss summary disagrees with raw samples")
    if not isclose(total/count, record.get("mean_negative_log_likelihood", -1), rel_tol=1e-12, abs_tol=1e-12):
        raise EvidenceError("assistant mean loss disagrees with raw samples")


def compare_assistant_quality(baseline: dict[str, Any], quantized: dict[str, Any]) -> dict[str, Any]:
    _verify_loss(baseline)
    _verify_loss(quantized)
    if baseline["sample_digest"] != quantized["sample_digest"] or baseline["token_count"] != quantized["token_count"]:
        raise EvidenceError("assistant evaluation samples differ")
    a, b = baseline["mean_negative_log_likelihood"], quantized["mean_negative_log_likelihood"]
    return {"protocol": {"id": PROTOCOL_ID,
            "metric": "mean_final_assistant_continuation_negative_log_likelihood",
            "scope": "final assistant continuation including terminating template tokens; prompt/header excluded"},
            "baseline": baseline, "quantized": quantized, "absolute_delta": b-a,
            "relative_delta": (b-a)/a if a else None, "regression_gate": None}
