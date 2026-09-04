"""Strict single-GPU vLLM loading and deterministic generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


class EvidenceError(RuntimeError):
    """Raised when runtime validation or evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class GenerationProtocol:
    max_output_tokens: int
    seed: int

    def validate(self) -> None:
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise EvidenceError("max_output_tokens must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EvidenceError("seed must be an integer")


def create_vllm_engine(
    artifact_path: str | Path,
    *,
    seed: int,
    llm_factory: Callable[..., object] | None = None,
    runtime: Mapping[str, object] | None = None,
) -> object:
    path = Path(artifact_path).resolve()
    if not path.is_dir():
        raise EvidenceError(f"artifact directory does not exist: {path}")
    options: dict[str, object] = {}
    if runtime is not None:
        if set(runtime) != {"dtype", "max_model_len", "gpu_memory_utilization"}:
            raise EvidenceError("runtime settings must explicitly pin dtype, context, and GPU utilization")
        if runtime["dtype"] not in {"float16", "bfloat16", "float32"}:
            raise EvidenceError("runtime dtype is unsupported")
        if type(runtime["max_model_len"]) is not int or runtime["max_model_len"] < 2:
            raise EvidenceError("runtime context bound is invalid")
        utilization = runtime["gpu_memory_utilization"]
        if not isinstance(utilization, str):
            raise EvidenceError("GPU utilization must be an explicit decimal string")
        try:
            value = float(utilization)
        except ValueError as error:
            raise EvidenceError("invalid GPU utilization") from error
        if not 0 < value < 1:
            raise EvidenceError("invalid GPU utilization")
        options = dict(runtime, gpu_memory_utilization=value)
    if llm_factory is None:
        try:
            from vllm import LLM
        except ImportError as error:
            raise RuntimeError(
                'vLLM validation requires: pip install "lazybrick[validation]"'
            ) from error
        llm_factory = LLM
    return llm_factory(
        model=str(path),
        tensor_parallel_size=1,
        trust_remote_code=False,
        seed=seed,
        **options,
    )


def validate_vllm_load(
    artifact_path: str | Path,
    *,
    seed: int,
    llm_factory: Callable[..., object] | None = None,
) -> dict[str, object]:
    engine = create_vllm_engine(artifact_path, seed=seed, llm_factory=llm_factory)
    try:
        config = getattr(engine, "llm_engine", None)
        model_config = getattr(config, "model_config", None)
        dtype = getattr(model_config, "dtype", None)
        return {
            "loaded": True,
            "runtime": "vllm",
            "tensor_parallel_size": 1,
            "trust_remote_code": False,
            "dtype": str(dtype) if dtype is not None else "unknown",
        }
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


def deterministic_sampling_params(
    protocol: GenerationProtocol,
    *,
    sampling_factory: Callable[..., object] | None = None,
    prompt_logprobs: int | None = None,
) -> object:
    protocol.validate()
    if sampling_factory is None:
        try:
            from vllm import SamplingParams
        except ImportError as error:
            raise RuntimeError(
                'vLLM validation requires: pip install "lazybrick[validation]"'
            ) from error
        sampling_factory = SamplingParams
    kwargs: dict[str, object] = {
        "temperature": 0,
        "max_tokens": protocol.max_output_tokens,
        "seed": protocol.seed,
    }
    if prompt_logprobs is not None:
        kwargs["prompt_logprobs"] = prompt_logprobs
    return sampling_factory(**kwargs)


def render_non_thinking_prompts(
    conversations: Iterable[list[dict[str, str]]], tokenizer: object
) -> list[str]:
    render = getattr(tokenizer, "apply_chat_template", None)
    if not callable(render):
        raise EvidenceError("tokenizer does not provide apply_chat_template")
    prompts: list[str] = []
    for conversation in conversations:
        prompt = render(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(prompt, str) or not prompt:
            raise EvidenceError("chat template produced an empty prompt")
        prompts.append(prompt)
    return prompts


def _output_text(candidate: object) -> str:
    outputs = getattr(candidate, "outputs", None)
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise EvidenceError("generation must return exactly one output per prompt")
    text = getattr(outputs[0], "text", None)
    if not isinstance(text, str):
        raise EvidenceError("generation output has no text")
    return text


def run_generations(
    engine: object,
    prompts: list[str],
    protocol: GenerationProtocol,
    *,
    sampling_factory: Callable[..., object] | None = None,
) -> list[dict[str, object]]:
    if not prompts:
        raise EvidenceError("generation prompts must not be empty")
    sampling = deterministic_sampling_params(protocol, sampling_factory=sampling_factory)
    generate = getattr(engine, "generate", None)
    if not callable(generate):
        raise EvidenceError("vLLM engine does not provide generate")
    outputs = generate(prompts, sampling)
    if not isinstance(outputs, list) or len(outputs) != len(prompts):
        raise EvidenceError("generation output count does not match prompt count")
    return [
        {"prompt_index": index, "prompt": prompt, "output": _output_text(output)}
        for index, (prompt, output) in enumerate(zip(prompts, outputs, strict=True))
    ]
