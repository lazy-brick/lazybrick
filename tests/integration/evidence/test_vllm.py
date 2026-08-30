from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lazybrick.evidence import (
    GenerationProtocol,
    deterministic_sampling_params,
    render_non_thinking_prompts,
    run_generations,
    validate_vllm_load,
)


class FakeEngine:
    def __init__(self) -> None:
        self.shutdown_called = False
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(dtype="bfloat16"))

    def shutdown(self) -> None:
        self.shutdown_called = True

    def generate(self, prompts: list[str], params: object) -> list[object]:
        return [SimpleNamespace(outputs=[SimpleNamespace(text=f"answer {index}")]) for index, _ in enumerate(prompts)]


def test_vllm_load_is_single_gpu_and_remote_code_disabled(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    captured: dict[str, object] = {}
    engine = FakeEngine()

    def factory(**kwargs: object) -> FakeEngine:
        captured.update(kwargs)
        return engine

    result = validate_vllm_load(artifact, seed=7, llm_factory=factory)

    assert result["loaded"] is True
    assert captured == {
        "model": str(artifact.resolve()),
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
        "seed": 7,
    }
    assert engine.shutdown_called is True


def test_sampling_is_greedy_and_seeded() -> None:
    captured: dict[str, object] = {}

    def factory(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    deterministic_sampling_params(
        GenerationProtocol(max_output_tokens=64, seed=42), sampling_factory=factory
    )

    assert captured == {"temperature": 0, "max_tokens": 64, "seed": 42}


def test_non_thinking_prompts_are_explicit() -> None:
    captured: dict[str, object] = {}

    class Tokenizer:
        def apply_chat_template(self, conversation: object, **kwargs: object) -> str:
            captured.update(kwargs)
            return "rendered"

    assert render_non_thinking_prompts(
        [[{"role": "user", "content": "hello"}]], Tokenizer()
    ) == ["rendered"]
    assert captured["enable_thinking"] is False
    assert captured["add_generation_prompt"] is True


def test_raw_generations_are_retained() -> None:
    outputs = run_generations(
        FakeEngine(),
        ["one", "two"],
        GenerationProtocol(max_output_tokens=8, seed=3),
        sampling_factory=lambda **kwargs: kwargs,
    )

    assert outputs == [
        {"prompt_index": 0, "prompt": "one", "output": "answer 0"},
        {"prompt_index": 1, "prompt": "two", "output": "answer 1"},
    ]
