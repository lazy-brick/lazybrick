"""Run vLLM evidence in its own hash-locked environment."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from lazybrick.evidence import (
    BenchmarkProtocol,
    GenerationProtocol,
    benchmark,
    compare_benchmarks,
    create_vllm_engine,
    measure_token_loss,
    quality_comparison,
    render_non_thinking_prompts,
    run_generations,
)


def _fixed_prompt(tokenizer: object, token_count: int) -> str:
    encode = getattr(tokenizer, "encode")
    decode = getattr(tokenizer, "decode")
    source = "LazyBrick reproducible benchmark input. " * token_count
    token_ids = encode(source, add_special_tokens=False)[:token_count]
    prompt = decode(token_ids, skip_special_tokens=False)
    if len(encode(prompt, add_special_tokens=False)) != token_count:
        raise RuntimeError("could not construct an exact-length benchmark prompt")
    return prompt


def execute(config: dict[str, Any]) -> dict[str, object]:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    model_path = Path(config["model_path"]).resolve()
    artifact_path = Path(config["artifact_path"]).resolve()
    seed = config["seed"]
    evaluation_prompts = config["evaluation_prompts"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    generation_prompts = render_non_thinking_prompts(
        [
            [{"role": "user", "content": "State one reason reproducible experiments matter."}],
            [{"role": "user", "content": "Reply with exactly three prime numbers."}],
        ],
        tokenizer,
    )
    benchmark_protocol = BenchmarkProtocol(
        input_tokens=128,
        output_tokens=16,
        concurrency=1,
        warmups=1,
        repetitions=3,
        seed=seed,
    )
    benchmark_prompts = [_fixed_prompt(tokenizer, benchmark_protocol.input_tokens)]

    def measure(path: Path) -> tuple[list[dict[str, object]], object, dict[str, object]]:
        engine = create_vllm_engine(path, seed=seed)
        try:
            generations = run_generations(
                engine,
                generation_prompts,
                GenerationProtocol(max_output_tokens=32, seed=seed),
            )
            loss = measure_token_loss(engine, evaluation_prompts, seed=seed)
            sampling = SamplingParams(
                temperature=0,
                max_tokens=benchmark_protocol.output_tokens,
                min_tokens=benchmark_protocol.output_tokens,
                ignore_eos=True,
                seed=benchmark_protocol.seed,
            )

            def generate(prompts: list[str]) -> list[int]:
                outputs = engine.generate(prompts, sampling)
                return [len(output.outputs[0].token_ids) for output in outputs]

            performance = benchmark(generate, benchmark_prompts, benchmark_protocol)
            return generations, loss, performance
        finally:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
            gc.collect()

    _, baseline_loss, baseline_performance = measure(model_path)
    generations, quantized_loss, quantized_performance = measure(artifact_path)
    return {
        "generations": generations,
        "quality": quality_comparison(baseline_loss, quantized_loss),
        "performance": compare_benchmarks(baseline_performance, quantized_performance),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.input.read_text(encoding="utf-8"))
    result = execute(config)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
