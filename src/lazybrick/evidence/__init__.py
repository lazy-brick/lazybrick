"""vLLM load validation, quality deltas, generations, and matched benchmarks."""

from lazybrick.evidence.benchmark import (
    BenchmarkProtocol,
    benchmark,
    compare_benchmarks,
)
from lazybrick.evidence.quality import (
    EVALUATION_DATASET_REVISION,
    EVALUATION_PROTOCOL_ID,
    TokenLoss,
    measure_token_loss,
    quality_comparison,
    token_loss_from_vllm,
)
from lazybrick.evidence.record import evidence_record
from lazybrick.evidence.vllm import (
    EvidenceError,
    GenerationProtocol,
    create_vllm_engine,
    deterministic_sampling_params,
    render_non_thinking_prompts,
    run_generations,
    validate_vllm_load,
)

__all__ = [
    "EVALUATION_DATASET_REVISION",
    "EVALUATION_PROTOCOL_ID",
    "BenchmarkProtocol",
    "EvidenceError",
    "GenerationProtocol",
    "TokenLoss",
    "benchmark",
    "compare_benchmarks",
    "create_vllm_engine",
    "deterministic_sampling_params",
    "evidence_record",
    "measure_token_loss",
    "quality_comparison",
    "render_non_thinking_prompts",
    "run_generations",
    "token_loss_from_vllm",
    "validate_vllm_load",
]
