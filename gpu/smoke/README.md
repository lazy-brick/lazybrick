# Qwen3-0.6B AWQ GPU smoke job

This job executes the complete first-slice path:

1. resolve the pinned recipe with `lazybrick plan`;
2. fetch the exact Qwen3-0.6B and UltraChat revisions;
3. select 32 calibration samples deterministically;
4. quantize with the isolated LLM Compressor plugin;
5. load and evaluate BF16 and AWQ artifacts in vLLM; and
6. persist a verified, content-addressed run bundle with raw evidence.

## Execution contract

- Linux x86-64 with glibc 2.31 or newer;
- Python 3.12;
- exactly one visible NVIDIA GPU with compute capability 8.0 or newer;
- enough local storage for the two locked environments, model inputs, and outputs;
- network access to PyPI and Hugging Face; and
- explicit human approval for GPU cost and execution.

LLM Compressor 0.13.0 and vLLM 0.28.0 require incompatible
`compressed-tensors` versions. `run.sh` therefore builds two separately
hash-locked environments and crosses the boundary through files and a
subprocess. This is intentional, not dependency drift.

After approval, run:

```bash
LAZYBRICK_SMOKE_ROOT=/absolute/output/path bash gpu/smoke/run.sh
```

The pytest entry point is additionally guarded by the `gpu` marker and
`LAZYBRICK_RUN_GPU_TESTS=1`; ordinary test runs skip it.
