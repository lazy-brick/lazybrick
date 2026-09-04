# What has actually been measured

Short answer: **nothing**.

This page exists because a compression project that is vague about which claims
are measured and which are aspirational is exactly the problem LazyBrick was
started to fix. So the rule is: state measured facts only for combinations
actually run.

## Measured

| Model | Recipe | Runtime | Hardware | Result |
| ----- | ------ | ------- | -------- | ------ |
| — | — | — | — | none yet |

No quantization has been executed. No artifact has been produced. No GPU has
been rented. No accuracy, latency, throughput, or memory number exists, and none
should be quoted from this repository.

## Verified without a GPU

These are real, tested, and run on CPU with no network:

- v0.1 recipe schema, with field-level reason codes
- opt-in v0.2 stage numerical declarations, a supplied-parameter grouped U4
  binary32 CPU oracle, and exact conformance report verification; this is a
  narrow proposed numerical profile, not a full AWQ algorithm definition
- canonical JSON and the three identities
- reference resolution against **recorded** Hugging Face metadata for
  `Qwen/Qwen3-4B`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen2.5-VL-7B-Instruct`, and
  `Qwen/Qwen3-30B-A3B`
- component-level compatibility rejection for multimodal and MoE architectures
- `inspect`, `plan`, and `build --dry-run`
- the subprocess plugin contract, pinned calibration selection, immutable run
  bundles, evidence calculations, and the GPU smoke workflow's fail-closed
  orchestration, all with CPU-only tests

`lazybrick inspect Qwen/Qwen3-4B` has also been run against the live Hub API and
returns the pinned SHA, dtype, parameter count, and licence shown in the README.

## Not yet true

- No model adapter has a tested mapping to the proposed numerical profile.
  Such stage declarations are rejected before execution. Bundle-attached
  conformance requires the separately reviewed complete bundle verifier.

- the AWQ and vLLM integration code exists, but no completed GPU evidence bundle
  has been published
- `RUNTIME_CAPABILITIES` in `capabilities.py` states what LazyBrick *believes*
  vLLM can load. Those entries are claims awaiting a real load, and the list is
  kept narrow for that reason
- the M1 exit gate has not been attempted

When any of this changes, it belongs in the table above with the exact model,
recipe, runtime, hardware, and workload -- or it does not get claimed.
