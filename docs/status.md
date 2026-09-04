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

- the AWQ and vLLM integration code exists, but no completed GPU evidence bundle
  has been published
- `RUNTIME_CAPABILITIES` in `capabilities.py` states what LazyBrick *believes*
  vLLM can load. Those entries are claims awaiting a real load, and the list is
  kept narrow for that reason
- the M1 exit gate has not been attempted

When any of this changes, it belongs in the table above with the exact model,
recipe, runtime, hardware, and workload -- or it does not get claimed.


## Prepared evidence protocol update (unmeasured)

The smoke job now prepares final-assistant continuation loss under protocol v2.
It checks token-prefix boundaries exactly and rejects templates whose generation
prefix does not match the complete conversation. The selected continuation
includes terminating template tokens; user/system content and the assistant
header are excluded. Input token IDs, scoring boundaries and selected logprobs
are retained. The legacy v1 full-prompt metric is not relabeled as assistant loss.

Build, baseline and quantized resources are sampled in their allocating process
trees. Baseline and quantized vLLM phases use separate fresh processes with
explicit BF16, context 2048, GPU utilization 0.85 and matched seeds/workloads.
Reports retain raw samples, interval and scope. Sampled peaks are lower bounds;
summed RSS can double-count shared pages. Missing sampling data fails the run.
No real tokenizer, vLLM, CUDA, or process-memory measurement has been validated
by CPU fixtures; those remain explicit GPU-smoke acceptance gates.

The smoke recipe's evaluation protocol changes intentionally from v1 to v2,
changing its authored recipe and plan digests. Its build-input artifact identity
is unchanged because evaluation is excluded. Existing Qwen3-4B golden fixtures
are unchanged; no historical result is reinterpreted.
