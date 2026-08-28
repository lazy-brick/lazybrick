# LazyBrick Implementation Blueprint

> **Status:** Draft for discussion
> **License:** Apache-2.0
> **Scope:** Quantization-first model-compression infrastructure

## 1. Project summary

LazyBrick is open infrastructure for composing, running, verifying, and sharing reproducible model-compression recipes across models, runtimes, and hardware.

The first useful workflow should be:

```text
model + recipe + target runtime/hardware
                    |
                    v
       validated compatibility plan
                    |
                    v
         reproducible compressed artifact
                    |
                    v
        accuracy and performance evidence
```

LazyBrick does not need to reimplement AWQ, GPTQ, SmoothQuant, or future algorithms. It needs to make external implementations composable, isolated, versioned, and measurable through a stable contract.

## 2. Problem statement

Compression results are currently difficult to trust or compare because the important inputs are fragmented:

- Algorithm code lives in separate repositories.
- Recipes are often embedded in scripts, notebooks, or command history.
- Checkpoints are published without complete provenance.
- Accuracy depends on calibration and evaluation details that are frequently omitted.
- Throughput depends on runtime, kernels, hardware, workload shape, and measurement policy.
- A checkpoint that loads in one runtime may be unsupported or slower in another.

LazyBrick should turn every result into a reproducible tuple:

```text
model revision
+ recipe revision
+ plugin implementation
+ calibration data
+ exported artifact
+ runtime version
+ hardware and software environment
+ evaluation protocol
= verifiable result
```

## 3. Initial scope

### Proposed MVP

- Local execution on Linux with NVIDIA CUDA GPUs.
- Decoder-only language models as the first stable profile.
- One reference model small enough for frequent CI smoke tests.
- One larger reference model for an end-to-end H100 test.
- One algorithm initially; expand to AWQ, GPTQ, and SmoothQuant after the contract works.
- Hugging Face/SafeTensors model input.
- vLLM as the first serving runtime.
- Accuracy, peak-memory, latency, and throughput reporting.
- Local artifact cache plus optional remote artifact pointers.

### Designed for later, not promised by the MVP

- Vision, audio, diffusion, and multimodal profiles.
- AMD, Apple, Intel, Qualcomm, and other hardware targets.
- TensorRT-LLM, SGLang, MLX-LM, llama.cpp, ONNX Runtime, and other runtimes.
- Hosted GPU execution.
- Public checkpoint hosting.
- Automated algorithm search or recommendation.

### Non-goals

- Rewriting every compression algorithm inside LazyBrick.
- Claiming every algorithm is interchangeable with every other algorithm.
- Treating QDQ/simulated accuracy as equivalent to deployed-runtime accuracy.
- Storing multi-gigabyte model weights directly in Git.
- Executing unpinned community code inside the LazyBrick process.
- Producing a leaderboard without publishing the complete measurement protocol.

## 4. Core concepts

### Model artifact

An immutable reference to a base or compressed model. It includes format, revision, hashes, architecture metadata, and license metadata.

### Recipe

A declarative, versioned plan describing the compression stages, parameters, calibration inputs, export target, evaluation protocol, and target environment.

### Plugin

An implementation of one transformation or integration behind a standard contract. A plugin may represent an algorithm, exporter, runtime, dataset adapter, or evaluator.

### Runtime

The software engine that executes the final artifact, such as vLLM, TensorRT-LLM, MLX-LM, llama.cpp, or ONNX Runtime.

### Target

The requested runtime and hardware environment, including accelerator type, supported precision, CUDA/ROCm version, memory constraints, and runtime capabilities.

### Run

One immutable execution attempt connecting all inputs, environment information, logs, outputs, and status transitions.

### Evidence

Accuracy and performance measurements produced under a fully specified protocol.

## 5. Compatibility model

Plug-and-play must mean **mechanically verified compatibility**, not universal interchangeability.

Before downloading large artifacts or reserving a GPU, LazyBrick should resolve:

```text
model capabilities
    intersect algorithm capabilities
    intersect exporter capabilities
    intersect runtime capabilities
    intersect hardware capabilities
```

If the intersection is empty, the run must fail during planning with an actionable explanation.

Example:

```text
Rejected: AWQ plugin supports the language backbone but has no declared
quantization path for the vision encoder required by this recipe.
```

For composite models, the plan should work at component level:

```text
vision encoder      -> BF16
projector           -> INT8
language backbone   -> AWQ INT4
```

## 6. Proposed public interface

### Installation

```bash
pip install lazybrick
```

Optional integrations may use extras:

```bash
pip install "lazybrick[vllm,awq]"
```

### Explicit Python workflow

```python
import lazybrick as lb

target = lb.Target(
    runtime="vllm",
    device="cuda:0",
)

artifact = lb.build(
    model="Qwen/Qwen3.5-9B",
    recipe="lazybrick://recipes/qwen3.5-9b/awq-w4a16",
    target=target,
)

server = lb.serve(
    artifact,
    runtime="vllm",
    port=8000,
)
```

### Convenience workflow

```python
import lazybrick as lb

model = lb.load(
    "Qwen/Qwen3.5-9B",
    recipe="lazybrick://recipes/qwen3.5-9b/awq-w4a16",
    runtime="vllm",
    device="cuda:0",
)

response = model.chat([
    {
        "role": "user",
        "content": [
            {"type": "image_url", "url": "https://example.com/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
])
```

The convenience API must delegate to the same resolver, build, and serve operations. It must not hide incompatibility or silently select an unverified implementation.

### CLI

```bash
lazybrick inspect Qwen/Qwen3.5-9B

lazybrick plan \
  Qwen/Qwen3.5-9B \
  --recipe lazybrick://recipes/qwen3.5-9b/awq-w4a16 \
  --runtime vllm \
  --device cuda:0

lazybrick build \
  Qwen/Qwen3.5-9B \
  --recipe lazybrick://recipes/qwen3.5-9b/awq-w4a16 \
  --runtime vllm \
  --device cuda:0

lazybrick serve ./artifacts/<artifact-id> \
  --runtime vllm \
  --port 8000

lazybrick verify ./artifacts/<artifact-id> \
  --suite standard-llm
```

## 7. Recipe format

Illustrative only; the first schema should remain intentionally small.

```yaml
schema_version: "0.1"

model:
  uri: hf://Qwen/Qwen3.5-9B
  revision: <immutable-commit>

stages:
  - id: quantize-language-backbone
    plugin: lazybrick.plugin/awq
    plugin_version: "1.0.0"
    implementation:
      git: https://github.com/example/awq-plugin
      commit: <immutable-commit>
      container: ghcr.io/example/awq-plugin@sha256:<digest>
    parameters:
      weight_bits: 4
      group_size: 128
      targets:
        - language_backbone.linear

calibration:
  dataset:
    uri: hf-dataset://example/calibration-set
    revision: <immutable-commit>
  samples: 512
  seed: 42

export:
  format: safetensors
  runtime: vllm

evaluation:
  accuracy_suite: standard-llm
  performance_suite: h100-vllm

target:
  accelerator: nvidia-h100
  runtime: vllm
```

Required properties:

- Every mutable external reference must resolve to an immutable revision.
- Secrets must never be embedded in a recipe.
- Model, dataset, and implementation licenses must be recorded.
- Hardware performance claims must include an exact workload protocol.

## 8. Plugin contract

Plugins should be declared through a manifest and executed out of process by default.

```yaml
name: awq
plugin_api: "0.1"
kind: transformation

entrypoint: lazybrick_plugin:AWQPlugin

accepts:
  model_profiles:
    - decoder-transformer
  artifact_formats:
    - pytorch
    - safetensors

produces:
  artifact_formats:
    - safetensors-awq

supports:
  precisions:
    - W4A16
  components:
    - linear
  runtimes:
    - vllm

requires:
  calibration: true
  activation_capture: true

licenses:
  plugin: Apache-2.0
  implementation: BSD-3-Clause
```

Proposed lifecycle:

```python
class CompressionPlugin(Protocol):
    def inspect(self, model, target) -> CapabilityReport: ...
    def plan(self, model, recipe, target) -> ExecutionPlan: ...
    def execute(self, plan, context) -> ArtifactRef: ...
    def validate(self, artifact, context) -> ValidationReport: ...
```

The concrete transport is an open question. Candidates include a Python subprocess protocol, JSON-RPC, or a container command contract. The public data schemas should not depend on the transport choice.

## 9. Run lifecycle

Proposed states:

```text
CREATED
  -> RESOLVING
  -> VALIDATING
  -> READY
  -> RUNNING
  -> EXPORTING
  -> EVALUATING
  -> SUCCEEDED
```

Terminal failure states:

```text
INCOMPATIBLE
BUILD_FAILED
EXECUTION_FAILED
EXPORT_FAILED
EVALUATION_FAILED
CANCELLED
```

Retries must create new attempts linked to the original run. A failed or incomplete execution must never be published as verified evidence.

## 10. Artifact and provenance model

Artifacts should be content-addressed where possible:

```text
<artifact-id> = hash(
    source model revision,
    recipe revision,
    plugin implementation digest,
    calibration revision,
    export configuration
)
```

Each artifact directory should contain:

```text
artifact/
├── model files or remote pointers
├── recipe.yaml
├── artifact.json
├── provenance.json
├── results.json
└── logs/
```

`provenance.json` should capture at minimum:

- Source model URI, immutable revision, and hashes.
- Recipe schema version and hash.
- Plugin name, version, Git commit, and container digest.
- Dataset URI, revision, sample selection, and preprocessing.
- Random seeds.
- Python, PyTorch, CUDA, driver, kernel, and runtime versions.
- Exact hardware identifiers and memory capacity.
- Commands and environment variables, with secrets redacted.
- Start/end timestamps and run status.

## 11. Benchmark contract

### Accuracy

- Dataset and task revisions.
- Preprocessing and tokenizer revisions.
- Prompt/chat template.
- Generation parameters.
- Metric implementation and version.
- Baseline score, compressed score, and permitted regression.

### Performance

- Runtime and kernel versions.
- GPU model, count, clocks/power policy when available, and topology.
- Input length and output length.
- Batch size and concurrency.
- Warm-up policy and number of measured repetitions.
- Prefill latency, decode latency, time to first token, and inter-token latency.
- Request and token throughput.
- Peak accelerator and host memory.
- Raw per-run measurements, not only aggregates.

Every claim should be addressed by the complete key:

```text
model + recipe + artifact format + runtime + hardware + workload
```

## 12. Security and trust

- Do not import arbitrary community plugin code into the main process.
- Pin Git repositories to commits and containers to image digests.
- Run external plugins with explicit filesystem, network, GPU, and secret permissions.
- Default to no network during transformation after inputs are materialized.
- Treat remote model code as untrusted; require an explicit opt-in and immutable revision.
- Generate a software bill of materials for published artifacts where practical.
- Verify artifact hashes before loading.
- Record model, dataset, plugin, and runtime licenses separately.
- Never imply that LazyBrick's Apache-2.0 license relicenses third-party assets.

## 13. Proposed repository layout

Start with one implementation repository:

```text
lazybrick/
├── src/lazybrick/
│   ├── api.py
│   ├── cli.py
│   ├── artifacts/
│   ├── compatibility/
│   ├── plugins/
│   ├── provenance/
│   ├── recipes/
│   ├── runners/
│   └── runtimes/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── examples/
├── docs/
├── pyproject.toml
├── LICENSE
├── NOTICE
└── README.md
```

Potential organization repositories, created only when justified:

```text
lazybrick      # SDK, CLI, contracts, adapters, and local execution
recipes        # curated recipes and verified result manifests
benchmarks     # independent evaluation harness and workload definitions
.github        # organization profile and shared community files
```

Possible later repositories:

```text
spec            # externally adopted stable schemas and conformance tests
plugin-template # community plugin starter
runner          # hosted sandboxed GPU execution
registry        # discovery website and API
integrations    # split only if first-party adapters outgrow the main repo
```

## 14. Milestones

### M0: Contract skeleton

- Package and CLI skeleton.
- Minimal recipe schema.
- Minimal plugin manifest.
- Capability vocabulary.
- `inspect` and `plan` commands.
- Dry-run compatibility resolver.
- Apache-2.0 `LICENSE` and initial `NOTICE`.

**Exit criterion:** A recipe can be parsed, resolved, and rejected or accepted without executing GPU code.

### M1: One complete vertical slice

- One model profile.
- One external algorithm plugin.
- One artifact format.
- One runtime.
- One NVIDIA GPU target.
- Local build, serve, and provenance capture.

**Exit criterion:** A fresh machine can reproduce the compressed artifact and its accuracy result from only the recipe and documented credentials.

### M2: Evidence and comparison

- Baseline-versus-compressed accuracy comparison.
- Standard performance workload.
- Raw measurement retention.
- Artifact caching and reuse.
- Regression detection.

**Exit criterion:** Two independent users produce results within defined accuracy and performance tolerances.

### M3: Multiple algorithms

- AWQ, GPTQ, and SmoothQuant adapters.
- Compatibility explanations.
- Recipe and result publication workflow.
- Initial `recipes` and `benchmarks` repositories if separation is useful.

**Exit criterion:** Users can swap compatible algorithms without changing evaluation or provenance code.

### M4: Ecosystem

- Public plugin template.
- Conformance suite.
- First third-party plugin maintained outside the LazyBrick organization.
- Second runtime or second hardware vendor.
- First modality-specific profile beyond the initial scope.

**Exit criterion:** An external contributor can publish a plugin and verified recipe without modifying LazyBrick core.

## 15. Success metrics

Prefer dependency and reproducibility metrics over GitHub stars:

- Percentage of published results reproduced by CI or a second machine.
- Number of verified model-recipe-runtime-hardware combinations.
- Number of external plugins passing conformance tests.
- Number of downstream organizations using LazyBrick in real workflows.
- Median time from recipe submission to reproducible result.
- Compatibility failures caught before GPU allocation.
- Benchmark variance across repeated runs.

## 16. Kill criteria

Reassess the project if, after the first vertical slices:

- Existing tools can represent and reproduce the same workflow without meaningful glue code.
- Plugin differences cannot be captured without leaking most implementation details into core.
- Published performance results remain too hardware- or environment-sensitive to reproduce within useful tolerances.
- Users primarily want a hosted quantization service rather than an interoperable recipe standard.
- Algorithm maintainers will not adopt or map to the proposed contract.

## 17. Open questions

1. Should the first model profile be text-only decoder LLMs, or should the first vertical slice deliberately include a multimodal model?
2. Which single algorithm is best for M1 based on active maintenance and runtime support?
3. Should the initial plugin transport be a Python subprocess or a container command protocol?
4. Where should immutable checkpoints live initially: local cache, Hugging Face Hub, or pluggable object storage?
5. Should recipes reference a specific implementation, or an abstract algorithm resolved by policy?
6. What constitutes a verified result: maintainer run, CI run, two independent runs, or hardware-provider attestation?
7. How should component-level mixed precision be represented for multimodal and mixture-of-experts models?
8. Which performance workload should be canonical, given that batch size, context length, and concurrency materially change conclusions?
9. Should LazyBrick recommend recipes, or remain a neutral executor and evidence registry?
10. When should schemas move from the main repository into a standalone `spec` repository?

## 18. Immediate next decisions

Before implementation begins, settle only these points:

1. M1 reference model.
2. M1 algorithm implementation.
3. M1 runtime and exact H100 environment.
4. Minimal recipe schema.
5. Plugin isolation boundary.
6. Accuracy and performance acceptance thresholds.

Everything else should remain replaceable until the first vertical slice is reproducible.
