<div align="center">

# lazybrick

**Compose, run, and benchmark reproducible model-compression recipes.**

Pin the model revision, the plugin commit, the calibration data, and the hardware
target. The planner resolves every reference to something immutable and rejects
what it cannot reproduce -- before a weight is downloaded or a GPU is allocated.

[![CI](https://github.com/lazy-brick/lazybrick/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lazy-brick/lazybrick/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lazybrick.svg?label=pypi)](https://pypi.org/project/lazybrick/)
[![Python](https://img.shields.io/pypi/pyversions/lazybrick.svg)](https://pypi.org/project/lazybrick/)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](docs/status.md)
[![Measured results](https://img.shields.io/badge/measured%20results-none%20yet-lightgrey.svg)](docs/status.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

[Install](#install) &middot;
[Quickstart](#validate-a-recipe) &middot;
[Docs](docs/README.md) &middot;
[Contributing](CONTRIBUTING.md) &middot;
[Security](SECURITY.md) &middot;
[Status](docs/status.md)

</div>

> **Pre-alpha:** `0.0.x` establishes the package, recipe envelope, provenance
> primitives, and an opt-in GPU smoke workflow. The normal CLI remains
> plan-only; no hardware result is claimed until an evidence bundle is published.

## Install

For the CLI with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install lazybrick
```

For a Python project managed by uv:

```bash
uv add lazybrick
```

The standard pip install remains supported:

```bash
pip install lazybrick
```

## Validate a recipe

```bash
lazybrick validate examples/qwen3-awq.yaml
```

## Digest a recipe

```bash
lazybrick digest examples/qwen3-awq.yaml
```

```python
from lazybrick import load_recipe

recipe = load_recipe("examples/qwen3-awq.yaml")
print(recipe.digest)
```

`recipe_digest` is calculated from canonical recipe content, so mapping-key order
does not change it.

It hashes the **authored recipe only**. It is not an artifact identity and does not
prove reproducibility. Artifact identity has to be computed from resolved immutable
inputs -- the pinned model revision, the plugin implementation, the calibration
data, and the export configuration. The planner resolves supported references;
the authored-content digest still does not prove provenance or reproducibility.

## Validation

The v0.1 schema is strict. It requires a pinned plugin implementation, rejects
unknown fields, and reports every problem at once with a field path and a stable
reason code:

```text
$ lazybrick validate broken.yaml
Invalid LazyBrick recipe:
  model.revision: required field is missing [missing_field]
  stages[0].implementation.commit: must be a full 40-character commit SHA; branches and tags move [mutable_reference]
```

The schema checks *shape* only. Whether a runtime, accelerator, or quantization
scheme is actually supported is a capability question, answered by the planner.

## Inspect a model

Metadata only -- no weights are downloaded.

```text
$ lazybrick inspect Qwen/Qwen3-4B
Qwen/Qwen3-4B
  revision       1cfa9a7208912126459214e8b04321603b3df60c
  requested      main
  profile        dense-decoder
  components     language_backbone
  architecture   Qwen3ForCausalLM
  dtype          bfloat16
  weights        safetensors
  parameters     4,022,468,096
  license        apache-2.0
```

## Plan

`plan` resolves every reference to an immutable revision, then intersects what
the model, plugin, exporter, runtime, and hardware each declare. It downloads no
weights, allocates no GPU, and executes no plugin.

```bash
lazybrick plan examples/qwen3-awq.yaml \
  --target examples/targets/a100-40gb.json \
  --plugin-manifest examples/plugins/awq.manifest.json
```

An incompatible plan is rejected with the component that has no path, before any
of it costs anything:

```text
REJECTED: 2 problem(s).
  plugin:awq: awq does not support the multimodal-decoder profile that
    Qwen/Qwen2.5-VL-7B-Instruct resolves to [unsupported_model_profile]
  plugin:awq: awq supports the language backbone but has no declared
    quantization path for the vision encoder required by
    Qwen/Qwen2.5-VL-7B-Instruct [unsupported_component]
```

`--offline` resolves from the local cache and never touches the network.
`--json` emits canonical JSON for every command.

### Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | success, or an accepted plan |
| 2 | the recipe is invalid |
| 3 | a reference could not be resolved |
| 4 | the plan is incompatible |
| 5 | usage error, or a refused operation |

## Three identities, not one

```text
recipe_digest   what the author wrote
plan_digest     what it resolved to, including the hardware target
artifact_id     the inputs that determine the output weights
```

`artifact_id` names *inputs*, not output bytes. GPU quantization is not
bit-reproducible, so two runs sharing an `artifact_id` are expected to agree
within tolerance, never byte-for-byte.

## Documentation

| Page | Covers |
| ---- | ------ |
| [CONTRIBUTING.md](CONTRIBUTING.md) | issues, branches, pull requests, AI usage, evidence rules, and checklists |
| [AGENTS.md](AGENTS.md) | concise repository instructions for Codex, Claude Code, and other coding agents |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | community expectations and enforcement |
| [SECURITY.md](SECURITY.md) | private vulnerability reporting and security scope |
| [docs/README.md](docs/README.md) | documentation index |
| [docs/development.md](docs/development.md) | local setup with uv or pip and the verification workflow |
| [docs/recipes.md](docs/recipes.md) | the v0.1 schema, field by field, and every reason code |
| [docs/identities.md](docs/identities.md) | `recipe_digest` vs `plan_digest` vs `artifact_id`, and why floats are banned |
| [docs/plugin-trust.md](docs/plugin-trust.md) | the plugin trust boundary, and what is *not* sandboxed yet |
| [docs/licenses.md](docs/licenses.md) | third-party licensing, kept separate from lazybrick's Apache-2.0 |
| [docs/status.md](docs/status.md) | what has actually been measured (currently: nothing) |

## Status

Pre-alpha. Planning is real: recipes are validated, references are resolved to
immutable revisions, and incompatible plans are rejected with component-level
reasons before anything is downloaded.

The repository contains an isolated Qwen3-0.6B AWQ-to-vLLM smoke workflow, but it
has not produced a published evidence bundle. Therefore no quantization,
accuracy, latency, throughput, or memory result is claimed. See
[docs/status.md](docs/status.md).

## Development

```bash
uv sync --extra dev
uv run pytest
uv build
uv run python -m twine check dist/*
```

The whole suite runs on CPU with no network, against recorded Hugging Face
fixtures. See [docs/development.md](docs/development.md) for the pip fallback and
optional integration environments.

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md); it
covers public issue hygiene, branch naming, review expectations, and
AI-assisted contribution requirements.

## License

Apache-2.0. Third-party models, datasets, runtimes, and plugin implementations
keep their own licenses -- see [docs/licenses.md](docs/licenses.md).
