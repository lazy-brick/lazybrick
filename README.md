# LazyBrick

Compose, run, and benchmark reproducible model-compression recipes.

> **Pre-alpha:** `0.0.x` establishes the package, recipe envelope, and provenance
> primitives. It does not yet execute quantization algorithms or inference.

## Install

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
data, and the export configuration -- none of which LazyBrick resolves yet.

## Validation

The v0.1 schema is strict. It requires a pinned plugin implementation, rejects
unknown fields, and reports every problem at once with a field path and a stable
reason code:

```text
$ lazybrick validate broken.yaml
Invalid LazyBrick recipe:
  model.revision: required field is missing [missing_field]
  stages[0].implementation: must pin the implementation with either 'git' plus
    'commit', or 'container' [missing_field]
  target.min_compute_capability: must be a 'major.minor' string such as '8.0' ...
```

The schema checks *shape* only. Whether a runtime, accelerator, or quantization
scheme is actually supported is a capability question, answered by the planner
against plugin and hardware manifests.

## Direction

LazyBrick is intended to connect versioned model artifacts, external compression
plugins, runtimes, hardware targets, and evidence through a reproducible contract.
The design is deliberately narrow until the first complete compression workflow is
working.

See [IMPLEMENTATION_BLUEPRINT.md](IMPLEMENTATION_BLUEPRINT.md) for the proposed
architecture, milestones, open questions, and kill criteria.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python -m build
python -m twine check dist/*
```

## License

Apache-2.0. Third-party models, datasets, runtimes, and plugin implementations keep
their own licenses.
