# Development setup

This page defines the supported local setup and verification workflow for
lazybrick contributors. Repository policy, branches, commits, and pull requests
are covered in [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Supported Python versions

The default test matrix covers Python 3.10 through 3.13. The core package is
lightweight, but optional ML dependencies may support a narrower combination of
Python, platform, accelerator, driver, and runtime versions. The committed
`.python-version` selects Python 3.13 for uv contributor environments.

## Preferred setup with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
repository, and create the locked development environment:

```bash
uv sync --extra dev
```

Run commands inside that environment without manually activating it:

```bash
uv run lazybrick validate examples/qwen3-awq.yaml
uv run pytest
uv build
uv run python -m twine check dist/*
```

`uv.lock` is committed for repeatable contributor and CI-adjacent development.
Project dependency constraints remain authoritative for downstream library
consumers.

## pip fallback

`uv` is preferred but not required. A standard virtual environment remains
supported:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m build
python -m twine check dist/*
```

On Windows PowerShell, activate the environment with
`.venv\\Scripts\\Activate.ps1`.

## Optional integrations

Install only the stack needed for the work:

```bash
uv sync --extra dev --extra awq
uv sync --extra dev --extra validation
```

The AWQ and validation extras can be large and platform-specific. Installing an
extra does not authorize model downloads, remote code, network access, GPU use,
or paid hardware. They are intentionally mutually exclusive in one environment:
the GPU smoke workflow uses separate locked build and vLLM validation processes
because their dependency stacks are incompatible.

## Verification order

Run the narrowest relevant test while iterating, then complete the default gate:

```bash
uv run pytest
uv build
uv run python -m twine check dist/*
```

The default suite must remain deterministic, CPU-only, offline, and
order-independent. GPU, model-download, external-service, gated-input, and paid
work must be explicitly marked, disabled by default, and separately approved.

When behavior, schemas, identities, evidence, or CLI output changes, update the
relevant page under `docs/`, its tests and examples, and any compatibility or
migration note in the same pull request.
