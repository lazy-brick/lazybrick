# Repository instructions

These instructions apply to coding agents working anywhere in this repository.
Keep this file concise. Detailed contributor policy lives in `CONTRIBUTING.md`.

## Start here

- Read `README.md`, `CONTRIBUTING.md`, and the relevant contract pages under
  `docs/` before changing behavior.
- Inspect the current branch, worktree, and diff before editing. Preserve
  unrelated and untracked user work.
- Base each topic branch on current `main`. Keep one logical task per branch and
  pull request. Never commit directly to `main`.
- Do not put private tracker identifiers or URLs, credentials, gated artifact
  URLs, private paths, or non-public data in branches, commits, fixtures, logs,
  issues, or pull requests.
- Do not start networked, GPU, paid, or destructive work without explicit human
  approval. CPU-only offline preparation and validation are allowed.

## Setup and commands

Use `uv` for the reproducible development environment:

```bash
uv sync --extra dev
uv run pytest
uv build
uv run python -m twine check dist/*
```

The `pip` fallback and optional integration setup are documented in
`docs/development.md`.

Run the smallest relevant test first, then the full default suite before handoff.
The default suite must stay deterministic, CPU-only, offline, and
order-independent. Never claim a command passed unless it was actually run.

## Engineering contracts

- Fail closed. Reject unknown, mutable, malformed, ambiguous, or unsupported
  inputs before downloads, plugin execution, or GPU allocation.
- Never silently change an algorithm, scheme, bit-width, component, runtime,
  hardware target, plugin, or evidence tier.
- Keep authored recipe identity, resolved plan identity, run and attempt
  identity, artifact identity, output-byte hashes, and evidence identity
  distinct.
- Treat schemas, canonical JSON, stable reason codes, CLI JSON and exit codes,
  plugin protocol messages, manifests, state transitions, and evidence records
  as versioned public contracts.
- Explain every intentional digest or golden-fixture change. Do not regenerate
  fixtures merely to make tests pass.
- Treat plugin subprocesses as untrusted-code boundaries, not sandboxes. Review
  timeout, termination, path, symlink, secret, environment, serialization,
  partial-output, retry, and cleanup behavior.
- Keep core dependencies lightweight. New runtime, model, quantizer, or GPU
  dependencies must remain optional and have license and supply-chain review.
- Update tests, docs, examples, compatibility notes, and migration behavior in
  the same change as the implementation.
- Use lowercase `lazybrick` for the project name. Keep **Measured**, **Planned**,
  and **Not Planned** claims explicit. A dry run, import, simulation, or schema
  check is not hardware evidence.

## Tests and evidence

- Add a regression test for every defect fix.
- Cover malformed input, unsupported capability, partial failure, and retry or
  recovery behavior when those paths are relevant.
- Use recorded metadata or synthetic fixtures in default tests. Do not allow a
  unit test to fall through to the live network.
- Measurement claims must identify immutable model and dataset revisions, the
  full recipe, artifact, evaluator, runtime and dependency versions, hardware,
  workload, seeds, raw samples, and variance where applicable.
- Do not publish generated artifacts or evidence until hashes, provenance,
  licensing, and the complete bundle have been verified.

## Git and pull requests

- Use the branch prefixes and naming rules in `CONTRIBUTING.md`.
- Use Conventional Commits:
  `<type>(optional-scope): <imperative summary>`.
- Allowed types are `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `research`,
  and `chore`. Mark a breaking change with `!` and explain it in the commit body.
- Keep commits reviewable and free of unrelated formatting or generated-file
  churn. Do not rewrite reviewed history or force-push without explicit approval.
- Do not commit, push, open or update a pull request, merge, or delete a branch
  unless the user asked for that action.
- Target `main`. Use a Conventional Commit-style PR title and the repository PR
  template.
- Link a public GitHub issue with `Closes #<number>` when one exists. Otherwise
  write `Issue: none`. Never expose a private tracker reference.
- Record exact verification commands and results, contract and risk effects,
  limitations, unmeasured claims, and material AI assistance.
- Never approve or merge your own material change. Leave merge decisions to the
  maintainer.

## Code review rules

- Prioritize concrete correctness, security, reproducibility, compatibility,
  recovery, and evidence failures over style commentary.
- State the failure mode and the smallest safe fix. Do not restate the diff.
- Reject weakened assertions, silent fallback, mutable external references,
  identity conflation, unsupported claims, and unreviewed dependency or license
  changes.
- Confirm that a successful artifact cannot retain apparently valid evidence
  after surrounding records, logs, indexes, or provenance have been modified.

## Handoff

Report changed files, commands actually run, results, remaining uncertainty, and
any work that still requires network, GPU, paid resources, legal review, or human
approval.
