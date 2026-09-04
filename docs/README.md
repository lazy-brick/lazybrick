# Documentation

lazybrick documentation is organized around public contracts and verified
project state rather than a speculative implementation roadmap.

## Contributor workflow

- [`development.md`](development.md) — local setup with uv or pip and the
  verification workflow
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — issues, branches, commits, pull
  requests, AI assistance, and review checklists
- [`../AGENTS.md`](../AGENTS.md) — concise repository instructions for coding
  agents

## Contracts

- [`recipes.md`](recipes.md) — recipe schema and stable validation reason codes
- [`identities.md`](identities.md) — recipe, plan, run, artifact, and evidence
  identity boundaries
- [`plugin-trust.md`](plugin-trust.md) — plugin execution and trust boundaries
- [`licenses.md`](licenses.md) — third-party material and license metadata
- [`status.md`](status.md) — measured, planned, and unmeasured project state

Behavior changes must update the relevant contract page, tests, examples, and
compatibility or migration notes in the same pull request.

## Proposed numerical semantics

- [Supplied-parameter affine U4/group-128/binary32 profile](semantics/affine-u4-g128-f32-v1.md)
- [Scoped conformance reports and verification](semantics/conformance.md)

These CPU contracts do not establish AWQ adapter support or hardware evidence.
