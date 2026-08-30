# Contributing to lazybrick

Thank you for helping build lazybrick. This project treats reproducibility,
provenance, and evidence as product contracts, not documentation polish. A
change is complete only when its behavior, failure modes, tests, and claims are
reviewable by someone who did not author it.

## Before you start

1. Search the public GitHub issues and pull requests for related work.
2. Read the relevant contracts in `docs/`, especially recipes, identities,
   plugin trust, licensing, and measured status.
3. Open a GitHub issue before a large change, public API change, schema change,
   new dependency, new quantization algorithm, or paid hardware experiment.
4. Do not put private tracker URLs, private issue keys, credentials, gated
   artifact URLs, or non-public customer/research data in GitHub content,
   commits, branch names, fixtures, logs, or screenshots.

Small typo, test, and narrowly scoped documentation fixes may go directly to a
pull request when the intent is obvious.

## Raising an issue

Use GitHub Issues for public project work. Choose the closest template and make
the issue independently understandable.

### Bug reports

Include:

- the smallest reproducible example;
- expected and actual behavior;
- exact lazybrick and Python versions;
- operating system and, only when relevant, accelerator, driver, runtime, and
  dependency versions;
- the recipe or a sanitized minimal substitute;
- complete error text and the command that produced it;
- whether the failure is deterministic;
- sanitized logs or evidence records.

Remove tokens, usernames, private paths, signed URLs, and proprietary data.
Never report a vulnerability as a normal public bug; follow `SECURITY.md`.

### Features and enhancements

State the user problem before the proposed implementation. Include:

- the use case and who is blocked;
- the smallest contract or behavior change that solves it;
- acceptance criteria and explicit non-goals;
- compatibility, migration, provenance, and evidence implications;
- realistic alternatives and why they are insufficient;
- expected CPU, GPU, memory, storage, network, and maintenance cost.

A new algorithm, model family, runtime, or hardware target must explain how it
tests the existing contracts. More benchmark rows alone are not a new contract.

### Research and measurement proposals

Define the hypothesis, baseline, workload, metrics, seeds, ablations, raw data
retention, expected cost, and failure criterion before requesting hardware.
Planning and CPU/offline validation do not authorize paid GPU execution.

## Branches

Create every topic branch from current `main`. Use lowercase kebab-case and one
of these public prefixes:

| Prefix | Use |
| --- | --- |
| `feat/` | new user-facing capability |
| `bug/` | defect or security-safe bug fix |
| `enhancement/` | improvement to existing behavior |
| `docs/` | documentation-only change |
| `test/` | test-only or fixture change |
| `refactor/` | behavior-preserving restructuring |
| `perf/` | measured performance improvement |
| `research/` | non-production experiment or evaluation scaffold |
| `chore/` | CI, packaging, dependency, or repository maintenance |

Examples: `feat/gptq-adapter`, `bug/resolver-cache-refresh`, and
`docs/contributing-guide`.

Branch rules:

- do not commit directly to `main`;
- do not include private tracker keys or personal names;
- keep one logical task per branch;
- do not reuse a merged branch for unrelated work;
- rebase or merge current `main` before final review when needed;
- document the base and merge order for an unavoidable stacked PR;
- never force-push after review without warning reviewers.

`main` is the only permanent development branch. Release tags are permanent.
Topic branches are deleted after merge once there is no open PR or unique
commit. GitHub preserves the PR, review, commits, and merge record.

Safe cleanup:

```bash
git fetch --prune
git branch --merged origin/main
git push origin --delete <merged-branch>
git branch -d <merged-branch>
```

Never delete `main`, a release tag, an unmerged branch, or a branch still used
by another worktree.

## Commits

Use short Conventional Commit messages:

```text
feat: add GPTQ capability declaration
fix: reject mutable dataset revisions
docs: clarify hardware evidence requirements
test: cover malformed plugin responses
chore: update the supported Python matrix
```

Keep commits reviewable. Do not mix generated artifacts, formatting churn, and
behavior changes unless they are inseparable. Do not commit model weights,
datasets, tokens, caches, local environments, or paid-run outputs that have not
passed the publication review.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python -m build
python -m twine check dist/*
```

The default suite must remain CPU-only and offline. Tests requiring a GPU,
model download, gated input, external service, or material cost must be marked,
disabled by default, and run only with explicit approval.

## Pull requests

Open the PR against the branch that should receive the change, normally
`main`. Use a draft PR while contracts or tests are still changing.

The description must include:

- a concise summary of behavior changed;
- a public GitHub issue reference such as `Closes #123`, when one exists;
- verification commands and results;
- compatibility, migration, security, license, and reproducibility impact;
- AI-assistance disclosure when material assistance was used;
- limitations, deferred work, and anything not measured.

Do not include private issue links. If no public issue was needed, write
`Issue: none` rather than inventing one.

Maintainers choose the merge method. Squash is preferred for one logical
change. A merge commit may be used when preserving an intentionally reviewed
stack or meaningful commit boundary matters. The source branch should be
deleted after merge.

## Engineering rules

### Contracts and compatibility

- Reject unknown or ambiguous input; do not silently coerce it.
- Never fall back to a different algorithm, bit-width, target list, runtime, or
  hardware path without an explicit versioned contract.
- Version public schema and protocol changes.
- Update golden fixtures deliberately and explain every digest change.
- Keep authored recipe, resolved plan, run, attempt, artifact, and evidence
  identities distinct.
- Preserve stable failure codes and fail before expensive work when possible.

### Tests

- Add a regression test for every bug fix.
- Cover success, malformed input, unsupported capability, failure, and retry
  behavior where relevant.
- Keep default tests deterministic, CPU-only, offline, and order-independent.
- Use recorded metadata or synthetic fixtures; never let a unit test fall
  through to the live network.
- Test supported Python versions and avoid claiming unverified versions.

### Dependencies and security

- Justify each new dependency and keep optional ML stacks out of the core.
- Pin external implementations and data to immutable revisions.
- Review transitive license, supply-chain, serialization, subprocess, path,
  symlink, network, and secret-handling risks.
- Treat plugin subprocess isolation as a process boundary, not a filesystem,
  network, or GPU sandbox.
- Never load remote model code by default.

### Licensing and third-party material

Contributions are submitted under the repository's Apache-2.0 license. You must
have the right to contribute every file. Preserve required notices and identify
third-party code, models, datasets, fixtures, and generated assets with their
source, immutable revision, and license metadata. A reported license string is
evidence about metadata, not legal approval for every downstream use.

### Evidence and performance claims

Every measurement claim must identify the model revision, recipe, artifact,
evaluator, runtime and versions, hardware, workload, dependencies, seeds, and
raw results. Label simulations and unexecuted workflows honestly. Do not turn a
CPU contract test, dry run, or successful import into a hardware benchmark.

Performance PRs must include a relevant baseline and matched workload. Keep the
raw samples and report variance; do not select only favorable runs.

### Documentation

Update public documentation in the same PR as behavior. Use lowercase
`lazybrick` for the project name. Distinguish **Measured**, **Planned**, and
**Not Planned** claims. Examples must either be runnable with immutable pins or
be clearly labeled non-runnable templates.

## AI-assisted contributions

AI tools are allowed, but the human contributor remains the author and is
responsible for correctness, security, licensing, tests, and every claim.

When AI materially influenced code, tests, documentation, review, or research,
add an `AI assistance` section to the PR describing:

- the tool or model family used;
- the files or decisions it materially influenced;
- how the output was reviewed and verified;
- known uncertainty or follow-up work.

Do not publish prompts or transcripts containing secrets or private data.
Routine editor completion does not need a transcript or line-by-line
attribution.

AI must not be used to:

- fabricate test results, benchmarks, citations, reviews, authorship, or
  reproduction evidence;
- claim a command, model, dataset, runtime, or GPU was used when it was not;
- copy code or text from an unknown or incompatible source;
- bypass review, security controls, license checks, or paid-hardware approval;
- approve its own material changes without human inspection;
- weaken tests merely to make generated code pass.

Generated tests must be checked for whether they can fail for the intended
reason. Generated dependency versions, API calls, and technical claims must be
verified against primary sources or the actual pinned implementation.

## Author checklist

- [ ] The branch uses the correct prefix and contains one logical task.
- [ ] The public issue is linked, or the PR explains why none was needed.
- [ ] No private tracker reference, secret, personal data, or gated URL is present.
- [ ] Behavior, acceptance criteria, non-goals, and failure modes are clear.
- [ ] Tests cover the change and the default suite remains CPU-only/offline.
- [ ] Schema, identity, compatibility, provenance, and evidence effects were reviewed.
- [ ] Dependency, security, and third-party license effects were reviewed.
- [ ] Documentation and runnable examples match the implementation.
- [ ] Measurement claims include raw evidence and complete execution identity.
- [ ] Material AI assistance is disclosed and independently verified.
- [ ] `pytest`, package build, and distribution checks pass where applicable.
- [ ] Deferred work and unmeasured claims are stated explicitly.

## Reviewer checklist

- [ ] The change solves the stated problem without an unrelated scope increase.
- [ ] Inputs fail closed and no silent fallback was introduced.
- [ ] Tests would catch a meaningful regression, not only exercise the happy path.
- [ ] Public contracts, reason codes, digests, and migrations are intentional.
- [ ] Security and licensing claims are supported rather than inferred.
- [ ] Evidence terminology matches what was actually executed.
- [ ] AI-assisted material received the same scrutiny as human-written material.
- [ ] The PR is safe to merge and its branch can be deleted afterward.

## Community expectations

Be direct, technical, and respectful. Critique claims and designs, not people.
Disclose conflicts of interest. Do not harass contributors, expose private
information, manipulate benchmarks, or misrepresent others' work. Maintainers
may edit, hide, lock, or remove content and participation that violates these
expectations.

Questions that can help future contributors belong in a public issue. Security
and private-data concerns use `SECURITY.md`; conduct concerns use
`CODE_OF_CONDUCT.md`. Do not publish sensitive reports in a public thread.
