# Three identities, not one

LazyBrick computes three different digests. They answer three different
questions, and substituting one for another is how a project ends up claiming
reproducibility it has not earned.

```text
recipe_digest   what the author wrote
plan_digest     what it resolved to, including the hardware target
artifact_id     the inputs that determine the output weights
```

## `recipe_digest`

The SHA-256 of the canonical encoding of the authored recipe. Mapping-key order
does not affect it, so reformatting a YAML file does not change its identity.

It proves one thing: two recipes with the same `recipe_digest` were authored
identically. It proves nothing about what they will produce. A recipe that names
`revision: main` has a perfectly stable `recipe_digest` and means something
different every week.

Version `0.0.1` called this the "fingerprint" and the README described it as
determining artifact identity. That was wrong and is corrected.

## `plan_digest`

The SHA-256 of the resolved [`ExecutionPlan`](../src/lazybrick/records.py): the
recipe with every reference pinned to an immutable revision, plus the hardware
target and the evaluation protocol.

Two plans with the same `plan_digest` describe the same work on the same kind of
machine. This is the identity that should appear in a run record.

## `artifact_id`

The SHA-256 of the inputs that determine the output weights:

- the resolved model (URI and commit SHA)
- every stage: plugin, version, pinned implementation, and parameters
- the calibration specification, if any
- the export configuration

It deliberately **excludes** `target` and `evaluation`. Measuring an artifact on
a different GPU, or against a different eval protocol, does not make it a
different artifact.

### `artifact_id` names inputs, not bytes

This is the important part.

AWQ calibration on a GPU is **not bit-reproducible**. Kernel non-determinism,
reduction order, and library versions all move the resulting weights. Two runs
that share an `artifact_id` are expected to produce artifacts that agree *within
a stated tolerance*, never byte-for-byte.

So "reproduced" means:

- the resolved inputs hash identically, **and**
- the evidence agrees within the tolerance recorded for that protocol

It does not mean identical weight hashes, and no LazyBrick gate should ever be
written as though it does.

## Why floats are banned

Every identity is the SHA-256 of a canonical byte string, and canonical means
one value has exactly one encoding. Floating-point numbers break that:

- `128` and `128.0` are the same number and different bytes
- `0.1` is not exactly representable
- repr rules differ between languages, so a Python-written digest and a
  Go-written one would disagree about the same document

A digest that changes because an author typed a trailing `.0` is not an
identity. So recipes and plans carry integers only, and the one genuinely
fractional-looking field -- `min_compute_capability` -- is a string (`"8.0"`).

Measurements really are fractional, so `EvidenceRecord` carries scores and
latency samples as **decimal strings**. The printed form is the measurement:
`8.42` and `8.4200000000001` are different claims and a float round-trip must
not be allowed to blur them.

Raw, non-identity run logs may contain finite JSON numbers for direct analysis.
They are not canonical identity inputs; a value promoted into a digested public
record must use the decimal-string contract above.


## Numerical semantic identity (opt-in v0.2)

A profile digest hashes the immutable numerical descriptor only. A plan's
`semantic_digest` hashes the ordered stage IDs and their semantic declarations,
including explicit unspecified stages, independently of implementation pins.
Every v0.2 plan has this digest, even when all declarations are unspecified;
stage order, IDs and absent declarations remain part of that identity. Only
legacy v0.1 plans return no semantic digest. The CLI derives declaration status
from the stages themselves, independently of whether an identity exists.
It is not a recipe, build-input, output-byte, or evidence identity. Stage
semantics also participate in v0.2 plan/build-input hashes. v0.1 serialization
and every existing golden digest are deliberately unchanged.
