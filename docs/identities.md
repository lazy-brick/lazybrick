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

## Bundle integrity

`artifact_id` and the artifact file hashes cover the produced weights. They say
nothing about the records stored beside them, so a successful attempt could keep
valid weight hashes while its recipe, resolved plan, provenance, results, state
history, or retained logs were changed underneath it.

Every promoted attempt therefore carries `bundle-manifest.json`, covering **every
regular file in the bundle except the manifest itself**:

```json
{
  "manifest_version": "0.1",
  "files": {
    "results.json": {"sha256": "...", "size": 122},
    "logs/run.log": {"sha256": "...", "size": 18},
    "artifact/model.safetensors": {"sha256": "...", "size": 7}
  }
}
```

`manifest_version` is a versioned public contract. An unrecognized version is
rejected rather than skipped.

Symlinks and non-regular files are rejected anywhere in the bundle. A FIFO or
device node is a hard failure, not a silent skip: skipping would leave the entry
outside the manifest and therefore outside verification.

### What the manifest does and does not prove

`verify_bundle()` detects **missing, added, modified, and swapped** files, and
then checks that the records agree with one another -- identity against plan,
artifact against identity, state history terminal, and every `artifact.json`
hash matching the same bytes in the manifest. Hashes alone would accept a bundle
assembled from the records of two different attempts; the link checks reject it.

It is an integrity manifest, **not a signature**. Anyone who can write to the
store can rewrite a record and rebuild the manifest. So `bundle_digest` -- the
SHA-256 of the canonical manifest -- is recorded *outside* the bundle, in the
artifact index entry under `artifacts/<artifact_id>/<attempt_id>.json`, and
verification takes it as an argument:

```python
verify_bundle(bundle, expected_digest=index["bundle_digest"])
```

Evidence published anywhere must carry that digest from a source the bundle does
not control. Verify the complete bundle before running inference on it or
publishing it.

### Promotion and indexing are separately recoverable

Before a successful attempt is promoted, its complete bundle is verified and
its original manifest digest and five run identity fields are durably recorded
in `anchors/<run_id>/<attempt_id>.json`, outside the bundle. This promotion
receipt has `receipt_version: "1"`. The receipt must be protected by the same
trust boundary as the external artifact index; it is not a signature and does
not defend against a writer who can alter both the store anchors and bundles.

The order is: write manifest, verify the staged bundle, persist the receipt,
promote by atomic rename, verify against the receipt, then create the index.
Receipt failure prevents promotion. An index failure after promotion leaves the
receipt available for recovery. `RunStore.reindex()` verifies each successful
bundle against that original receipt before creating a missing index. It never
computes a replacement trusted digest from promoted bundle content. Existing
index entries must be byte-identical and are never overwritten, even when a
concurrent writer creates the destination first. Failed attempts are not indexed.

Recovery refuses a missing or malformed receipt, changed evidence, a conflicting
index, or identity fields that disagree with the receipt or the actual
`runs/<run_id>/attempts/<attempt_id>` location. Legacy promoted bundles without a
receipt cannot be recovered automatically; preserve and consult an independently
trusted original digest through a separately reviewed migration process. Do not
bootstrap a receipt from the bundle being recovered.

Run/attempt IDs are single ASCII path components, 1–128 characters, beginning
with an alphanumeric character and followed by alphanumerics, dots, underscores
or hyphens. Artifact, recipe and plan digests must be 64 lowercase hexadecimal
characters. Receipt/index directories and files use descriptor-relative,
no-follow access; atomic create-only publication refuses symlinks, nonregular
files and replacement of an existing record. Store directories must remain
under trusted ownership, and writers must be quiescent during verification.

Manifest parsing requires exactly `manifest_version` and `files`; every file
entry requires exactly a lowercase SHA-256 and a nonnegative integer size
(booleans are invalid). Paths must be normalized relative POSIX paths with no
traversal, backslashes, drive prefixes, or self-reference. Invalid types,
unknown fields, duplicate JSON keys, nonfinite values, and malformed JSON raise
`BundleIntegrityError`, with or without an expected digest. Valid v0.1 manifest
bytes and digests are unchanged; only invalid input is rejected more strictly.

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
Artifact hashing also rejects non-regular entries immediately, before writing an
artifact inventory. Directory traversal is allowed, but the artifact root must be
a real directory and no symlink, FIFO, socket, or device may be omitted silently.

Artifact hashing walks opened directory descriptors and opens each child with
no-follow and nonblocking flags. It compares the inspected inode/type with the
opened descriptor before reading, rechecks metadata and directory entries after
reading, and bounds reads to the inspected size. Symlink, FIFO, directory, or
regular-file replacement cannot redirect hashing through an unchecked path.
Platforms without the required descriptor-relative/no-follow operations fail
closed; there is no path-based fallback. Linux and macOS support this path.
These checks detect observed mutation; they are not an atomic filesystem
snapshot. Artifact writers must be stopped before finalization, and completed
bundles must still pass the independent integrity/publication gate.
