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

An attempt is promoted by an atomic rename, and only then indexed. If the index
write fails the bundle is still real evidence, so the error names the repair
rather than discarding it. `RunStore.reindex()` rebuilds any missing entry from
the bundles on disk and is idempotent. Failed attempts are manifested too -- so a
doctored failure cannot be presented as a success -- but are never indexed.

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
