# The plugin trust boundary

LazyBrick does not implement AWQ, and it will not implement GPTQ or SmoothQuant.
It makes external implementations composable, isolated, versioned, and
measurable. That means running other people's code, which means being explicit
about what is trusted.

## Community plugin code executes outside the LazyBrick process

Plugins are executed as **subprocesses** speaking a versioned JSON protocol.
They are not imported. A plugin cannot monkeypatch the planner, read a digest
out of memory, or make a rejected plan look accepted, because it is not in the
same process to begin with.

This is also why the plugin API version is checked *before* the process starts:
an incompatible plugin should never get as far as running.

## What must be pinned

A recipe cannot name a plugin implementation loosely. Every stage must pin
either:

- `git` plus a **full 40-character commit SHA**, or
- `container` pinned by `sha256` digest

Abbreviated SHAs are rejected. A 7-character prefix is ambiguous today and can
become *more* ambiguous as a repository grows, so it is not immutable and must
not be accepted as though it were.

## Remote model code is not trusted by default

If a model's `config.json` contains an `auto_map`, loading it executes code from
the model repository. The resolver records this as
`requires_remote_code: true`, and the planner **rejects** the plan unless the
recipe opts in explicitly with `model.trust_remote_code: true`.

The check happens during planning, from metadata, before anything is
downloaded. A trust decision made after the download is a trust decision made
too late.

## What is not built yet

The following are stated as intent, not as implemented behaviour:

- filesystem, network, and GPU sandboxing of the plugin subprocess
- a default of no network during transformation once inputs are materialised
- a software bill of materials for published artifacts

The subprocess boundary and the pinning rules above are real today. The
sandboxing is not. Do not run an untrusted plugin and assume otherwise.


## Execution binding (transport manifest v0.2)

The capability manifest and recipe identify the **upstream quantizer**. The
transport manifest identifies the **adapter package**, its declared source pin,
and its command. They are deliberately different implementations. Transport
v0.2 adds an explicit upstream version and canonical implementation reference.
`ExecutionBinding.create` rejects disagreements before plugin launch; the smoke
job creates the binding before downloading weights. The effective interpreter
is resolved once from the supported `python -m module` command and recorded with
the stage, capability, and transport digests in invocation provenance.

Existing v0.1 serializers and identities are unchanged. v0.2 transport requires
an execution binding on every runner invocation. A binding demonstrates metadata
agreement and installed package-version agreement, not that installed code
matches a source commit, nor semantic conformance, sandboxing, or hardware
support. Source attestation remains unimplemented and must not be inferred.
