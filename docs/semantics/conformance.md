# Scoped conformance reports

`lazybrick conformance profile` prints the bundled numerical descriptor and hash.
`lazybrick conformance reference --output report.json` executes the CPU oracle
against the hand-calculated suite, writes a new report (never overwriting), and
prints its digest and binding context. The report is labeled `reference_evaluator`;
this is not third-party AWQ execution or hardware evidence.

A caller can explicitly execute a trusted candidate through the Python
`run_conformance` API. It validates the contract before execution, gives each
case a fresh copy, records errors without leaking exception messages, and keeps
all case outputs. It is not an untrusted-code sandbox or automatic plugin loader.

Verification requires the externally supplied report digest and context:

```
lazybrick conformance verify report.json --expected-digest <sha256> --context trusted-context.json
```

Use the context/digest from a trusted prior run or registry, not values read back
from the untrusted report being checked. Verification recomputes every case and
summary status against the packaged suite and rejects substitutions of profile,
suite, implementation, environment, comparison policy or attempt binding. A
failed candidate report can be intact but remains failed (CLI exit 4). Invalid
or unverifiable records return exit 5. No verification authenticates a publisher
or proves that a self-reported external run happened.

`verify_bundle_conformance` first requires full bundle integrity under an external
bundle digest, then checks the separately anchored report and its exact resolved
stage/run/attempt binding. Without the separately maintained `runs.bundle`
verifier, it fails with `bundle_integrity_unavailable`. Standalone reports do not
need run/artifact identities; bundle-attached reports must carry all five run
identity fields plus stage ID/digest. A passed reference suite never promotes
whole-model evidence or an AWQ runtime-support claim.
