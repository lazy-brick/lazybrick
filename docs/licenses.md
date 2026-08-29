# Licensing

LazyBrick's own source is Apache-2.0. That licence covers **this repository and
nothing else**.

Models, datasets, runtimes, and plugin implementations keep their own licences.
Nothing in LazyBrick relicenses them, and running a model through LazyBrick does
not change what you are permitted to do with the result.

## What LazyBrick records

Resolution captures the licence of each external input from its Hub metadata:

| Input | Recorded as |
| ----- | ----------- |
| model | `ResolvedModel.license` |
| calibration dataset | `ResolvedDataset.license` |
| evaluation dataset | `ResolvedDataset.license` |
| plugin and implementation | `PluginManifest.licenses` |

These are recorded so that a published artifact can state its inputs' licences
accurately. They are recorded, not interpreted: LazyBrick does not decide
whether a combination is permissible for your use, and reporting a licence
string is not legal advice.

## Third-party components

| Component | Licence | Used for |
| --------- | ------- | -------- |
| PyYAML | MIT | reading YAML recipes |
| Qwen3 models | Apache-2.0 | the reference and smoke models |
| vLLM LLM Compressor | Apache-2.0 | the AWQ implementation (not yet integrated) |
| vLLM | Apache-2.0 | the target runtime (not yet integrated) |

The last two are listed because recipes reference them, not because LazyBrick
ships or executes them today.

## The calibration dataset is undecided

No calibration dataset has been chosen. This is a **blocking decision**, not an
oversight: the dataset must permit calibration use, redistribution of the
derived artifact, and publication of the resulting evidence, and its licence has
to be compatible with distributing that evidence alongside Apache-2.0 code.

Until it is settled:

- `examples/qwen3-awq.yaml` carries a placeholder dataset URI and will not plan
- the test fixture for a dataset is hand-written and labelled synthetic, so that
  no fixture implies a choice that has not been made

If no clearly compatible dataset is found, the correct outcome is to stop and
escalate, not to substitute one whose terms are unclear.
