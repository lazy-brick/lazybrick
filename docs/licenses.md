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
| vLLM LLM Compressor | Apache-2.0 | the isolated AWQ adapter and smoke workflow |
| vLLM | Apache-2.0 | the isolated smoke-workflow validator |
| HuggingFaceH4/ultrachat_200k | MIT (dataset metadata) | pinned smoke calibration and evaluation input |

These dependencies are optional and do not ship in the dependency-light core.

## Calibration dataset decision

The smoke workflow pins `HuggingFaceH4/ultrachat_200k` at commit
`8049631c405ae6576f93f445c6b8166f76f5505a`; its Hub metadata reports MIT. This
is a recorded project input, not a legal conclusion that every resulting model
artifact may be redistributed. Model terms, dataset content rights, and the
intended use still need an independent review before publication.

The generic example continues to use synthetic fixture metadata; the concrete
dataset pin is limited to the opt-in smoke workflow.
