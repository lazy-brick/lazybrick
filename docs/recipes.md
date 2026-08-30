# Recipe reference, schema v0.1

A recipe is a YAML or JSON document. The schema is strict: unknown fields and
unknown schema versions are rejected, and every problem is reported at once with
a field path and a stable reason code.

The schema checks **shape**, never **support**. `runtime: tensorrt` is
well-formed and will be rejected later, by the planner, with an explanation.
Keeping the two apart means the list of supported runtimes lives in exactly one
place instead of two that can disagree.

## Two example files

| File | Purpose |
| ---- | ------- |
| `examples/qwen3-awq.yaml` | real pinned revisions; the shape to copy |
| `examples/qwen3-awq.template.yaml` | placeholders; **does not validate**, by design |

The template is deliberately invalid. A template that validates is a template
somebody ships by accident.

`examples/qwen3-awq.yaml` pins the model and the plugin implementation to real
commits, but its calibration and evaluation datasets are placeholders, so
`lazybrick plan` refuses it. That is not an oversight -- see
[licenses.md](licenses.md#the-calibration-dataset-is-undecided).

## Fields

### `schema_version` (required)

Must be `"0.1"`. An unknown version is rejected on its own, without validating
the body: checking a document against a schema it was not written for produces
noise, not diagnosis.

### `model` (required)

| Field | Required | Notes |
| ----- | -------- | ----- |
| `uri` | yes | `hf://owner/name` |
| `revision` | yes | a commit SHA, or a branch/tag the resolver will pin |
| `trust_remote_code` | no | defaults to `false`; see [plugin-trust.md](plugin-trust.md) |

A mutable `revision` is accepted *here* on purpose. The resolver pins it, and
execution is refused if it is still mutable at plan time. Refusing at authoring
time would make it impossible to write a recipe against a branch and pin it
afterwards, which is the normal workflow.

### `stages` (required, non-empty)

| Field | Required | Notes |
| ----- | -------- | ----- |
| `id` | yes | unique within the recipe |
| `plugin` | yes | e.g. `lazybrick.plugin/awq` |
| `plugin_version` | yes | |
| `implementation` | yes | `git` + `commit`, or `container` |
| `parameters` | no | passed to the plugin |

`implementation.commit` must be a full 40-character SHA; `implementation.container`
must be pinned as `image@sha256:<64 hex>`. Branches, tags, and abbreviated SHAs
are refused.

### `calibration` (optional at the schema level)

`dataset` (`uri` + `revision`), `samples`, `seed`, `preprocessing_id`,
`max_sequence_length`.

Optional here because whether calibration is *needed* depends on the plugin. A
plugin whose manifest says `requires.calibration: true` causes the planner to
reject a recipe that omits it, with `calibration_required`.

### `export` (required)

`format` and `runtime`.

### `target` (required)

`accelerator_family`, `min_compute_capability`, `device_count`,
`min_memory_gib`, `runtime`.

`min_compute_capability` is a **string** (`"8.0"`), not a number. Floats are
banned from recipes entirely; see [identities.md](identities.md#why-floats-are-banned).

### `evaluation` (optional)

`protocol_id`, `dataset` (`uri` + `revision`), `max_samples`, `seed`.

## Reason codes

Reported by `validate` and by `plan --json`.

### Schema

`unknown_schema_version`, `unknown_field`, `missing_field`, `invalid_type`,
`invalid_value`, `duplicate_id`, `unsupported_scheme`, `mutable_reference`,
`float_not_allowed`, `invalid_key`, `not_serializable`

### Resolution

`unsupported_scheme`, `unresolvable_revision`, `offline_unresolved`,
`not_found`, `unauthorized`, `forbidden`, `network_error`, `invalid_response`

### Compatibility

`unsupported_model_profile`, `unsupported_component`,
`unsupported_quantization_scheme`, `unsupported_export_format`,
`unsupported_input_format`, `unsupported_runtime`, `runtime_mismatch`,
`unknown_plugin`, `calibration_required`, `remote_code_required`,
`mutable_reference`, `missing_accelerator`, `accelerator_vendor_mismatch`,
`insufficient_compute_capability`, `insufficient_devices`, `insufficient_memory`
