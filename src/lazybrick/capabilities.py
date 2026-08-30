"""Capability vocabulary and the compatibility check.

The schema decides whether a recipe is well-formed. This module decides whether
it can actually run, by intersecting what each participant declares:

    model  ∩  plugin  ∩  exporter  ∩  runtime  ∩  hardware

If any intersection is empty the plan is rejected *before* a weight is
downloaded or a GPU is allocated. Every rejection carries a stable reason code
and an explanation naming the participant that blocked it, because "unsupported"
on its own tells a user nothing they can act on.

The vocabulary is deliberately small. A controlled vocabulary that everyone
declares against is what makes the intersection meaningful; free-form strings
would turn it into a spelling contest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any, Final

from lazybrick.records import (
    CapabilityReport,
    ExecutionPlan,
    PluginManifest,
    StageSpec,
)
from lazybrick.resolve import ResolvedModel

__all__ = [
    "CAPABILITY_KEYS",
    "CompatibilityResult",
    "HardwareProfile",
    "Reason",
    "check_compatibility",
    "hardware_report",
    "model_report",
    "runtime_report",
    "scheme_for_stage",
]

#: The controlled vocabulary. Every participant declares against these keys.
CAPABILITY_KEYS: Final = (
    "model_profile",
    "architecture",
    "component",
    "input_format",
    "output_format",
    "quantization_scheme",
    "runtime",
    "accelerator_vendor",
    "compute_capability",
    "calibration_required",
    "remote_code_required",
)

_COMPUTE_CAPABILITY = re.compile(r"\A(\d+)\.(\d+)\Z")

#: What LazyBrick believes each runtime can load.
#:
#: These are *claims*, and they are only as good as the runtime validation that
#: checks them (to-do M1 §4). A claim here that vLLM never verifies is exactly
#: the kind of unearned assertion this project exists to avoid, so keep the list
#: narrow and let it grow when a real load proves an entry.
RUNTIME_CAPABILITIES: Final = {
    "vllm": {
        "input_format": ("compressed-tensors", "safetensors"),
        "quantization_scheme": ("W4A16",),
        "accelerator_vendor": ("nvidia",),
        "model_profile": ("dense-decoder", "moe-decoder", "multimodal-decoder"),
    }
}

#: accelerator_family -> vendor.
_VENDORS: Final = {
    "nvidia-cuda": "nvidia",
    "amd-rocm": "amd",
    "apple-metal": "apple",
}


@dataclass(frozen=True, slots=True)
class Reason:
    """Why a plan was rejected, in a form both a human and a script can use."""

    code: str
    subject: str
    detail: str
    capability: str | None = None
    required: tuple[str, ...] = ()
    available: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.subject}: {self.detail} [{self.code}]"

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "subject": self.subject,
            "detail": self.detail,
            "required": list(self.required),
            "available": list(self.available),
        }
        if self.capability is not None:
            result["capability"] = self.capability
        return result


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """The outcome of the intersection, and the evidence behind it."""

    accepted: bool
    reasons: tuple[Reason, ...] = ()
    reports: tuple[CapabilityReport, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)

    def to_json(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": [reason.to_json() for reason in self.reasons],
            "reports": [report.to_json() for report in self.reports],
        }


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """What a machine actually has, as opposed to what a recipe asks for.

    ``TargetSpec`` is the requirement; this is the reality. Keeping them
    separate is what allows a plan to be checked on a laptop against a
    description of the GPU it will eventually run on.
    """

    vendor: str
    device_count: int
    compute_capability: str
    memory_gib: int
    name: str | None = None

    @classmethod
    def none(cls) -> HardwareProfile:
        """No accelerator. The honest default when nothing was detected."""

        return cls(vendor="none", device_count=0, compute_capability="0.0", memory_gib=0)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "vendor": self.vendor,
            "device_count": self.device_count,
            "compute_capability": self.compute_capability,
            "memory_gib": self.memory_gib,
        }
        if self.name is not None:
            result["name"] = self.name
        return result

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> HardwareProfile:
        return cls(
            vendor=data["vendor"],
            device_count=int(data["device_count"]),
            compute_capability=str(data["compute_capability"]),
            memory_gib=int(data["memory_gib"]),
            name=data.get("name"),
        )


def _capability_tuple(value: int | tuple[int, int] | str) -> tuple[int, int]:
    match = _COMPUTE_CAPABILITY.match(str(value))
    if match is None:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def scheme_for_stage(stage: StageSpec) -> str:
    """Derive the quantization scheme a stage asks for.

    v0.1 is weight-only, so activations stay at the model dtype and the scheme
    is ``W<bits>A16``. An explicit ``scheme`` parameter wins, so a plugin that
    needs to say something this rule cannot express is not blocked by it.
    """

    explicit = stage.parameters.get("scheme")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    bits = stage.parameters.get("weight_bits")
    if isinstance(bits, int) and not isinstance(bits, bool):
        return f"W{bits}A16"
    return "unknown"


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def model_report(model: ResolvedModel) -> CapabilityReport:
    return CapabilityReport(
        subject="model",
        capabilities={
            "model_profile": (model.model_profile,),
            "architecture": model.architectures,
            "component": model.components,
            "input_format": (model.weight_format,),
            "remote_code_required": ("true" if model.requires_remote_code else "false",),
        },
    )


def hardware_report(hardware: HardwareProfile) -> CapabilityReport:
    return CapabilityReport(
        subject="hardware",
        capabilities={
            "accelerator_vendor": (hardware.vendor,),
            "compute_capability": (hardware.compute_capability,),
        },
    )


def runtime_report(runtime: str) -> CapabilityReport:
    declared = RUNTIME_CAPABILITIES.get(runtime, {})
    return CapabilityReport(subject=f"runtime:{runtime}", capabilities=dict(declared))


def plugin_report(manifest: PluginManifest) -> CapabilityReport:
    return CapabilityReport(
        subject=f"plugin:{manifest.name}", capabilities=dict(manifest.capabilities)
    )


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------


class _Rejections:
    def __init__(self) -> None:
        self.reasons: list[Reason] = []

    def add(self, reason: Reason) -> None:
        self.reasons.append(reason)

    def intersect(
        self,
        *,
        code: str,
        subject: str,
        capability: str,
        required: Iterable[str],
        available: Iterable[str],
        detail: str,
    ) -> bool:
        """Require a non-empty intersection, recording the sets on failure."""

        required = tuple(required)
        available = tuple(available)
        if set(required) & set(available):
            return True
        self.add(
            Reason(
                code=code,
                subject=subject,
                detail=detail,
                capability=capability,
                required=required,
                available=available,
            )
        )
        return False


def check_compatibility(
    plan: ExecutionPlan,
    model: ResolvedModel,
    plugins: Mapping[str, PluginManifest] | Sequence[PluginManifest],
    hardware: HardwareProfile | None = None,
) -> CompatibilityResult:
    """Intersect every participant's capabilities and explain any empty result.

    ``hardware`` describes the machine the artifact will be built on. Omitting
    it means "nothing detected", which is a rejection rather than a pass: a plan
    that silently assumes a GPU is how a build fails an hour later instead of
    immediately.
    """

    if not isinstance(plugins, Mapping):
        plugins = {manifest.name: manifest for manifest in plugins}
    hardware = hardware if hardware is not None else HardwareProfile.none()

    rejections = _Rejections()
    reports = [model_report(model), hardware_report(hardware)]

    # -- references must be immutable before anything else matters ----------
    if not model.is_pinned:
        rejections.add(
            Reason(
                code="mutable_reference",
                subject="model",
                detail=(
                    f"{model.uri} is pinned to {model.requested_revision!r}, which "
                    "can move. Resolve the recipe before planning, so the plan "
                    "records the commit it actually means"
                ),
            )
        )

    # -- trust ---------------------------------------------------------------
    if model.requires_remote_code and not plan.model.trust_remote_code:
        rejections.add(
            Reason(
                code="remote_code_required",
                subject="model",
                detail=(
                    f"{model.repo_id} declares an auto_map, so loading it executes "
                    "code from the repository. Set model.trust_remote_code: true "
                    "to accept that explicitly, or choose a model that does not "
                    "need it"
                ),
                capability="remote_code_required",
            )
        )

    # -- per stage -----------------------------------------------------------
    for index, stage in enumerate(plan.stages):
        plugin_name = stage.plugin.rsplit("/", 1)[-1]
        manifest = plugins.get(plugin_name) or plugins.get(stage.plugin)
        if manifest is None:
            rejections.add(
                Reason(
                    code="unknown_plugin",
                    subject=f"stages[{index}]",
                    detail=(
                        f"no manifest for plugin {stage.plugin!r}; it must be "
                        "discoverable before a plan can be checked"
                    ),
                )
            )
            continue

        report = plugin_report(manifest)
        reports.append(report)
        subject = report.subject

        rejections.intersect(
            code="unsupported_model_profile",
            subject=subject,
            capability="model_profile",
            required=(model.model_profile,),
            available=report.get("model_profile"),
            detail=(
                f"{manifest.name} does not support the {model.model_profile} "
                f"profile that {model.repo_id} resolves to"
            ),
        )

        # Component-level, so the rejection names the part that has no path.
        supported_components = set(report.get("component"))
        for component in model.components:
            if component not in supported_components:
                rejections.add(
                    Reason(
                        code="unsupported_component",
                        subject=subject,
                        detail=(
                            f"{manifest.name} supports the language backbone but "
                            f"has no declared quantization path for the "
                            f"{component.replace('_', ' ')} required by "
                            f"{model.repo_id}"
                        ),
                        capability="component",
                        required=(component,),
                        available=tuple(sorted(supported_components)),
                    )
                )

        scheme = scheme_for_stage(stage)
        rejections.intersect(
            code="unsupported_quantization_scheme",
            subject=subject,
            capability="quantization_scheme",
            required=(scheme,),
            available=report.get("quantization_scheme"),
            detail=f"{manifest.name} does not implement {scheme}",
        )

        rejections.intersect(
            code="unsupported_export_format",
            subject=subject,
            capability="output_format",
            required=(plan.export.format,),
            available=report.get("output_format"),
            detail=(
                f"{manifest.name} cannot export {plan.export.format!r}"
            ),
        )

        if manifest.requires.get("calibration") and plan.calibration is None:
            rejections.add(
                Reason(
                    code="calibration_required",
                    subject=subject,
                    detail=(
                        f"{manifest.name} requires calibration data, and the "
                        "recipe has no calibration section"
                    ),
                    capability="calibration_required",
                )
            )

    # -- runtime -------------------------------------------------------------
    runtime = plan.export.runtime
    runtime_capabilities = runtime_report(runtime)
    reports.append(runtime_capabilities)

    if runtime not in RUNTIME_CAPABILITIES:
        rejections.add(
            Reason(
                code="unsupported_runtime",
                subject=f"runtime:{runtime}",
                detail=(
                    f"{runtime!r} is not a runtime LazyBrick knows how to load an "
                    "artifact into; known runtimes are "
                    + ", ".join(sorted(RUNTIME_CAPABILITIES))
                ),
                capability="runtime",
                required=(runtime,),
                available=tuple(sorted(RUNTIME_CAPABILITIES)),
            )
        )
    else:
        rejections.intersect(
            code="unsupported_input_format",
            subject=runtime_capabilities.subject,
            capability="input_format",
            required=(plan.export.format,),
            available=runtime_capabilities.get("input_format"),
            detail=f"{runtime} cannot load {plan.export.format!r} artifacts",
        )
        rejections.intersect(
            code="unsupported_model_profile",
            subject=runtime_capabilities.subject,
            capability="model_profile",
            required=(model.model_profile,),
            available=runtime_capabilities.get("model_profile"),
            detail=f"{runtime} does not serve {model.model_profile} models",
        )

    if plan.target.runtime != runtime:
        rejections.add(
            Reason(
                code="runtime_mismatch",
                subject="target",
                detail=(
                    f"the recipe exports for {runtime!r} but targets "
                    f"{plan.target.runtime!r}; an artifact built for one runtime "
                    "is not evidence about another"
                ),
                capability="runtime",
                required=(runtime,),
                available=(plan.target.runtime,),
            )
        )

    # -- hardware ------------------------------------------------------------
    _check_hardware(plan, hardware, rejections)

    return CompatibilityResult(
        accepted=not rejections.reasons,
        reasons=tuple(rejections.reasons),
        reports=tuple(reports),
    )


def _check_hardware(
    plan: ExecutionPlan, hardware: HardwareProfile, rejections: _Rejections
) -> None:
    target = plan.target
    wanted_vendor = _VENDORS.get(target.accelerator_family, target.accelerator_family)

    if hardware.vendor == "none" or hardware.device_count < 1:
        rejections.add(
            Reason(
                code="missing_accelerator",
                subject="hardware",
                detail=(
                    f"the recipe needs a {target.accelerator_family} accelerator "
                    "and none was detected. Pass --target with a description of "
                    "the machine that will run the build"
                ),
                capability="accelerator_vendor",
                required=(wanted_vendor,),
                available=(),
            )
        )
        return

    if hardware.vendor != wanted_vendor:
        rejections.add(
            Reason(
                code="accelerator_vendor_mismatch",
                subject="hardware",
                detail=(
                    f"the recipe needs {wanted_vendor}, the machine has "
                    f"{hardware.vendor}"
                ),
                capability="accelerator_vendor",
                required=(wanted_vendor,),
                available=(hardware.vendor,),
            )
        )

    if _capability_tuple(hardware.compute_capability) < _capability_tuple(
        target.min_compute_capability
    ):
        rejections.add(
            Reason(
                code="insufficient_compute_capability",
                subject="hardware",
                detail=(
                    f"the recipe needs compute capability "
                    f"{target.min_compute_capability} or higher, the machine "
                    f"reports {hardware.compute_capability}"
                ),
                capability="compute_capability",
                required=(target.min_compute_capability,),
                available=(hardware.compute_capability,),
            )
        )

    if hardware.device_count < target.device_count:
        rejections.add(
            Reason(
                code="insufficient_devices",
                subject="hardware",
                detail=(
                    f"the recipe needs {target.device_count} device(s), the "
                    f"machine has {hardware.device_count}"
                ),
            )
        )

    if hardware.memory_gib < target.min_memory_gib:
        rejections.add(
            Reason(
                code="insufficient_memory",
                subject="hardware",
                detail=(
                    f"the recipe needs {target.min_memory_gib} GiB of accelerator "
                    f"memory, the machine has {hardware.memory_gib} GiB"
                ),
            )
        )
