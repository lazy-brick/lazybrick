"""Resolve mutable references to immutable ones, using metadata only.

Planning must never download weights. Everything here reads Hugging Face
metadata -- the revision endpoint and ``config.json`` -- which is a few
kilobytes per model, and caches the result so a plan can be re-run offline.

The transport is injected. Tests use :class:`RecordedTransport` over fixtures
recorded from the real API, so the whole suite runs on CPU with no network.

Core stays dependency-light on purpose: this speaks to the HTTP API through
``urllib`` rather than pulling in ``huggingface_hub``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol
import urllib.error
import urllib.request

from lazybrick.errors import ValidationError, ValidationIssue
from lazybrick.records import DatasetRef, ExecutionPlan, ModelRef
from lazybrick.schema import is_immutable_revision

__all__ = [
    "HttpTransport",
    "OfflineTransport",
    "RecordedTransport",
    "ResolutionError",
    "ResolvedDataset",
    "ResolvedModel",
    "ResolvedRecipe",
    "Resolver",
    "ResolverCache",
]

DEFAULT_ENDPOINT = "https://huggingface.co"

_URI = re.compile(r"\A(?P<scheme>[a-z][a-z0-9+.-]*)://(?P<repo>[^/]+/[^/]+)\Z")

#: Config keys that mean the architecture has a component AWQ-for-text cannot
#: see. Kept explicit rather than inferred, so a new modality fails loudly.
_COMPONENT_KEYS = (
    ("vision_config", "vision_encoder"),
    ("audio_config", "audio_encoder"),
    ("vision_tower_config", "vision_encoder"),
)
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


class ResolutionError(ValidationError):
    """Raised when a reference cannot be pinned to an immutable revision."""

    summary = "Cannot resolve recipe references"


# --------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------


class Transport(Protocol):
    """Fetches bytes for a URL. The only thing that touches the network."""

    def fetch(self, url: str) -> bytes: ...


class HttpTransport:
    """Reads the public Hugging Face HTTP API.

    ``HF_TOKEN`` is forwarded when set, so gated repositories resolve for a user
    who already has access. It is never written to the cache or a plan.
    """

    def __init__(self, token: str | None = None, timeout: int = 30) -> None:
        self._token = token if token is not None else os.environ.get("HF_TOKEN")
        self._timeout = timeout

    def fetch(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "lazybrick"})
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            code = {401: "unauthorized", 403: "forbidden", 404: "not_found"}.get(
                error.code, "http_error"
            )
            raise ResolutionError(
                [
                    ValidationIssue(
                        url,
                        code,
                        f"HTTP {error.code} fetching metadata"
                        + (
                            "; the repository may be gated or private. Set HF_TOKEN."
                            if error.code in (401, 403)
                            else ""
                        ),
                    )
                ]
            ) from error
        except urllib.error.URLError as error:
            raise ResolutionError(
                [ValidationIssue(url, "network_error", f"cannot reach host: {error.reason}")]
            ) from error


class OfflineTransport:
    """Refuses every fetch. Used when ``--offline`` is set and nothing is cached."""

    def fetch(self, url: str) -> bytes:
        raise ResolutionError(
            [
                ValidationIssue(
                    url,
                    "offline_unresolved",
                    "offline mode is on and this reference is not in the cache",
                )
            ]
        )


class RecordedTransport:
    """Serves recorded responses. Test-only, and deliberately strict.

    An unrecorded URL is an error rather than a fallthrough to the network, so a
    test can never accidentally start depending on the internet.
    """

    def __init__(self, responses: Mapping[str, bytes]) -> None:
        self._responses = dict(responses)
        self.requested: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.requested.append(url)
        if url not in self._responses:
            raise ResolutionError(
                [ValidationIssue(url, "not_recorded", "no recorded response")]
            )
        return self._responses[url]


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class ResolverCache:
    """A tiny on-disk cache of resolved metadata.

    Keyed by URI *and* requested revision, because ``main`` today and ``main``
    tomorrow are different answers and must not share an entry.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
            root = Path(base) / "lazybrick" / "resolve"
        self.root = Path(root)

    def _path(self, uri: str, revision: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", f"{uri}@{revision}")
        return self.root / f"{safe}.json"

    def get(self, uri: str, revision: str) -> dict[str, Any] | None:
        path = self._path(uri, revision)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt cache entry is a cache miss, never a hard failure.
            return None

    def put(self, uri: str, revision: str, payload: Mapping[str, Any]) -> None:
        path = self._path(uri, revision)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)


# --------------------------------------------------------------------------
# Resolved records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Everything a plan needs to know about a model, without its weights."""

    uri: str
    repo_id: str
    requested_revision: str
    revision: str
    model_type: str
    architectures: tuple[str, ...]
    model_profile: str
    components: tuple[str, ...]
    weight_format: str
    dtype: str | None = None
    parameter_count: int | None = None
    license: str | None = None
    requires_remote_code: bool = False
    gated: bool = False

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "uri": self.uri,
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "revision": self.revision,
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "model_profile": self.model_profile,
            "components": list(self.components),
            "weight_format": self.weight_format,
            "requires_remote_code": self.requires_remote_code,
            "gated": self.gated,
        }
        for name in ("dtype", "parameter_count", "license"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ResolvedModel:
        return cls(
            uri=data["uri"],
            repo_id=data["repo_id"],
            requested_revision=data["requested_revision"],
            revision=data["revision"],
            model_type=data["model_type"],
            architectures=tuple(data.get("architectures") or ()),
            model_profile=data["model_profile"],
            components=tuple(data.get("components") or ()),
            weight_format=data["weight_format"],
            dtype=data.get("dtype"),
            parameter_count=data.get("parameter_count"),
            license=data.get("license"),
            requires_remote_code=bool(data.get("requires_remote_code", False)),
            gated=bool(data.get("gated", False)),
        )


@dataclass(frozen=True, slots=True)
class ResolvedDataset:
    uri: str
    repo_id: str
    requested_revision: str
    revision: str
    license: str | None = None

    def to_json(self) -> dict[str, Any]:
        result = {
            "uri": self.uri,
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "revision": self.revision,
        }
        if self.license is not None:
            result["license"] = self.license
        return result

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ResolvedDataset:
        return cls(
            uri=data["uri"],
            repo_id=data["repo_id"],
            requested_revision=data["requested_revision"],
            revision=data["revision"],
            license=data.get("license"),
        )


@dataclass(frozen=True, slots=True)
class ResolvedRecipe:
    """The authored recipe with every reference pinned.

    Written to ``resolved_recipe.json``, kept separate from the authored recipe
    so that the two can always be diffed: one is what a human wrote, the other
    is what it meant at resolution time.
    """

    recipe_digest: str
    recipe: Mapping[str, Any]
    model: ResolvedModel
    calibration_dataset: ResolvedDataset | None = None
    evaluation_dataset: ResolvedDataset | None = None

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "resolved_recipe_version": "0.1",
            "recipe_digest": self.recipe_digest,
            "recipe": dict(self.recipe),
            "model": self.model.to_json(),
        }
        if self.calibration_dataset is not None:
            result["calibration_dataset"] = self.calibration_dataset.to_json()
        if self.evaluation_dataset is not None:
            result["evaluation_dataset"] = self.evaluation_dataset.to_json()
        return result

    @property
    def plan(self) -> ExecutionPlan:
        """The execution plan built from *resolved* references."""

        return ExecutionPlan.from_recipe(self.recipe, self.recipe_digest)


# --------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------


def _split_uri(uri: str, path: str, expected_scheme: str) -> str:
    match = _URI.match(uri)
    if match is None or match.group("scheme") != expected_scheme:
        raise ResolutionError(
            [
                ValidationIssue(
                    path,
                    "unsupported_scheme",
                    f"expected a '{expected_scheme}://owner/name' URI, got {uri!r}",
                )
            ]
        )
    return match.group("repo")


def classify(config: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Derive a model profile and component list from ``config.json``.

    Component-level classification is what lets the planner say *which part* of
    a model has no quantization path, instead of rejecting the whole thing with
    an unexplained "unsupported".
    """

    components: list[str] = ["language_backbone"]

    for key, component in _COMPONENT_KEYS:
        if key in config and component not in components:
            components.append(component)

    model_type = str(config.get("model_type", ""))
    is_moe = model_type.endswith(("_moe", "-moe")) or any(
        key in config for key in _MOE_KEYS
    )
    if is_moe:
        components.append("moe_experts")

    multimodal = any(
        component in components for component in ("vision_encoder", "audio_encoder")
    )
    if multimodal:
        profile = "multimodal-decoder"
    elif is_moe:
        profile = "moe-decoder"
    else:
        profile = "dense-decoder"

    return profile, tuple(components)


class Resolver:
    """Turns authored references into pinned, inspected ones."""

    def __init__(
        self,
        transport: Transport | None = None,
        cache: ResolverCache | None = None,
        *,
        offline: bool = False,
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        if transport is None:
            transport = OfflineTransport() if offline else HttpTransport()
        self._transport = transport
        self._cache = cache if cache is not None else ResolverCache()
        self._offline = offline
        self._endpoint = endpoint.rstrip("/")

    # -- low level ---------------------------------------------------------

    def _get_json(self, url: str) -> dict[str, Any]:
        raw = self._transport.fetch(url)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ResolutionError(
                [ValidationIssue(url, "invalid_response", f"response is not JSON: {error}")]
            ) from error
        if not isinstance(payload, dict):
            raise ResolutionError(
                [ValidationIssue(url, "invalid_response", "expected a JSON object")]
            )
        return payload

    # -- models ------------------------------------------------------------

    def resolve_model(self, ref: ModelRef, path: str = "model") -> ResolvedModel:
        cached = self._cache.get(ref.uri, ref.revision)
        if cached is not None:
            return ResolvedModel.from_json(cached)
        if self._offline:
            raise ResolutionError(
                [
                    ValidationIssue(
                        path,
                        "offline_unresolved",
                        f"offline mode is on and {ref.uri}@{ref.revision} is not "
                        "cached; run once with network access to populate it",
                    )
                ]
            )

        repo_id = _split_uri(ref.uri, f"{path}.uri", "hf")
        info = self._get_json(
            f"{self._endpoint}/api/models/{repo_id}/revision/{ref.revision}"
        )

        revision = info.get("sha")
        if not is_immutable_revision(revision):
            raise ResolutionError(
                [
                    ValidationIssue(
                        f"{path}.revision",
                        "unresolvable_revision",
                        f"the API did not return a commit SHA for "
                        f"{repo_id}@{ref.revision}",
                    )
                ]
            )

        config = self._get_json(
            f"{self._endpoint}/{repo_id}/resolve/{revision}/config.json"
        )
        profile, components = classify(config)
        siblings = [s.get("rfilename", "") for s in info.get("siblings") or ()]
        safetensors = info.get("safetensors") or {}

        resolved = ResolvedModel(
            uri=ref.uri,
            repo_id=repo_id,
            requested_revision=ref.revision,
            revision=revision,
            model_type=str(config.get("model_type", "")),
            architectures=tuple(config.get("architectures") or ()),
            model_profile=profile,
            components=components,
            weight_format=(
                "safetensors"
                if any(name.endswith(".safetensors") for name in siblings)
                else "pytorch"
            ),
            dtype=config.get("torch_dtype") or config.get("dtype"),
            parameter_count=safetensors.get("total"),
            license=(info.get("cardData") or {}).get("license"),
            # An auto_map means loading the model executes code from the repo.
            requires_remote_code="auto_map" in config,
            gated=bool(info.get("gated")),
        )
        self._cache.put(ref.uri, ref.revision, resolved.to_json())
        return resolved

    # -- datasets ----------------------------------------------------------

    def resolve_dataset(self, ref: DatasetRef, path: str = "dataset") -> ResolvedDataset:
        cached = self._cache.get(ref.uri, ref.revision)
        if cached is not None:
            return ResolvedDataset.from_json(cached)
        if self._offline:
            raise ResolutionError(
                [
                    ValidationIssue(
                        path,
                        "offline_unresolved",
                        f"offline mode is on and {ref.uri}@{ref.revision} is not cached",
                    )
                ]
            )

        repo_id = _split_uri(ref.uri, f"{path}.uri", "hf-dataset")
        info = self._get_json(
            f"{self._endpoint}/api/datasets/{repo_id}/revision/{ref.revision}"
        )
        revision = info.get("sha")
        if not is_immutable_revision(revision):
            raise ResolutionError(
                [
                    ValidationIssue(
                        f"{path}.revision",
                        "unresolvable_revision",
                        f"the API did not return a commit SHA for "
                        f"{repo_id}@{ref.revision}",
                    )
                ]
            )

        resolved = ResolvedDataset(
            uri=ref.uri,
            repo_id=repo_id,
            requested_revision=ref.revision,
            revision=revision,
            license=(info.get("cardData") or {}).get("license"),
        )
        self._cache.put(ref.uri, ref.revision, resolved.to_json())
        return resolved

    # -- whole recipes -----------------------------------------------------

    def resolve_recipe(
        self, recipe: Mapping[str, Any], recipe_digest: str
    ) -> ResolvedRecipe:
        """Pin every reference in ``recipe`` and return the resolved form.

        Collects failures across sections so one call reports every unresolvable
        reference, not just the first.
        """

        issues: list[ValidationIssue] = []
        pinned: dict[str, Any] = json.loads(json.dumps(recipe))

        model: ResolvedModel | None = None
        try:
            model = self.resolve_model(ModelRef.from_json(recipe["model"]))
            pinned["model"]["revision"] = model.revision
        except ResolutionError as error:
            issues.extend(error.issues)

        datasets: dict[str, ResolvedDataset | None] = {
            "calibration": None,
            "evaluation": None,
        }
        for section in ("calibration", "evaluation"):
            block = recipe.get(section)
            if not block:
                continue
            try:
                resolved = self.resolve_dataset(
                    DatasetRef.from_json(block["dataset"]), f"{section}.dataset"
                )
            except ResolutionError as error:
                issues.extend(error.issues)
                continue
            datasets[section] = resolved
            pinned[section]["dataset"]["revision"] = resolved.revision

        issues.extend(_unpinned_implementations(recipe))

        if issues:
            raise ResolutionError(issues)

        assert model is not None  # no issues implies the model resolved
        return ResolvedRecipe(
            recipe_digest=recipe_digest,
            recipe=pinned,
            model=model,
            calibration_dataset=datasets["calibration"],
            evaluation_dataset=datasets["evaluation"],
        )


def _unpinned_implementations(recipe: Mapping[str, Any]) -> list[ValidationIssue]:
    """Plugin implementations are pinned by the author, never by the resolver.

    There is nothing to look up: a recipe either names an immutable commit or
    container digest, or it is not executable. The schema already rejects
    obviously mutable ones; this catches a plan assembled programmatically.
    """

    from lazybrick.records import ImplementationRef

    issues: list[ValidationIssue] = []
    for index, stage in enumerate(recipe.get("stages") or ()):
        implementation = stage.get("implementation") or {}
        if not ImplementationRef.from_json(implementation).is_pinned:
            issues.append(
                ValidationIssue(
                    f"stages[{index}].implementation",
                    "mutable_reference",
                    "plugin implementation is not pinned to an immutable commit "
                    "or container digest",
                )
            )
    return issues
