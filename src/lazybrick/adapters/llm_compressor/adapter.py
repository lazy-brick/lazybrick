"""Strict W4A16 AWQ translation and execution through LLM Compressor."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


class AdapterInputError(ValueError):
    """Raised before GPU work when adapter inputs violate the first-slice contract."""


@dataclass(frozen=True, slots=True)
class AWQSettings:
    """The deliberately narrow AWQ surface supported by the first vertical slice."""

    weight_bits: int = 4
    group_size: int = 128
    symmetric: bool = False
    targets: tuple[str, ...] = ("Linear",)
    ignore: tuple[str, ...] = ("lm_head",)

    @classmethod
    def from_mapping(cls, value: object) -> AWQSettings:
        if not isinstance(value, Mapping):
            raise AdapterInputError("settings must be an object")
        allowed = {"weight_bits", "group_size", "symmetric", "targets", "ignore"}
        unknown = set(value) - allowed
        if unknown:
            raise AdapterInputError(f"unknown AWQ settings: {', '.join(sorted(unknown))}")
        settings = cls(
            weight_bits=value.get("weight_bits", 4),
            group_size=value.get("group_size", 128),
            symmetric=value.get("symmetric", False),
            targets=_string_tuple(value.get("targets", ["Linear"]), "targets"),
            ignore=_string_tuple(value.get("ignore", ["lm_head"]), "ignore"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if isinstance(self.weight_bits, bool) or self.weight_bits != 4:
            raise AdapterInputError("first-slice AWQ requires weight_bits=4")
        if isinstance(self.group_size, bool) or self.group_size != 128:
            raise AdapterInputError("first-slice AWQ requires group_size=128")
        if self.symmetric is not False:
            raise AdapterInputError("first-slice AWQ requires asymmetric weights")
        if self.targets != ("Linear",):
            raise AdapterInputError("first-slice AWQ targets must be exactly ['Linear']")
        if self.ignore != ("lm_head",):
            raise AdapterInputError("first-slice AWQ ignore list must be exactly ['lm_head']")


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise AdapterInputError(f"{field} must be a non-empty-string list")
    return tuple(value)


def recipe_spec(settings: AWQSettings) -> list[dict[str, Any]]:
    """Return the exact serializable LLM Compressor recipe being requested."""

    settings.validate()
    config_group = {
        "targets": list(settings.targets),
        "input_activations": None,
        "output_activations": None,
        "weights": {
            "num_bits": settings.weight_bits,
            "type": "int",
            "symmetric": settings.symmetric,
            "strategy": "group",
            "group_size": settings.group_size,
        },
    }
    return [
        {
            "modifier": "AWQModifier",
            "config_groups": {"group_0": config_group},
            "ignore": list(settings.ignore),
        },
        {
            "modifier": "QuantizationModifier",
            "config_groups": {"group_0": config_group},
            "ignore": list(settings.ignore),
        },
    ]


def build_llm_compressor_recipe(
    settings: AWQSettings,
    *,
    awq_factory: Callable[..., object] | None = None,
    quantization_factory: Callable[..., object] | None = None,
) -> list[object]:
    """Construct actual modifier objects, importing the optional stack lazily."""

    settings.validate()
    if awq_factory is None or quantization_factory is None:
        try:
            from llmcompressor.modifiers.quantization import QuantizationModifier
            from llmcompressor.modifiers.transform.awq import AWQModifier
        except ImportError as error:
            raise RuntimeError(
                'the AWQ adapter requires the optional dependency group: pip install "lazybrick[awq]"'
            ) from error
        awq_factory = awq_factory or AWQModifier
        quantization_factory = quantization_factory or QuantizationModifier

    config_group = recipe_spec(settings)[0]["config_groups"]
    kwargs = {"config_groups": config_group, "ignore": list(settings.ignore)}
    return [awq_factory(**kwargs), quantization_factory(**kwargs)]


def _local_directory(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AdapterInputError(f"{field} must be a local directory path")
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise AdapterInputError(f"{field} must be an existing absolute directory")
    return path.resolve()


def _calibration_texts(path: Path, expected_samples: int) -> list[str]:
    records_path = path / "calibration.jsonl"
    if not records_path.is_file():
        raise AdapterInputError(f"missing materialized calibration file: {records_path}")
    texts: list[str] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise AdapterInputError(
                    f"calibration.jsonl line {line_number} is invalid JSON"
                ) from error
            if set(record) != {"id", "text"} or not isinstance(record["id"], str) or not isinstance(
                record["text"], str
            ):
                raise AdapterInputError(
                    "each calibration record must contain only string id and text fields"
                )
            texts.append(record["text"])
    if len(texts) != expected_samples:
        raise AdapterInputError(
            f"expected {expected_samples} calibration samples, found {len(texts)}"
        )
    return texts


def execute_awq(
    payload: Mapping[str, object],
    output_dir: str | Path,
    *,
    model_loader: Callable[..., object] | None = None,
    tokenizer_loader: Callable[..., object] | None = None,
    dataset_factory: Callable[[Iterable[Mapping[str, str]]], object] | None = None,
    oneshot_fn: Callable[..., object] | None = None,
    recipe_builder: Callable[[AWQSettings], list[object]] = build_llm_compressor_recipe,
) -> dict[str, object]:
    """Execute the pinned local-input AWQ transformation without any fallback."""

    allowed = {
        "model_path",
        "calibration_path",
        "calibration_samples",
        "max_sequence_length",
        "settings",
        "export_format",
        "runtime",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise AdapterInputError(f"unknown execute fields: {', '.join(sorted(unknown))}")
    model_path = _local_directory(payload.get("model_path"), "model_path")
    calibration_path = _local_directory(payload.get("calibration_path"), "calibration_path")
    samples = payload.get("calibration_samples")
    max_length = payload.get("max_sequence_length")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise AdapterInputError("calibration_samples must be a positive integer")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise AdapterInputError("max_sequence_length must be a positive integer")
    if payload.get("export_format") != "compressed-tensors/safetensors":
        raise AdapterInputError("export_format must be compressed-tensors/safetensors")
    if payload.get("runtime") != "vllm":
        raise AdapterInputError("runtime must be vllm")
    settings = AWQSettings.from_mapping(payload.get("settings", {}))
    texts = _calibration_texts(calibration_path, samples)

    if any(item is None for item in (model_loader, tokenizer_loader, dataset_factory, oneshot_fn)):
        try:
            from datasets import Dataset
            from llmcompressor import oneshot
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                'the AWQ adapter requires the optional dependency group: pip install "lazybrick[awq]"'
            ) from error
        model_loader = model_loader or AutoModelForCausalLM.from_pretrained
        tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
        dataset_factory = dataset_factory or Dataset.from_list
        oneshot_fn = oneshot_fn or oneshot

    model = model_loader(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype="auto",
        device_map="auto",
    )
    tokenizer = tokenizer_loader(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    dataset = dataset_factory([{"text": text} for text in texts])
    recipe = recipe_builder(settings)
    oneshot_fn(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=max_length,
        num_calibration_samples=samples,
    )

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target), save_compressed=True, safe_serialization=True)
    tokenizer.save_pretrained(str(target))
    validation = validate_artifact(target)
    return {
        "algorithm": "awq",
        "scheme": "W4A16",
        "recipe": recipe_spec(settings),
        "artifact": validation,
    }


def validate_artifact(path: str | Path) -> dict[str, object]:
    """Check the minimum compressed-tensors/SafeTensors export contract."""

    target = Path(path)
    config_path = target / "config.json"
    if not config_path.is_file():
        raise AdapterInputError("export is missing config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdapterInputError("export config.json is invalid") from error
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise AdapterInputError("export config.json is missing quantization_config")
    safetensors = sorted(item.name for item in target.glob("*.safetensors"))
    if not safetensors:
        raise AdapterInputError("export contains no SafeTensors weights")
    if not (target / "tokenizer_config.json").is_file():
        raise AdapterInputError("export is missing tokenizer_config.json")
    return {
        "path": str(target.resolve()),
        "format": "compressed-tensors/safetensors",
        "weights": safetensors,
        "quantization_config": quantization,
    }
