from __future__ import annotations

import json
from pathlib import Path

import pytest

from lazybrick.adapters.llm_compressor import (
    AWQSettings,
    AdapterInputError,
    build_llm_compressor_recipe,
    execute_awq,
    recipe_spec,
    validate_artifact,
)


def test_recipe_is_exact_asymmetric_w4a16_group_128() -> None:
    spec = recipe_spec(AWQSettings())

    assert [item["modifier"] for item in spec] == ["AWQModifier", "QuantizationModifier"]
    for modifier in spec:
        assert modifier["ignore"] == ["lm_head"]
        assert modifier["config_groups"]["group_0"]["weights"] == {
            "num_bits": 4,
            "type": "int",
            "symmetric": False,
            "strategy": "group",
            "group_size": 128,
        }


@pytest.mark.parametrize(
    "settings",
    [
        {"weight_bits": 8},
        {"group_size": 64},
        {"symmetric": True},
        {"targets": ["Conv2d"]},
        {"ignore": []},
    ],
)
def test_settings_never_silently_fall_back(settings: dict[str, object]) -> None:
    with pytest.raises(AdapterInputError):
        AWQSettings.from_mapping(settings)


def test_recipe_builder_uses_both_modifiers() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def awq(**kwargs: object) -> object:
        calls.append(("awq", kwargs))
        return "awq"

    def quant(**kwargs: object) -> object:
        calls.append(("quant", kwargs))
        return "quant"

    result = build_llm_compressor_recipe(
        AWQSettings(), awq_factory=awq, quantization_factory=quant
    )

    assert result == ["awq", "quant"]
    assert [name for name, _ in calls] == ["awq", "quant"]
    assert calls[0][1] == calls[1][1]


class FakeModel:
    def __init__(self) -> None:
        self.saved: dict[str, object] = {}

    def save_pretrained(self, path: str, **kwargs: object) -> None:
        self.saved = {"path": path, **kwargs}
        target = Path(path)
        (target / "config.json").write_text(
            json.dumps({"quantization_config": {"format": "pack-quantized"}}),
            encoding="utf-8",
        )
        (target / "model.safetensors").write_bytes(b"weights")


class FakeTokenizer:
    def __init__(self) -> None:
        self.loaded_kwargs: dict[str, object] = {}

    def save_pretrained(self, path: str) -> None:
        Path(path, "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_execute_uses_only_local_inputs_and_compressed_export(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    calibration_dir = tmp_path / "calibration"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    calibration_dir.mkdir()
    (calibration_dir / "calibration.jsonl").write_text(
        '{"id":"a","text":"one"}\n{"id":"b","text":"two"}\n',
        encoding="utf-8",
    )
    model = FakeModel()
    tokenizer = FakeTokenizer()
    model_load: dict[str, object] = {}
    tokenizer_load: dict[str, object] = {}
    oneshot_call: dict[str, object] = {}

    def model_loader(path: str, **kwargs: object) -> FakeModel:
        model_load.update({"path": path, **kwargs})
        return model

    def tokenizer_loader(path: str, **kwargs: object) -> FakeTokenizer:
        tokenizer_load.update({"path": path, **kwargs})
        return tokenizer

    def oneshot(**kwargs: object) -> None:
        oneshot_call.update(kwargs)

    result = execute_awq(
        {
            "model_path": str(model_dir.resolve()),
            "calibration_path": str(calibration_dir.resolve()),
            "calibration_samples": 2,
            "max_sequence_length": 128,
            "settings": {},
            "export_format": "compressed-tensors/safetensors",
            "runtime": "vllm",
        },
        output_dir,
        model_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        dataset_factory=list,
        oneshot_fn=oneshot,
        recipe_builder=lambda settings: ["awq", "quant"],
    )

    assert model_load == {
        "path": str(model_dir.resolve()),
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "auto",
        "device_map": "auto",
    }
    assert tokenizer_load["local_files_only"] is True
    assert tokenizer_load["trust_remote_code"] is False
    assert oneshot_call["num_calibration_samples"] == 2
    assert oneshot_call["max_seq_length"] == 128
    assert model.saved["save_compressed"] is True
    assert model.saved["safe_serialization"] is True
    assert result["artifact"]["weights"] == ["model.safetensors"]


def test_upstream_error_is_not_replaced_with_fallback(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    calibration_dir = tmp_path / "calibration"
    model_dir.mkdir()
    calibration_dir.mkdir()
    (calibration_dir / "calibration.jsonl").write_text(
        '{"id":"a","text":"one"}\n', encoding="utf-8"
    )

    def fail_loader(*args: object, **kwargs: object) -> object:
        raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError, match="upstream failed"):
        execute_awq(
            {
                "model_path": str(model_dir.resolve()),
                "calibration_path": str(calibration_dir.resolve()),
                "calibration_samples": 1,
                "max_sequence_length": 128,
                "settings": {},
                "export_format": "compressed-tensors/safetensors",
                "runtime": "vllm",
            },
            tmp_path / "output",
            model_loader=fail_loader,
            tokenizer_loader=lambda *args, **kwargs: object(),
            dataset_factory=list,
            oneshot_fn=lambda **kwargs: None,
        )


def test_validate_artifact_rejects_uncompressed_export(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AdapterInputError, match="quantization_config"):
        validate_artifact(tmp_path)
