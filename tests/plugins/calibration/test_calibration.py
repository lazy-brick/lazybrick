from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from lazybrick.adapters.llm_compressor import (
    CalibrationProtocol,
    materialize_calibration,
    select_records,
)
from lazybrick.adapters.llm_compressor.adapter import AdapterInputError
from lazybrick.adapters.llm_compressor.calibration import (
    DATASET_LICENSE,
    DATASET_REVISION,
    DATASET_SPLIT,
    load_pinned_dataset,
)


def records() -> list[dict[str, object]]:
    return [
        {
            "prompt_id": f"id-{index}",
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ],
        }
        for index in range(10)
    ]


class FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(f"{item['role']}: {item['content']}" for item in conversation)


def protocol() -> CalibrationProtocol:
    return CalibrationProtocol(
        sample_count=3,
        seed=42,
        max_sequence_length=2048,
        tokenizer_uri="hf://Qwen/Qwen3-0.6B",
        tokenizer_revision="a" * 40,
    )


def test_selection_is_deterministic_and_input_order_independent() -> None:
    forward = select_records(records(), sample_count=3, seed=42)
    reverse = select_records(reversed(records()), sample_count=3, seed=42)

    assert [item["prompt_id"] for item in forward] == [
        item["prompt_id"] for item in reverse
    ]
    assert len({item["prompt_id"] for item in forward}) == 3


def test_selection_changes_with_seed() -> None:
    first = select_records(records(), sample_count=3, seed=1)
    second = select_records(records(), sample_count=3, seed=2)

    assert [item["prompt_id"] for item in first] != [
        item["prompt_id"] for item in second
    ]


def test_materialization_records_license_ids_and_hash(tmp_path: Path) -> None:
    output = tmp_path / "calibration"
    manifest = materialize_calibration(records(), FakeTokenizer(), protocol(), output)

    text = (output / "calibration.jsonl").read_text(encoding="utf-8")
    disk_manifest = json.loads(
        (output / "calibration-manifest.json").read_text(encoding="utf-8")
    )
    assert disk_manifest == manifest
    assert manifest["dataset"]["revision"] == DATASET_REVISION
    assert manifest["dataset"]["license"] == "MIT"
    assert manifest["selection"]["seed"] == 42
    assert manifest["selection"]["sample_count"] == 3
    assert len(manifest["selection"]["selected_ids"]) == 3
    assert manifest["records_sha256"] == sha256(text.encode("utf-8")).hexdigest()


def test_materialization_is_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    materialize_calibration(records(), FakeTokenizer(), protocol(), first)
    materialize_calibration(reversed(records()), FakeTokenizer(), protocol(), second)

    assert (first / "calibration.jsonl").read_bytes() == (
        second / "calibration.jsonl"
    ).read_bytes()
    assert (first / "calibration-manifest.json").read_bytes() == (
        second / "calibration-manifest.json"
    ).read_bytes()


def test_duplicate_ids_are_rejected() -> None:
    duplicate = records()
    duplicate.append(duplicate[0])

    with pytest.raises(AdapterInputError, match="duplicate"):
        select_records(duplicate, sample_count=3, seed=42)


def test_loader_always_pins_revision_and_split() -> None:
    captured: dict[str, object] = {}

    def fake_loader(repo_id: str, **kwargs: object) -> list[object]:
        captured.update({"repo_id": repo_id, **kwargs})
        return []

    assert list(load_pinned_dataset(load_dataset_fn=fake_loader)) == []
    assert captured["revision"] == DATASET_REVISION
    assert captured["split"] == DATASET_SPLIT
    assert DATASET_LICENSE == "MIT"
