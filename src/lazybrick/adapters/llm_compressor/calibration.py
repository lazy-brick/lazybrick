"""Pinned, deterministic calibration materialization for the Qwen AWQ slice."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import heapq
import json
from pathlib import Path
from typing import Any, Protocol

from lazybrick.adapters.llm_compressor.adapter import AdapterInputError


DATASET_URI = "hf-dataset://HuggingFaceH4/ultrachat_200k"
DATASET_REPO_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DATASET_SPLIT = "train_sft"
DATASET_LICENSE = "MIT"
DATASET_CARD_URL = (
    "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/"
    "blob/8049631c405ae6576f93f445c6b8166f76f5505a/README.md"
)
LICENSE_TEXT_URL = "https://opensource.org/license/mit"
PREPROCESSING_ID = "lazybrick.qwen-chat-template.v1"
DEFAULT_SMOKE_SAMPLES = 32
DEFAULT_REFERENCE_SAMPLES = 512


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class CalibrationProtocol:
    sample_count: int
    seed: int
    max_sequence_length: int
    tokenizer_uri: str
    tokenizer_revision: str

    def validate(self) -> None:
        for name, value in (
            ("sample_count", self.sample_count),
            ("seed", self.seed),
            ("max_sequence_length", self.max_sequence_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise AdapterInputError(f"{name} must be an integer")
        if self.sample_count <= 0:
            raise AdapterInputError("sample_count must be positive")
        if self.max_sequence_length <= 0:
            raise AdapterInputError("max_sequence_length must be positive")
        if not self.tokenizer_uri.startswith("hf://"):
            raise AdapterInputError("tokenizer_uri must use hf://")
        if len(self.tokenizer_revision) != 40 or any(
            char not in "0123456789abcdef" for char in self.tokenizer_revision.lower()
        ):
            raise AdapterInputError("tokenizer_revision must be a 40-character Git commit")


def _record_id(record: Mapping[str, object]) -> str:
    identifier = record.get("prompt_id")
    if not isinstance(identifier, str) or not identifier:
        raise AdapterInputError("calibration records require a non-empty prompt_id")
    return identifier


def _rank(seed: int, identifier: str) -> bytes:
    return sha256(f"{seed}:{identifier}".encode("utf-8")).digest()


def select_records(
    records: Iterable[Mapping[str, object]], *, sample_count: int, seed: int
) -> list[Mapping[str, object]]:
    """Select the lowest hash ranks in bounded memory, independent of input order."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise AdapterInputError("sample_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AdapterInputError("seed must be an integer")
    selected: list[tuple[int, str, Mapping[str, object]]] = []
    seen: set[str] = set()
    for record in records:
        identifier = _record_id(record)
        if identifier in seen:
            raise AdapterInputError(f"duplicate calibration prompt_id: {identifier}")
        seen.add(identifier)
        rank = int.from_bytes(_rank(seed, identifier), "big")
        candidate = (-rank, identifier, record)
        if len(selected) < sample_count:
            heapq.heappush(selected, candidate)
        elif candidate > selected[0]:
            heapq.heapreplace(selected, candidate)
    if len(selected) != sample_count:
        raise AdapterInputError(
            f"requested {sample_count} calibration samples but found {len(selected)}"
        )
    return [item[2] for item in sorted(selected, key=lambda item: (-item[0], item[1]))]


def _messages(record: Mapping[str, object]) -> list[dict[str, str]]:
    value = record.get("messages")
    if not isinstance(value, list) or not value:
        raise AdapterInputError(f"record {_record_id(record)} has no messages")
    result: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise AdapterInputError(
                f"record {_record_id(record)} message {index} must contain only role and content"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise AdapterInputError(
                f"record {_record_id(record)} message {index} is invalid"
            )
        result.append({"role": role, "content": content})
    return result


def materialize_calibration(
    records: Iterable[Mapping[str, object]],
    tokenizer: ChatTokenizer,
    protocol: CalibrationProtocol,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write selected chat-template text and a complete reproducibility manifest."""

    protocol.validate()
    selected = select_records(
        records, sample_count=protocol.sample_count, seed=protocol.seed
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    output_records: list[dict[str, str]] = []
    for record in selected:
        identifier = _record_id(record)
        text = tokenizer.apply_chat_template(
            _messages(record), tokenize=False, add_generation_prompt=False
        )
        if not isinstance(text, str) or not text:
            raise AdapterInputError(f"chat template produced empty text for {identifier}")
        output_records.append({"id": identifier, "text": text})

    records_path = target / "calibration.jsonl"
    records_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        for record in output_records
    )
    records_path.write_text(records_text, encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": "0.1",
        "dataset": {
            "uri": DATASET_URI,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "license": DATASET_LICENSE,
            "dataset_card": DATASET_CARD_URL,
            "license_text": LICENSE_TEXT_URL,
        },
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "tokenizer_uri": protocol.tokenizer_uri,
            "tokenizer_revision": protocol.tokenizer_revision,
            "chat_template_source": "pinned tokenizer",
        },
        "selection": {
            "algorithm": "sha256-rank-v1",
            "seed": protocol.seed,
            "sample_count": protocol.sample_count,
            "selected_ids": [record["id"] for record in output_records],
        },
        "max_sequence_length": protocol.max_sequence_length,
        "records_sha256": sha256(records_text.encode("utf-8")).hexdigest(),
    }
    (target / "calibration-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_pinned_dataset(
    *, load_dataset_fn: Callable[..., Iterable[Mapping[str, object]]] | None = None
) -> Iterable[Mapping[str, object]]:
    """Load exactly the approved dataset revision and split."""

    if load_dataset_fn is None:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                'calibration materialization requires: pip install "lazybrick[awq]"'
            ) from error
        load_dataset_fn = load_dataset
    return load_dataset_fn(
        DATASET_REPO_ID,
        revision=DATASET_REVISION,
        split=DATASET_SPLIT,
    )
