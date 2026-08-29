"""Shared fixtures.

`valid_recipe` returns a deep-copyable v0.1 recipe that every test mutates into
whatever shape it needs. Tests must never share a mutable recipe instance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

MODEL_SHA = "1cfa9a7208912126459214e8b04321603b3df60c"
PLUGIN_SHA = "de46bfd53513aa87571a8b056a06aeaa5da1c69c"


def build_recipe() -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "model": {
            "uri": "hf://Qwen/Qwen3-4B",
            "revision": MODEL_SHA,
            "trust_remote_code": False,
        },
        "stages": [
            {
                "id": "quantize",
                "plugin": "lazybrick.plugin/awq",
                "plugin_version": "0.1.0",
                "implementation": {
                    "git": "https://github.com/vllm-project/llm-compressor",
                    "commit": PLUGIN_SHA,
                },
                "parameters": {
                    "weight_bits": 4,
                    "group_size": 128,
                    "ignore": ["lm_head"],
                },
            }
        ],
        "calibration": {
            "dataset": {
                "uri": "hf-dataset://example/calibration-set",
                "revision": "a" * 40,
            },
            "samples": 256,
            "seed": 42,
            "preprocessing_id": "qwen3-chat-template-v1",
            "max_sequence_length": 2048,
        },
        "export": {"format": "compressed-tensors", "runtime": "vllm"},
        "target": {
            "accelerator_family": "nvidia-cuda",
            "min_compute_capability": "8.0",
            "device_count": 1,
            "min_memory_gib": 40,
            "runtime": "vllm",
        },
        "evaluation": {
            "protocol_id": "token-loss-v1",
            "dataset": {
                "uri": "hf-dataset://example/held-out-set",
                "revision": "b" * 40,
            },
            "max_samples": 128,
            "seed": 1234,
        },
    }


@pytest.fixture
def valid_recipe() -> dict[str, Any]:
    return build_recipe()


# --------------------------------------------------------------------------
# Offline Hugging Face fixtures
# --------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "hf"
ENDPOINT = "https://huggingface.co"

#: repo id -> the SHA its fixtures were captured at, read from the fixtures
#: themselves so the two can never drift apart.
RECORDED_MODELS = {
    json.loads(path.read_text(encoding="utf-8"))["id"]: json.loads(
        path.read_text(encoding="utf-8")
    )["sha"]
    for path in sorted(FIXTURES.glob("Qwen_*.info.json"))
}

#: Dataset revisions the synthetic fixture answers to. "b" appears because the
#: default recipe gives calibration and evaluation different revisions.
_DATASET_REVISIONS = ("a" * 40, "b" * 40, "main")


def recorded_responses() -> dict[str, bytes]:
    """Map every URL the resolver may request onto a recorded fixture.

    Each model is registered under both its pinned SHA and the mutable aliases
    a recipe might legitimately name, so branch-to-SHA resolution can be tested
    without inventing a second fixture.
    """

    responses: dict[str, bytes] = {}
    for repo, sha in RECORDED_MODELS.items():
        slug = repo.replace("/", "_")
        info = (FIXTURES / f"{slug}.info.json").read_bytes()
        config = (FIXTURES / f"{slug}.config.json").read_bytes()
        for revision in (sha, "main", "refs/pr/1"):
            responses[f"{ENDPOINT}/api/models/{repo}/revision/{revision}"] = info
        responses[f"{ENDPOINT}/{repo}/resolve/{sha}/config.json"] = config

    dataset = (FIXTURES / "synthetic_dataset.info.json").read_bytes()
    for name in ("calibration-set", "held-out-set"):
        for revision in _DATASET_REVISIONS:
            responses[
                f"{ENDPOINT}/api/datasets/example/{name}/revision/{revision}"
            ] = dataset
    return responses


@pytest.fixture
def recorded_models() -> dict[str, str]:
    return dict(RECORDED_MODELS)


@pytest.fixture
def hf_transport():
    from lazybrick.resolve import RecordedTransport

    return RecordedTransport(recorded_responses())


@pytest.fixture
def resolver(hf_transport, tmp_path):
    from lazybrick.resolve import Resolver, ResolverCache

    return Resolver(hf_transport, ResolverCache(tmp_path / "cache"))
