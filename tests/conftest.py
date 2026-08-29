"""Shared fixtures.

`valid_recipe` returns a deep-copyable v0.1 recipe that every test mutates into
whatever shape it needs. Tests must never share a mutable recipe instance.
"""

from __future__ import annotations

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
