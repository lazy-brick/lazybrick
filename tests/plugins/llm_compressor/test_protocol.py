from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def invoke(request: dict[str, object], cwd: Path) -> tuple[dict[str, object], str]:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[3] / "src"
    environment["PYTHONPATH"] = str(source_root.resolve())
    completed = subprocess.run(
        [sys.executable, "-m", "lazybrick.adapters.llm_compressor"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        cwd=cwd,
        env=environment,
        check=True,
    )
    return json.loads(completed.stdout), completed.stderr


def test_inspect_reports_narrow_capability_surface(tmp_path: Path) -> None:
    response, stderr = invoke(
        {
            "protocol_version": "0.1",
            "request_id": "inspect-1",
            "operation": "inspect",
            "input_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "payload": {},
        },
        tmp_path,
    )

    assert stderr == ""
    assert response["status"] == "success"
    assert response["result"]["model_profiles"] == ["dense_decoder_only_causal_lm"]
    assert response["result"]["remote_code"] is False


def test_plan_rejects_scheme_fallback(tmp_path: Path) -> None:
    response, _ = invoke(
        {
            "protocol_version": "0.1",
            "request_id": "plan-1",
            "operation": "plan",
            "input_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "payload": {
                "settings": {"weight_bits": 8},
                "export_format": "compressed-tensors/safetensors",
                "runtime": "vllm",
            },
        },
        tmp_path,
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "adapter_input_invalid"
