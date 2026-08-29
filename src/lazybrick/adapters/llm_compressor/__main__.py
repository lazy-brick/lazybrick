"""JSON subprocess entry point for the first-party AWQ adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback
from typing import Any

from lazybrick.adapters.llm_compressor.adapter import (
    AWQSettings,
    AdapterInputError,
    execute_awq,
    recipe_spec,
    validate_artifact,
)


API_VERSION = "0.1"


def _response(request: dict[str, Any], *, result: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "protocol_version": API_VERSION,
        "request_id": request.get("request_id"),
        "operation": request.get("operation"),
        "status": "error" if error else "success",
        **({"error": error} if error else {"result": result or {}}),
    }


def handle(request: dict[str, Any]) -> dict[str, Any]:
    operation = request.get("operation")
    payload = request.get("payload", {})
    if operation == "inspect":
        return _response(
            request,
            result={
                "model_profiles": ["dense_decoder_only_causal_lm"],
                "algorithms": ["awq"],
                "schemes": ["W4A16"],
                "output_formats": ["compressed-tensors/safetensors"],
                "runtimes": ["vllm"],
                "accelerator_vendor": "nvidia",
                "device_count": 1,
                "remote_code": False,
            },
        )
    if operation == "plan":
        settings = AWQSettings.from_mapping(payload.get("settings", {}))
        if payload.get("export_format") != "compressed-tensors/safetensors":
            raise AdapterInputError("export_format must be compressed-tensors/safetensors")
        if payload.get("runtime") != "vllm":
            raise AdapterInputError("runtime must be vllm")
        return _response(
            request,
            result={
                "accepted": True,
                "algorithm": "awq",
                "scheme": "W4A16",
                "recipe": recipe_spec(settings),
                "requires_materialized_model": True,
                "requires_materialized_calibration": True,
            },
        )
    if operation == "execute":
        return _response(request, result=execute_awq(payload, request["output_dir"]))
    if operation == "validate":
        artifact_path = payload.get("artifact_path")
        if not isinstance(artifact_path, str):
            artifact_path = request["input_dir"]
        return _response(request, result=validate_artifact(Path(artifact_path)))
    raise AdapterInputError(f"unsupported operation: {operation}")


def main() -> int:
    request: dict[str, Any] = {}
    try:
        value = json.load(sys.stdin)
        if not isinstance(value, dict):
            raise AdapterInputError("request must be an object")
        request = value
        if request.get("protocol_version") != API_VERSION:
            raise AdapterInputError("incompatible plugin protocol version")
        response = handle(request)
    except AdapterInputError as error:
        response = _response(
            request,
            error={"code": "adapter_input_invalid", "message": str(error), "details": {}},
        )
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        response = _response(
            request,
            error={
                "code": "adapter_execution_failed",
                "message": str(error),
                "details": {"exception_type": type(error).__name__},
            },
        )
    print(json.dumps(response, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
