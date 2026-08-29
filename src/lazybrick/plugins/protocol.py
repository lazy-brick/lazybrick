"""Versioned JSON request and response records for subprocess plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from lazybrick.plugins.errors import PluginError, PluginFailure
from lazybrick.plugins.manifest import PLUGIN_API_VERSION, SUPPORTED_OPERATIONS


def _validate_json_value(value: object, path: str = "payload") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise PluginError(
            PluginFailure(
                "plugin_protocol_error",
                f"{path} contains a floating-point value",
            )
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PluginError(
                    PluginFailure(
                        "plugin_protocol_error",
                        f"{path} contains a non-string mapping key",
                    )
                )
            _validate_json_value(item, f"{path}.{key}")
        return
    raise PluginError(
        PluginFailure(
            "plugin_protocol_error",
            f"{path} contains unsupported value type {type(value).__name__}",
        )
    )


@dataclass(frozen=True, slots=True)
class PluginRequest:
    operation: str
    input_dir: str
    output_dir: str
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)
    protocol_version: str = PLUGIN_API_VERSION

    def __post_init__(self) -> None:
        if self.operation not in SUPPORTED_OPERATIONS:
            raise PluginError(
                PluginFailure(
                    "unsupported_plugin_operation",
                    f"unsupported plugin operation: {self.operation}",
                )
            )
        _validate_json_value(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "operation": self.operation,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class PluginResponse:
    request_id: str
    operation: str
    status: str
    result: dict[str, Any] | None = None
    error: PluginFailure | None = None
    protocol_version: str = PLUGIN_API_VERSION

    @classmethod
    def from_dict(cls, value: object, request: PluginRequest) -> PluginResponse:
        if not isinstance(value, dict):
            raise PluginError(
                PluginFailure("plugin_protocol_error", "plugin response must be an object")
            )
        if value.get("protocol_version") != PLUGIN_API_VERSION:
            raise PluginError(
                PluginFailure(
                    "plugin_protocol_error",
                    "plugin response uses an incompatible protocol version",
                )
            )
        if value.get("request_id") != request.request_id:
            raise PluginError(
                PluginFailure("plugin_protocol_error", "plugin response request_id mismatch")
            )
        if value.get("operation") != request.operation:
            raise PluginError(
                PluginFailure("plugin_protocol_error", "plugin response operation mismatch")
            )
        status = value.get("status")
        if status not in {"success", "error"}:
            raise PluginError(
                PluginFailure("plugin_protocol_error", "plugin response status is invalid")
            )
        result = value.get("result")
        error_value = value.get("error")
        if status == "success":
            if not isinstance(result, dict) or error_value is not None:
                raise PluginError(
                    PluginFailure(
                        "plugin_protocol_error",
                        "successful response requires an object result and no error",
                    )
                )
            _validate_json_value(result, "result")
            return cls(
                request_id=request.request_id,
                operation=request.operation,
                status=status,
                result=result,
            )
        if not isinstance(error_value, dict):
            raise PluginError(
                PluginFailure(
                    "plugin_protocol_error",
                    "error response requires an error object",
                )
            )
        code = error_value.get("code")
        message = error_value.get("message")
        details = error_value.get("details", {})
        if not isinstance(code, str) or not isinstance(message, str) or not isinstance(details, dict):
            raise PluginError(
                PluginFailure("plugin_protocol_error", "plugin error fields are invalid")
            )
        _validate_json_value(details, "error.details")
        return cls(
            request_id=request.request_id,
            operation=request.operation,
            status=status,
            error=PluginFailure(code, message, details),
        )
