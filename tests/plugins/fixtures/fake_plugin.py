"""Fake JSON subprocess plugin used only by contract tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


request = json.load(sys.stdin)
mode = request.get("payload", {}).get("mode", "success")

if mode == "crash":
    print("intentional fake-plugin crash", file=sys.stderr)
    raise SystemExit(17)
if mode == "malformed":
    print("not-json")
    raise SystemExit(0)
if mode == "timeout":
    time.sleep(5)
if mode == "error":
    print(
        json.dumps(
            {
                "protocol_version": "0.1",
                "request_id": request["request_id"],
                "operation": request["operation"],
                "status": "error",
                "error": {
                    "code": "fake_rejected",
                    "message": "fake plugin rejected the request",
                    "details": {"mode": mode},
                },
            }
        )
    )
    raise SystemExit(0)

output = Path(request["output_dir"]) / "fake-output.json"
output.write_text('{"ok":true}\n', encoding="utf-8")
print("fake plugin diagnostic", file=sys.stderr)
print(
    json.dumps(
        {
            "protocol_version": "0.1",
            "request_id": request["request_id"],
            "operation": request["operation"],
            "status": "success",
            "result": {
                "output": output.name,
                "secret_visible": "LAZYBRICK_TEST_SECRET" in os.environ,
            },
        }
    )
)
