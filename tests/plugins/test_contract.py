from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from lazybrick.plugins import PluginError, PluginManifest, PluginRunner, discover_plugins


FIXTURES = Path(__file__).parent / "fixtures"


def manifest_dict(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "manifest_version": "0.1",
        "plugin_api_version": "0.1",
        "name": "fake",
        "package": "lazybrick-fake-plugin",
        "package_version": "1.2.3",
        "implementation": {
            "source": "https://example.invalid/lazybrick-fake-plugin",
            "commit": "0123456789abcdef0123456789abcdef01234567",
        },
        "command": [sys.executable, str(FIXTURES / "fake_plugin.py")],
        "operations": ["inspect", "plan", "execute", "validate"],
        "runtime_dependencies": ["pytest", "package-that-does-not-exist"],
    }
    value.update(overrides)
    return value


def runner(timeout_seconds: int = 2) -> PluginRunner:
    return PluginRunner(PluginManifest.from_dict(manifest_dict()), timeout_seconds=timeout_seconds)


def test_fake_plugin_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAZYBRICK_TEST_SECRET", "must-not-leak")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = runner().run("plan", input_dir=input_dir, output_dir=output_dir)

    assert result.response.result == {
        "output": "fake-output.json",
        "secret_visible": False,
    }
    assert (output_dir / "fake-output.json").is_file()
    assert result.invocation.stderr == "fake plugin diagnostic\n"
    assert result.invocation.command[0] == sys.executable
    assert result.invocation.runtime_dependencies["pytest"] != "not-installed"
    assert result.invocation.runtime_dependencies["package-that-does-not-exist"] == "not-installed"
    assert result.invocation.provenance()["plugin"] == {
        "name": "fake",
        "package": "lazybrick-fake-plugin",
        "package_version": "1.2.3",
        "implementation": {
            "source": "https://example.invalid/lazybrick-fake-plugin",
            "commit": "0123456789abcdef0123456789abcdef01234567",
        },
        "plugin_api_version": "0.1",
    }


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("malformed", "plugin_protocol_error"),
        ("crash", "plugin_crashed"),
        ("error", "fake_rejected"),
    ],
)
def test_plugin_failures_have_stable_codes(
    tmp_path: Path, mode: str, code: str
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(PluginError) as caught:
        runner().run(
            "execute",
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            payload={"mode": mode},
        )

    assert caught.value.failure.code == code


def test_timeout_has_stable_code_and_logs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(PluginError) as caught:
        runner(timeout_seconds=1).run(
            "execute",
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            payload={"mode": "timeout"},
        )

    assert caught.value.failure.code == "plugin_timeout"
    assert "duration_ms" in caught.value.failure.details


def test_incompatible_api_is_rejected_before_start(tmp_path: Path) -> None:
    marker = tmp_path / "started"
    incompatible = manifest_dict(
        plugin_api_version="9.9",
        command=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
    )

    with pytest.raises(PluginError) as caught:
        PluginManifest.from_dict(incompatible)

    assert caught.value.failure.code == "incompatible_plugin_api"
    assert not marker.exists()


def test_manifest_requires_immutable_implementation() -> None:
    value = manifest_dict(implementation={"source": "https://example.invalid/plugin"})

    with pytest.raises(PluginError) as caught:
        PluginManifest.from_dict(value)

    assert caught.value.failure.code == "mutable_plugin_implementation"


def test_discovery_does_not_import_plugin_code(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "fake"
    plugin_dir.mkdir()
    (plugin_dir / "plugin-manifest.json").write_text(
        json.dumps(manifest_dict()), encoding="utf-8"
    )

    discovered = discover_plugins([tmp_path])

    assert discovered["fake"].package_version == "1.2.3"


def test_protocol_rejects_floating_point_payload(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(PluginError) as caught:
        runner().run(
            "plan",
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            payload={"ratio": 0.5},
        )

    assert caught.value.failure.code == "plugin_protocol_error"
