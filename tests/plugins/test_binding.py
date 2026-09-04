from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from lazybrick import load_recipe
from lazybrick.records import StageSpec, PluginManifest as CapabilityManifest
from lazybrick.plugins import load_manifest, PluginRunner, PluginError
from lazybrick.plugins.binding import ExecutionBinding

ROOT = Path(__file__).parents[2]


def inputs():
    import yaml
    recipe = yaml.safe_load((ROOT / "gpu/smoke/qwen3-0.6b-awq.yaml").read_text())
    stage = StageSpec.from_json(recipe["stages"][0])
    capability = CapabilityManifest.from_json(json.loads((ROOT / "gpu/smoke/planner-plugin-manifest.json").read_text()))
    transport = load_manifest(ROOT / "src/lazybrick/adapters/llm_compressor/plugin-manifest.json")
    return stage, capability, transport


def test_packaged_first_party_binding_and_protocol(tmp_path):
    stage, capability, transport = inputs()
    binding = ExecutionBinding.create(stage, capability, transport)
    result = PluginRunner(transport).run("inspect", input_dir=tmp_path,
        output_dir=tmp_path / "out", binding=binding,
        environment={"PYTHONPATH": str(ROOT / "src")})
    assert result.response.result["algorithms"] == ["awq"]
    assert result.invocation.command[0] == sys.executable
    assert result.invocation.provenance()["execution_binding"] == binding.to_dict()
    assert binding.stage_digest != binding.transport_digest


@pytest.mark.parametrize("field,value", [("version", "9.9"), ("name", "other"), ("plugin_api", "9.9")])
def test_capability_mismatch_rejected(field, value):
    stage, capability, transport = inputs()
    with pytest.raises(PluginError, match="differ"):
        ExecutionBinding.create(stage, replace(capability, **{field: value}), transport)


def test_upstream_mismatch_rejected():
    stage, capability, transport = inputs()
    bad = replace(transport, upstream={**transport.upstream, "version": "9.9"})
    with pytest.raises(PluginError, match="versions differ"):
        ExecutionBinding.create(stage, capability, bad)


def test_command_mutation_rejected_before_start(tmp_path):
    stage, capability, transport = inputs()
    binding = ExecutionBinding.create(stage, capability, transport)
    marker = tmp_path / "launched"
    bad = replace(transport, command=(sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"))
    with pytest.raises(PluginError, match="changed after binding"):
        PluginRunner(bad).run("inspect", input_dir=tmp_path, output_dir=tmp_path / "out", binding=binding)
    assert not marker.exists()
    assert not (tmp_path / "out").exists()


def test_v02_requires_binding(tmp_path):
    _, _, transport = inputs()
    with pytest.raises(PluginError, match="requires an execution binding"):
        PluginRunner(transport).run("inspect", input_dir=tmp_path, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_changed_execute_settings_rejected_before_launch(tmp_path):
    stage, capability, transport = inputs()
    binding = ExecutionBinding.create(stage, capability, transport)
    with pytest.raises(PluginError, match="settings differ"):
        PluginRunner(transport).run("execute", input_dir=tmp_path, output_dir=tmp_path/"out",
            binding=binding, payload={"settings": {"weight_bits":8}})
    assert not (tmp_path/"out").exists()


def test_installed_upstream_version_must_match(tmp_path, monkeypatch):
    stage, capability, transport = inputs()
    binding = ExecutionBinding.create(stage, capability, transport)
    monkeypatch.setattr("lazybrick.plugins.binding.version", lambda package:"9.9")
    with pytest.raises(PluginError, match="upstream package version differs"):
        binding.validate_payload(transport, {"settings": dict(stage.parameters)})


def test_semantic_declarations_cannot_bypass_mapping_gate():
    from types import SimpleNamespace
    stage, capability, transport = inputs()
    with pytest.raises(PluginError, match="tested adapter mapping"):
        ExecutionBinding.create(SimpleNamespace(semantics={"profile":"unmapped"}), capability, transport)
    binding = ExecutionBinding.create(stage, capability, transport)
    with pytest.raises(PluginError, match="tested adapter mapping"):
        binding.validate_payload(transport, {"settings":dict(stage.parameters), "semantics":None})


def test_smoke_reads_the_actual_resolved_recipe_envelope(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("binding_smoke_test", ROOT / "gpu/smoke/job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage, capability, transport = inputs()
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.shutil, "which", lambda value: "nvidia-smi")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True, device_count=lambda:1)))
    monkeypatch.setattr(module, "_hardware_profile", lambda path:path)
    envelope = {"resolved_recipe_version":"0.1", "recipe":{"stages":[stage.to_json()]}}
    monkeypatch.setattr(module, "_run_plan", lambda *args:{"recipe_digest":"a"*64,"plan_digest":"b"*64,"artifact_id":"c"*64,"resolved_recipe":envelope,"compatibility":{"accepted":True}})
    def reached_download(path):
        raise RuntimeError("binding passed before download")
    monkeypatch.setattr(module,"_download_model",reached_download)
    with pytest.raises(RuntimeError,match="binding passed before download"):
        module.execute(ROOT,tmp_path/"run")
