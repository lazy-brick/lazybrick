"""Unattended Qwen3-0.6B -> AWQ -> vLLM -> evidence smoke job."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

from lazybrick.adapters.llm_compressor import (
    CalibrationProtocol,
    materialize_calibration,
    select_records,
)
from lazybrick.adapters.llm_compressor.calibration import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
)
from lazybrick.evidence import (
    evidence_record,
    compare_benchmarks,
)
from lazybrick.evidence.assistant_quality import assistant_samples, compare_assistant_quality

from lazybrick.runs import (
    RunIdentity,
    RunState,
    RunStateMachine,
    RunStore,
    collect_provenance,
)
from lazybrick.plugins import PluginRunner, load_manifest
from lazybrick.plugins.binding import ExecutionBinding
from lazybrick.records import StageSpec, PluginManifest as CapabilityManifest


MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
MODEL_URI = f"hf://{MODEL_REPO_ID}"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PLUGIN_COMMIT = "de46bfd53513aa87571a8b056a06aeaa5da1c69c"
SMOKE_SAMPLES = 32
EVALUATION_SAMPLES = 8
CALIBRATION_SEED = 42
EVALUATION_SEED = 1234
MAX_SEQUENCE_LENGTH = 2048
RUNTIME_SETTINGS = {"dtype": "bfloat16", "max_model_len": 2048, "gpu_memory_utilization": "0.85"}


class SmokeJobError(RuntimeError):
    pass


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_plan(recipe: Path, target: Path, plugin_manifest: Path, logs: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "lazybrick",
        "plan",
        str(recipe),
        "--target",
        str(target),
        "--plugin-manifest",
        str(plugin_manifest),
        "--json",
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    (logs / "plan.stderr.log").write_text(completed.stderr, encoding="utf-8")
    (logs / "plan.stdout.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise SmokeJobError(f"lazybrick plan failed with status {completed.returncode}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SmokeJobError("lazybrick plan did not emit one JSON document") from error
    if not isinstance(value, dict):
        raise SmokeJobError("lazybrick plan output must be an object")
    return value


def require_plan_fields(
    plan: Mapping[str, Any],
) -> tuple[str, str, str, Mapping[str, Any]]:
    recipe_digest = plan.get("recipe_digest")
    plan_digest = plan.get("plan_digest")
    planned_artifact_id = plan.get("artifact_id")
    resolved = plan.get("resolved_recipe")
    for name, value in (
        ("recipe_digest", recipe_digest),
        ("plan_digest", plan_digest),
        ("artifact_id", planned_artifact_id),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise SmokeJobError(f"plan is missing a SHA-256 {name}")
    if not isinstance(resolved, Mapping):
        raise SmokeJobError("plan is missing resolved_recipe")
    compatibility = plan.get("compatibility")
    if not isinstance(compatibility, Mapping) or compatibility.get("accepted") is not True:
        raise SmokeJobError("planner did not accept the smoke recipe")
    return recipe_digest, plan_digest, planned_artifact_id, resolved


def _hardware_profile(path: Path) -> Path:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(lines) != 1:
        raise SmokeJobError("could not inventory exactly one NVIDIA GPU")
    name, memory_mib, compute_capability = (part.strip() for part in lines[0].split(","))
    _json(
        path,
        {
            "vendor": "nvidia",
            "device_count": 1,
            "compute_capability": compute_capability,
            "memory_gib": int(memory_mib) // 1024,
            "name": name,
        },
    )
    return path


def _download_model(target: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        local_dir=target,
    )
    return target


def _execute_adapter(
    repo_root: Path,
    *,
    model_path: Path,
    calibration_path: Path,
    artifact_path: Path,
    binding: ExecutionBinding,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    manifest_path = (
        repo_root
        / "src"
        / "lazybrick"
        / "adapters"
        / "llm_compressor"
        / "plugin-manifest.json"
    )
    manifest = load_manifest(manifest_path)
    forwarded_environment = {
        key: os.environ[key]
        for key in (
            "CUDA_DEVICE_ORDER",
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "LD_LIBRARY_PATH",
            "NVIDIA_VISIBLE_DEVICES",
            "PYTHONHASHSEED",
            "TOKENIZERS_PARALLELISM",
        )
        if key in os.environ
    }
    result = PluginRunner(manifest, timeout_seconds=7200).run(
        "execute",
        input_dir=model_path,
        output_dir=artifact_path,
        payload={
            "model_path": str(model_path),
            "calibration_path": str(calibration_path),
            "calibration_samples": SMOKE_SAMPLES,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "settings": {
                "weight_bits": 4,
                "group_size": 128,
                "symmetric": False,
                "targets": ["Linear"],
                "ignore": ["lm_head"],
            },
            "export_format": "compressed-tensors/safetensors",
            "runtime": "vllm",
        },
        environment=forwarded_environment,
        binding=binding,
    )
    return (
        result.response.result or {},
        result.invocation.provenance(),
        result.invocation.stdout,
        result.invocation.stderr,
    )


def _load_dataset(split: str) -> Iterable[Mapping[str, object]]:
    from datasets import load_dataset

    return load_dataset(
        DATASET_REPO_ID,
        revision=DATASET_REVISION,
        split=split,
    )


def _load_tokenizer(model_path: Path) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )


def _eval_conversations(records: Iterable[Mapping[str, object]]) -> list[list[dict[str, str]]]:
    selected = select_records(
        records, sample_count=EVALUATION_SAMPLES, seed=EVALUATION_SEED
    )
    conversations: list[list[dict[str, str]]] = []
    for record in selected:
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise SmokeJobError("evaluation record is missing messages")
        conversation: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise SmokeJobError("evaluation message is invalid")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise SmokeJobError("evaluation role/content is invalid")
            conversation.append({"role": role, "content": content})
        conversations.append(conversation)
    return conversations


def _vllm_evidence(model_path: Path, artifact_path: Path,
                   evaluation_samples: list[dict[str, Any]], work_root: Path, logs: Path):
    vllm_python = os.environ.get("LAZYBRICK_VLLM_PYTHON")
    if not vllm_python or not Path(vllm_python).is_file():
        raise SmokeJobError("LAZYBRICK_VLLM_PYTHON does not name the locked vLLM interpreter")
    phases = {}
    for phase in ("baseline", "quantized"):
        input_path = work_root / f"vllm-{phase}-input.json"
        output_path = work_root / f"vllm-{phase}-evidence.json"
        _json(input_path, {"model_path": str(model_path), "artifact_path": str(artifact_path),
            "evaluation_samples": evaluation_samples, "seed": EVALUATION_SEED,
            "phase": phase, "runtime": RUNTIME_SETTINGS})
        completed = subprocess.run([vllm_python, str(Path(__file__).with_name("validate.py")),
            "--input", str(input_path), "--output", str(output_path)],
            text=True, capture_output=True, check=False)
        (logs / f"vllm-{phase}.stdout.log").write_text(completed.stdout)
        (logs / f"vllm-{phase}.stderr.log").write_text(completed.stderr)
        if completed.returncode or not output_path.is_file():
            raise SmokeJobError(f"locked vLLM {phase} validation failed")
        value = json.loads(output_path.read_text())
        if value.get("schema_version") != "0.2" or value.get("phase") != phase:
            raise SmokeJobError("vLLM returned mismatched evidence phase")
        phases[phase] = value
    baseline, quantized = phases["baseline"], phases["quantized"]
    if baseline["runtime"] != quantized["runtime"]:
        raise SmokeJobError("baseline and quantized runtime settings differ")
    return (quantized["generations"],
        compare_assistant_quality(baseline["quality"], quantized["quality"]),
        compare_benchmarks(baseline["performance"], quantized["performance"]),
        {"baseline": baseline["resources"], "quantized": quantized["resources"]},
        baseline["runtime"])


def execute(repo_root: Path, work_root: Path) -> Path:
    if sys.platform != "linux":
        raise SmokeJobError("GPU smoke job requires Linux")
    if shutil.which("nvidia-smi") is None:
        raise SmokeJobError("nvidia-smi is unavailable")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SmokeJobError("GPU smoke job requires exactly one visible CUDA GPU")

    work_root.mkdir(parents=True, exist_ok=False)
    logs = work_root / "bootstrap-logs"
    logs.mkdir()
    state = RunStateMachine()
    recipe_path = repo_root / "gpu" / "smoke" / "qwen3-0.6b-awq.yaml"
    plugin_manifest_path = repo_root / "gpu" / "smoke" / "planner-plugin-manifest.json"
    target_path = _hardware_profile(work_root / "hardware.json")
    state.transition(RunState.RESOLVING)
    plan = _run_plan(recipe_path, target_path, plugin_manifest_path, logs)
    recipe_digest, plan_digest, planned_artifact_id, resolved_recipe = require_plan_fields(plan)

    if resolved_recipe.get("resolved_recipe_version") != "0.1":
        raise SmokeJobError("unsupported resolved recipe envelope")
    recipe_body = resolved_recipe.get("recipe")
    stages = recipe_body.get("stages") if isinstance(recipe_body, Mapping) else None
    if not isinstance(stages, list) or len(stages) != 1:
        raise SmokeJobError("smoke execution requires exactly one planned stage")
    transport = load_manifest(repo_root / "src/lazybrick/adapters/llm_compressor/plugin-manifest.json")
    capability = CapabilityManifest.from_json(json.loads(plugin_manifest_path.read_text()))
    binding = ExecutionBinding.create(StageSpec.from_json(stages[0]), capability, transport)

    model_path = _download_model(work_root / "inputs" / "model")
    tokenizer = _load_tokenizer(model_path)
    calibration_path = work_root / "inputs" / "calibration"
    calibration_manifest = materialize_calibration(
        _load_dataset(DATASET_SPLIT),
        tokenizer,
        CalibrationProtocol(
            sample_count=SMOKE_SAMPLES,
            seed=CALIBRATION_SEED,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            tokenizer_uri=MODEL_URI,
            tokenizer_revision=MODEL_REVISION,
        ),
        calibration_path,
    )
    state.transition(RunState.VALIDATING)
    state.transition(RunState.READY)

    resolved_inputs = {
        "model": {"uri": MODEL_URI, "revision": MODEL_REVISION},
        "plugin": {
            "package": "llmcompressor",
            "version": "0.13.0",
            "commit": PLUGIN_COMMIT,
        },
        "calibration": calibration_manifest,
        "quantization": {
            "algorithm": "awq",
            "weight_bits": 4,
            "group_size": 128,
            "symmetric": False,
            "targets": ["Linear"],
            "ignore": ["lm_head"],
        },
        "export": {"format": "compressed-tensors/safetensors", "runtime": "vllm"},
    }
    identity = RunIdentity.create(
        recipe_digest=recipe_digest,
        plan_digest=plan_digest,
        artifact_id=planned_artifact_id,
    )
    store = RunStore(work_root / "store")
    bundle = store.begin(identity)
    bundle.write_text("recipe.yaml", recipe_path.read_text(encoding="utf-8"))
    bundle.write_json("resolved_recipe.json", dict(resolved_recipe))
    bundle.write_json("plan.json", plan)
    bundle.write_log("plan.stdout.log", (logs / "plan.stdout.log").read_text())
    bundle.write_log("plan.stderr.log", (logs / "plan.stderr.log").read_text())

    try:
        state.transition(RunState.RUNNING)
        started = time.perf_counter()
        adapter_result, plugin_provenance, plugin_stdout, plugin_stderr = _execute_adapter(
            repo_root,
            model_path=model_path,
            calibration_path=calibration_path,
            artifact_path=bundle.artifact_dir,
            binding=binding,
        )
        bundle.write_log("plugin.stdout.log", plugin_stdout)
        bundle.write_log("plugin.stderr.log", plugin_stderr)
        build_time = time.perf_counter() - started
        gc.collect()
        torch.cuda.empty_cache()
        state.transition(RunState.EXPORTING)
        state.transition(RunState.EVALUATING)
        evaluation_records = _load_dataset("test_sft")
        evaluation_samples = assistant_samples(
            _eval_conversations(evaluation_records), tokenizer, max_model_len=MAX_SEQUENCE_LENGTH
        )
        generations, quality, performance, resources, runtime = _vllm_evidence(
            model_path,
            bundle.artifact_dir,
            evaluation_samples,
            work_root,
            logs,
        )
        for phase in ("baseline", "quantized"):
            for stream in ("stdout", "stderr"):
                name = f"vllm-{phase}.{stream}.log"
                bundle.write_log(name, (logs / name).read_text())
        resources["build"] = adapter_result["resources"]
        results = {"schema_version": "0.2", "generations": generations,
                   "quality": quality, "performance": performance,
                   "resources": resources, "runtime": runtime,
                   "build_time_seconds": str(build_time),
                   "claims": {"quality_regression_gate": None,
                              "note": "sampled process-tree resources; no universal threshold"}}
        bundle.write_json("results.json", results)
        provenance = collect_provenance(
            commands=[
                [
                    sys.executable,
                    "-m",
                    "lazybrick",
                    "plan",
                    str(recipe_path),
                    "--target",
                    str(target_path),
                    "--plugin-manifest",
                    str(plugin_manifest_path),
                    "--json",
                ],
                [sys.executable, str(Path(__file__).resolve())],
            ],
            seeds={"calibration": CALIBRATION_SEED, "evaluation": EVALUATION_SEED},
            environment=os.environ,
        )
        provenance["plugin_invocation"] = plugin_provenance
        bundle.write_json("provenance.json", provenance)
        state.transition(RunState.SUCCEEDED)
        bundle.write_json("state-history.json", state.to_dict())
        destination = bundle.finalize_success(
            {
                "format": "compressed-tensors/safetensors",
                "resolved_inputs": resolved_inputs,
            }
        )
        print(json.dumps({"status": "SUCCEEDED", "bundle": str(destination)}))
        return destination
    except Exception as error:
        failure_state = {
            RunState.RUNNING: RunState.BUILD_FAILED,
            RunState.EXPORTING: RunState.EXPORT_FAILED,
            RunState.EVALUATING: RunState.EVALUATION_FAILED,
        }.get(state.state, RunState.EXECUTION_FAILED)
        if state.terminal:
            raise
        state.transition(
            failure_state,
            reason_code="smoke_job_failed",
            message=str(error) or type(error).__name__,
        )
        bundle.write_json("state-history.json", state.to_dict())
        bundle.finalize_failure(
            failure_state.value,
            {"code": "smoke_job_failed", "message": str(error) or type(error).__name__},
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    work_root = args.work_root.resolve()
    try:
        execute(args.repo_root.resolve(), work_root)
    except Exception as error:
        work_root.mkdir(parents=True, exist_ok=True)
        _json(
            work_root / "bootstrap-failure.json",
            {
                "state": "EXECUTION_FAILED",
                "code": "smoke_job_failed",
                "message": str(error) or type(error).__name__,
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
