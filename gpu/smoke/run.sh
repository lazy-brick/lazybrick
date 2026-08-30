#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
job_root="${LAZYBRICK_SMOKE_ROOT:-${repo_root}/.lazybrick-smoke}"
build_venv="${job_root}/build-venv"
vllm_venv="${job_root}/vllm-venv"
mkdir -p "${job_root}"

python3.12 -c 'import platform,sys; v=tuple(map(int, platform.libc_ver()[1].split(".")[:2])); sys.exit(0 if v >= (2,31) else "glibc 2.31 or newer is required")'

python3.12 -m venv "${build_venv}"
python3.12 -m venv "${vllm_venv}"
"${build_venv}/bin/python" -m pip install --upgrade pip==26.0.1
"${vllm_venv}/bin/python" -m pip install --upgrade pip==26.0.1
"${build_venv}/bin/python" -m pip install \
  --require-hashes \
  --only-binary=:all: \
  -r "${script_dir}/requirements-build.lock"
"${vllm_venv}/bin/python" -m pip install \
  --require-hashes \
  --only-binary=:all: \
  -r "${script_dir}/requirements-vllm.lock"
"${build_venv}/bin/python" -m pip install --no-deps "${repo_root}"
"${vllm_venv}/bin/python" -m pip install --no-deps "${repo_root}"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export LAZYBRICK_VLLM_PYTHON="${vllm_venv}/bin/python"

"${build_venv}/bin/python" "${script_dir}/job.py" \
  --repo-root "${repo_root}" \
  --work-root "${job_root}/work" \
  2>&1 | tee "${job_root}/job.log"

"${build_venv}/bin/python" "${script_dir}/verify_bundle.py" \
  "${job_root}/work/store"
