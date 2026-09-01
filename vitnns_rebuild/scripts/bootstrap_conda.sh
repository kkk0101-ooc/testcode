#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "${script_dir}/.." && pwd)
workspace_dir=$(cd -- "${project_dir}/.." && pwd)
conda_root="${workspace_dir}/.conda"

if [[ ! -x "${conda_root}/bin/conda" ]]; then
  printf 'Missing project-local Conda at %s\n' "${conda_root}" >&2
  printf 'Install Miniforge there first, then rerun this script.\n' >&2
  exit 1
fi

source "${conda_root}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx vitnns; then
  conda env create -f "${project_dir}/environment.yml"
fi

conda activate vitnns
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r "${project_dir}/requirements.lock.txt"
python -m pip install -e "${project_dir}" --no-deps
python -m pip check
