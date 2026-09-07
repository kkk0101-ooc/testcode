#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "${script_dir}/.." && pwd)
workspace_dir=$(cd -- "${project_dir}/.." && pwd)

d8_root="${D8_ROOT:-evidence/stage_d/d8_recovery_v2_20260904/formal_r2}"
d9_report="${D9_REPORT:-reports/stage_d_d9_recovery_v2_20260906/d9_formal_validation.json}"
d9_budget="${D9_FORMAL_BUDGET:-config/stage_d_d7_budget_20260904_r4.json}"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
rerun_root="${D10_RERUN_ROOT:-evidence/stage_d/d10_recovery_v2_20260906/rerun_lightweight_${timestamp}}"
analysis_root="${rerun_root}/analysis"
paper_root="${rerun_root}/paper"
receipt_dir="${rerun_root}/receipts"

expected_d9_report_sha256="bbfb6b5ad879116f6849e6f0a058780fdfd5e54b656e62f0674a3c3184c810d5"
expected_d8_final_status_sha256="9c2a7f6bbe89f393b17d39128c908b764c5481cb2ab640eea557ea87032e9f4c"
expected_budget_sha256="bb73456a0fcf4bd3b5e0c56fd2ad93931c0d83a2d3c6803ad927e14976308fbf"

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

require_file_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  if [[ ! -f "${path}" ]]; then
    printf 'Missing %s: %s\n' "${label}" "${path}" >&2
    exit 2
  fi
  local actual
  actual=$(sha256_of "${path}")
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'SHA-256 mismatch for %s: %s\n' "${label}" "${path}" >&2
    printf 'expected %s\nactual   %s\n' "${expected}" "${actual}" >&2
    exit 4
  fi
}

cd "${project_dir}"

if [[ ! -d "${d8_root}" ]]; then
  printf 'Missing sealed D8 root: %s\n' "${d8_root}" >&2
  printf 'Restore it locally or regenerate D8/D9 before rerunning D10.\n' >&2
  exit 2
fi

require_file_sha256 \
  "${d8_root}/final/status.json" \
  "${expected_d8_final_status_sha256}" \
  "D8 final status"
require_file_sha256 "${d9_report}" "${expected_d9_report_sha256}" "D9 validation report"
require_file_sha256 "${d9_budget}" "${expected_budget_sha256}" "D9 formal budget"

if [[ -e "${rerun_root}" ]]; then
  printf 'D10 rerun output already exists under %s\n' "${rerun_root}" >&2
  printf 'Set D10_RERUN_ROOT to a fresh path or remove the previous local rerun output.\n' >&2
  exit 2
fi

mkdir -p "${receipt_dir}"

python_bin="${PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  if [[ -x "${workspace_dir}/.conda/envs/vitnns/bin/python" ]]; then
    python_bin="${workspace_dir}/.conda/envs/vitnns/bin/python"
  else
    python_bin="python3"
  fi
fi

export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m vitnns.stage_d_cli aggregate \
  --repo-root "${project_dir}" \
  --allow-formal-root \
  --output-root "${d8_root}" \
  --analysis-root "${analysis_root}" \
  --d9-validation-report "${d9_report}" \
  --d9-formal-budget "${d9_budget}" \
  > "${receipt_dir}/01_aggregate.json"

"${python_bin}" -m vitnns.stage_d_cli build-paper-package \
  --repo-root "${project_dir}" \
  --allow-formal-package \
  --analysis-root "${analysis_root}" \
  --paper-root "${paper_root}" \
  > "${receipt_dir}/02_build_paper_package.json"

"${python_bin}" -m vitnns.stage_d_cli validate-paper-package \
  --repo-root "${project_dir}" \
  --allow-formal-package \
  --paper-root "${paper_root}" \
  > "${receipt_dir}/03_validate_paper_package.json"

printf 'D10 lightweight rerun complete: %s\n' "${rerun_root}"
printf 'Receipts: %s\n' "${receipt_dir}"
