#!/usr/bin/env bash
set -euo pipefail

url='https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
asset_dir="${script_dir}/imagenet"
partial_file="${asset_dir}/ILSVRC2012_img_val.tar"
parts_dir="${asset_dir}/ILSVRC2012_img_val.parts"
assembled_file="${asset_dir}/ILSVRC2012_img_val.complete.tar"
total_bytes=6744924160
chunk_bytes=268435456
max_parallel=12

mkdir -p "${parts_dir}"

prefix_bytes=0
if [[ -f "${partial_file}" ]]; then
  prefix_bytes=$(stat -c '%s' "${partial_file}")
  prefix_part=$(printf '%s/part-%012d-%012d' "${parts_dir}" 0 $((prefix_bytes - 1)))
  if [[ ! -f "${prefix_part}" ]]; then
    mv "${partial_file}" "${prefix_part}"
  fi
fi

download_part() {
  local start=$1
  local end=$2
  local target expected actual
  target=$(printf '%s/part-%012d-%012d' "${parts_dir}" "${start}" "${end}")
  expected=$((end - start + 1))
  if [[ -f "${target}" ]]; then
    actual=$(stat -c '%s' "${target}")
    if [[ "${actual}" -eq "${expected}" ]]; then
      return 0
    fi
  fi
  curl -fsSL --retry 10 --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
    --range "${start}-${end}" "${url}" -o "${target}.partial"
  actual=$(stat -c '%s' "${target}.partial")
  if [[ "${actual}" -ne "${expected}" ]]; then
    printf 'wrong length for %s: expected %s, got %s\n' "${target}" "${expected}" "${actual}" >&2
    return 1
  fi
  mv "${target}.partial" "${target}"
}

pids=()
next_start=${prefix_bytes}
while [[ "${next_start}" -lt "${total_bytes}" ]]; do
  next_end=$((next_start + chunk_bytes - 1))
  if [[ "${next_end}" -ge "${total_bytes}" ]]; then
    next_end=$((total_bytes - 1))
  fi
  download_part "${next_start}" "${next_end}" &
  pids+=("$!")
  if [[ "${#pids[@]}" -eq "${max_parallel}" ]]; then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
  fi
  next_start=$((next_end + 1))
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

truncate -s "${total_bytes}" "${assembled_file}"
for part in "${parts_dir}"/part-*; do
  base=${part##*/}
  start=${base#part-}
  start=${start%%-*}
  dd if="${part}" of="${assembled_file}" bs=16M seek=$((10#${start})) \
    oflag=seek_bytes conv=notrunc status=none
done

actual_total=$(stat -c '%s' "${assembled_file}")
if [[ "${actual_total}" -ne "${total_bytes}" ]]; then
  printf 'assembled length mismatch: expected %s, got %s\n' "${total_bytes}" "${actual_total}" >&2
  exit 1
fi

printf '29b22e2961454d5413ddabcf34fc5622  %s\n' "${assembled_file}" | md5sum --check --strict
