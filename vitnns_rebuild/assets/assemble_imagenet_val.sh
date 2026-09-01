#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
imagenet_dir="${script_dir}/imagenet"
parts="${imagenet_dir}/ILSVRC2012_img_val.parts"
output="${imagenet_dir}/ILSVRC2012_img_val.tar"

case "${1:-assemble}" in
  assemble)
    ;;
  --verify-only)
    (cd "${parts}" && sha256sum --check --strict ../ILSVRC2012_img_val.parts.sha256)
    exit 0
    ;;
  *)
    printf 'Usage: %s [--verify-only]\n' "$0" >&2
    exit 2
    ;;
esac

(cd "${parts}" && sha256sum --check --strict ../ILSVRC2012_img_val.parts.sha256)
cat "${parts}"/part-* > "${output}"
(cd "${imagenet_dir}" && md5sum --check --strict ILSVRC2012_img_val.tar.md5)
