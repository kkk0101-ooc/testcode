#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "${script_dir}/.." && pwd)
asset_dir="${project_dir}/assets"

(cd "${asset_dir}/imagenet/ILSVRC2012_img_val.parts" && \
  sha256sum --check --strict ../ILSVRC2012_img_val.parts.sha256)

expected_devkit='b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953'
expected_labels='313fc3f23c864dde9183c9a368809065765dd1b657a6904fded44e3e46d34604'
[[ $(sha256sum "${asset_dir}/imagenet/ILSVRC2012_devkit_t12.tar.gz" | awk '{print $1}') == "${expected_devkit}" ]]
[[ $(sha256sum "${asset_dir}/imagenet/imagenet_2012_validation_synset_labels.txt" | awk '{print $1}') == "${expected_labels}" ]]

expected_manifest='042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89'
actual_manifest=$(sha256sum "${project_dir}/manifests/stage_c_flat_val_500/manifest.csv" | awk '{print $1}')
[[ "${actual_manifest}" == "${expected_manifest}" ]]

printf 'recovery asset manifest and ImageNet part checks passed\n'
