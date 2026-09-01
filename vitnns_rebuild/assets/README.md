# Recovery assets

This directory contains the recovered inputs required to rebuild the active
ViT-NNS experiment chain. Large files are stored with Git LFS.

The ImageNet assets remain subject to the ImageNet license. Keep the remote
repository private and restrict LFS access to users who are authorized to use
the data.

Key files:

- `data/ILSVRC2012_val/`: the verified Stage C 500-image subset.
- `imagenet/ILSVRC2012_devkit_t12.tar.gz`: official devkit.
- `imagenet/ILSVRC2012_img_val.parts/`: split official validation archive.
- `models/`: official model formats and the recovered semantic candidate.
- `RECOVERY_STATUS.md`: provenance boundary and missing-artifact summary.
- `manifests/RECOVERY_HASHES.md`: checksums for recovered files.

Run `assemble_imagenet_val.sh` only when a complete validation tar is needed.
The Stage D reserve JPEGs are not extracted by default.
