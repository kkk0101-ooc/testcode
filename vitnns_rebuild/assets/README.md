# Recovery assets

This directory is a local cache for recovered inputs required to rebuild the
active ViT-NNS experiment chain. On the lightweight rerun branch, Git tracks
only documentation, helper scripts, and checksum manifests. Large payloads
remain local and are restored or downloaded before a run.

The ImageNet assets remain subject to the ImageNet license. Only authorized
users should download or use them.

Key files:

- `data/ILSVRC2012_val/`: local materialization of the verified Stage C
  500-image subset; ignored by Git.
- `imagenet/ILSVRC2012_devkit_t12.tar.gz`: local official devkit; ignored by
  Git.
- `imagenet/ILSVRC2012_img_val.parts/`: local split official validation
  archive; ignored by Git.
- `models/`: local official model formats and the recovered semantic candidate;
  ignored by Git.
- `RECOVERY_STATUS.md`: provenance boundary and missing-artifact summary.
- `manifests/RECOVERY_HASHES.md`: checksums for recovered files.
- `../config/d10_lightweight_assets.json`: D10 lightweight rerun asset list.
- `../docs/d10_lightweight_rerun.md`: commands for rebuilding the runnable
  environment and rerunning D10.

Use `download_imagenet_val.sh` when authorized network access to ImageNet is
available. Use `assemble_imagenet_val.sh` only when the four local split parts
already exist and a complete validation tar is needed.
