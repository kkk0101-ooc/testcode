# Recovery Assets and Rerun Guide

Last updated: 2026-09-01.

This repository uses `vitnns_rebuild` as the active implementation. The old
`49_vit ` tree has been removed from the active checkout; use Git history for
legacy source inspection. Do not run legacy scripts as part of the current
experiment chain.

The authoritative recovery overlay is
[`recovery_protocol_amendment_20260901.md`](recovery_protocol_amendment_20260901.md).
It governs the recovery-v2 model identity, C7--C10 rerun, evidence custody, and
Stage D rebinding while preserving the historical protocols and records.

## Current state

- Recovery asset baseline commit: `7697f4a86adfb8490f6b07ab5725e0998d016e65`;
  amendment parent commit: `33f814723e03dfca419fbeb90de65e280590ba2f`.
  The containing amendment commit is the current recovery authority.
- Stage D implementation status: D0-D6 are implemented and tested on
  synthetic fixtures.
- Stage D formal execution status: not authorized and not started.
- Stage C historical summary remains available in the documents, but the
  C7/C8/C9/C10 runtime evidence roots are not present.

Current D7 cannot be started from this checkout because D3/D7 validation
requires the historical sealed C10 root, and that root is absent. Without an
external backup, the earliest auditable recovery-replication point is C7 under
the versioned recovery/protocol amendment. The original 15 holdout outcomes are
historically known, so rebuilding the same chain does not make them unseen or
newly confirmatory.

## Environment

A local Conda installation and the `vitnns` environment are stored at the
repository root:

```bash
cd /data01/home/yeziqi.831/nxtrec_wksp/testcode
source .conda/etc/profile.d/conda.sh
conda activate vitnns
cd vitnns_rebuild
python --version
python -m pip check
```

This is a project-local Conda environment, not the system Python. The
environment is rebuildable from `requirements.lock.txt` and is intentionally
not committed. A local `.venv/` is retained only as a temporary fallback.

Expected versions:

- Python 3.10.21
- torch 2.2.2+cu121
- torchvision 0.17.2+cu121
- timm 0.9.16

## Git LFS

Large assets and future experiment evidence are tracked through Git LFS. A new
machine must have Git LFS installed before cloning or pulling the asset commit:

```bash
git lfs install
git lfs pull
```

Before the first push of this recovery commit, authenticate Git with a GitHub
credential that has write access to `kkk0101-ooc/testcode` **and** enough Git
LFS storage quota for the approximately 7.4 GB asset set. Then publish the
current branch and verify both the commit and LFS payloads:

```bash
git push origin stage-d-d3-pass
git lfs status
git ls-remote --heads origin stage-d-d3-pass
```

This repository keeps `.conda/` and `.venv/` locally but ignores them in Git.
A fresh clone should create the `vitnns` Conda environment from
`requirements.lock.txt` or use a compatible Python 3.10 CUDA environment.

The current tracked LFS classes are:

- `vitnns_rebuild/assets/models/*`
- `vitnns_rebuild/assets/data/**/*.JPEG`
- `vitnns_rebuild/assets/imagenet/ILSVRC2012_img_val.parts/part-*`
- `vitnns_rebuild/assets/imagenet/*.tar.gz`
- `vitnns_rebuild/evidence/**`
- `*.pt`, `*.pth`, `*.npy`, `*.npz`, `*.parquet`

## Assets

All recovered assets live under `vitnns_rebuild/assets/`.

Verified ImageNet resources:

```text
assets/imagenet/ILSVRC2012_img_val.parts/part-00..part-03
assets/imagenet/ILSVRC2012_img_val.parts.sha256
assets/imagenet/ILSVRC2012_img_val.tar.md5
assets/imagenet/ILSVRC2012_devkit_t12.tar.gz
assets/imagenet/imagenet_2012_validation_synset_labels.txt
```

The complete validation tar is split to stay below GitHub LFS per-file limits.
Reassemble it only when needed:

```bash
cd /data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild
assets/assemble_imagenet_val.sh
md5sum -c assets/imagenet/ILSVRC2012_img_val.tar.md5
sha256sum -c assets/imagenet/ILSVRC2012_img_val.parts.sha256
```

To validate the four versioned parts without creating the redundant 6.74 GB
combined tar, run:

```bash
assets/assemble_imagenet_val.sh --verify-only
```

The restored Stage C image root is:

```text
assets/data/ILSVRC2012_val
```

It contains exactly the 500 images from
`manifests/stage_c_flat_val_500/manifest.csv`. A regenerated manifest with the
same basename is byte-identical to the tracked CSV and has SHA-256:

```text
042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89
```

The Stage C 500-image set has zero sample-ID and zero image-SHA overlap with
the Stage D 500-image registry. The Stage D reserve images were not extracted
or individually decoded during recovery.

Recovered model candidates:

```text
assets/models/model.safetensors
assets/models/pytorch_model.bin
assets/models/vit_base_patch16_224_weights.semantic_candidate.pth
```

These three files are parameter-equivalent: 152 key set, 152/152 tensors
bitwise equal, and 86,567,656 parameters. They are not byte-identical to the
historical frozen state-dict artifact:

```text
required historical state dict SHA-256:
2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0
```

Do not place the semantic candidate into the historical formal path and claim
it is the old artifact. A new run must explicitly version and freeze the new
model artifact identity.

The current historical Stage C/C/D contracts still bind the old state-dict
SHA-256 `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`.
The active amendment selects a recovery-v2 identity, but formal use remains
blocked until a later clean commit updates and validates every machine binding
and passes Stage A verification.

## Validation checks

Environment:

```bash
source /data01/home/yeziqi.831/nxtrec_wksp/testcode/.conda/etc/profile.d/conda.sh
conda activate vitnns
cd /data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild
python -m pip check
python -c 'import torch,timm; print(torch.__version__, timm.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

Stage C data:

```bash
vitnns-stage-a manifest \
  --data-root assets/data/ILSVRC2012_val \
  --devkit assets/imagenet/ILSVRC2012_devkit_t12.tar.gz \
  --output-dir /tmp/vitnns_stage_c_manifest_check

sha256sum /tmp/vitnns_stage_c_manifest_check/manifest.csv
cmp manifests/stage_c_flat_val_500/manifest.csv /tmp/vitnns_stage_c_manifest_check/manifest.csv
```

D6 implementation smoke:

```bash
python -m pytest -q tests/test_stage_d_analysis.py tests/test_stage_d_cli.py
PYTHONPATH=tests/d6_guard:src VITNNS_D6_STRICT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  python -m pytest -q tests/test_stage_d_d6_guard.py
```

## Rerun path

If an external backup restores the exact historical state dict and the exact
sealed C10 root, validate them first and then continue with D7.

If no external backup exists, follow this order:

1. Use and validate the committed `RECOVERY-C7-RERUN-20260901` amendment.
2. Bind its recovery-v2 model file and canonical tensor digest in every
   applicable machine contract, commit that binding, pass Stage A verification,
   and issue the separate C7 recovery launch record required by the amendment.
3. Re-run C7 diagnostic and benchmark from a clean commit.
4. Re-run C8 calibration selection from a fresh empty root.
5. Re-run C9 pilot from a fresh empty root.
6. Rebuild C10 from sealed C9 evidence.
7. Rebind/revalidate Stage D D0-D6 against the new frozen identities.
8. Run D7. Do not start D8 until D7 issues a valid execution authorization.

The tracked `selection/stage_c_selection.json` records the historical selected
heads, but it does not replace C8's source evidence. Do not skip C8 if the goal
is to restore the original audit chain.

The rebuilt C10 must apply the frozen calculations for reproducibility but be
labeled a recovery replication, not a new confirmatory result. A new
confirmatory claim requires a separately frozen, previously unseen registry.

## Persisting future formal outputs

Do not use `/tmp`, a home-directory cache, `outputs/`, or `artifacts/` for any
formal C7--D8 evidence that must survive a machine reset. Create each new
formal run root under the versioned `evidence/` tree and give its `--output-root`
an explicit path there, for example:

```bash
mkdir -p evidence/stage_c/c7_diagnostic_YYYYMMDD
vitnns-stage-c run-sample \
  --output-root evidence/stage_c/c7_diagnostic_YYYYMMDD \
  # remaining frozen arguments from the authorized run specification
```

`evidence/**` is covered by Git LFS. After a formal run reaches its sealed
terminal state, review it, commit it, and run `git push` before deleting the
local source or moving to the next stage. Keep cache files and temporary
reassembled archives outside that evidence root.

## Useful command skeletons

C7 diagnostic:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
vitnns-stage-c run-sample \
  --config config/stage_c.json \
  --run-spec config/stage_c_c7_diagnostic_run_spec.json \
  --sample-id ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG \
  --data-root assets/data/ILSVRC2012_val \
  --state-dict /path/to/newly-frozen-state-dict.pth \
  --output-root /path/to/fresh-c7-diagnostic-root \
  --device cuda:0 \
  --no-resume
```

C8 shard example:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
vitnns-stage-c c8-run-shard \
  --config config/stage_c.json \
  --run-spec config/stage_c_c8_run_spec.json \
  --data-root assets/data/ILSVRC2012_val \
  --state-dict /path/to/newly-frozen-state-dict.pth \
  --output-root /path/to/fresh-c8-root \
  --device cuda:0 \
  --shard-index 0 \
  --shard-count 6
```

C9 in the current historical code path does not expose `--data-root` or
`--state-dict` overrides. A recovery amendment should either restore the exact
historical defaults or update the C9/D7 code and contracts to bind the new
asset paths explicitly. Do not run C9 from the current historical contract with
the semantic candidate while pretending it is the old `2e718...` artifact.
