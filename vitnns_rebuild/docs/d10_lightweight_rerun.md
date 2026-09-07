# D10 lightweight rerun

This branch is the lightweight runnable form of `vitnns_rebuild`: code, tests,
docs, environment locks, configuration, manifests, helper scripts, and asset
checksums are tracked. Large local payloads and generated roots are intentionally
ignored by Git and must be restored or downloaded before use.

## Branch

```bash
git checkout preserve-code-docs-d10-rerun-20260907
```

The branch was created from `stage-d-d3-pass` at
`c4df92ff05054712406caa5524c5d129da7dab42`.

## Environment

From the workspace root:

```bash
cd /data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild
scripts/bootstrap_conda.sh
```

For an already bootstrapped workspace:

```bash
cd /data01/home/yeziqi.831/nxtrec_wksp/testcode
source .conda/etc/profile.d/conda.sh
conda activate vitnns
cd vitnns_rebuild
python -m pip install -e . --no-deps
```

Pinned runtime: Python 3.10.21, torch 2.2.2+cu121, torchvision 0.17.2+cu121,
timm 0.9.16. The authoritative lock is `requirements.lock.txt`.

## D10-only inputs

D10 does not need to load the model, read JPEGs, or use CUDA. It requires the
already sealed numeric D8 root and the D9 validation report:

```text
evidence/stage_d/d8_recovery_v2_20260904/formal_r2
reports/stage_d_d9_recovery_v2_20260906/d9_formal_validation.json
config/stage_d_d7_budget_20260904_r4.json
```

Expected identity:

```text
D8 root inventory digest:
76d7d532049a5c90a09c74ba56b019042d7f2f0174ccb439bd8f67713514cc5a
D8 final/status.json SHA-256:
9c2a7f6bbe89f393b17d39128c908b764c5481cb2ab640eea557ea87032e9f4c
D9 report SHA-256:
bbfb6b5ad879116f6849e6f0a058780fdfd5e54b656e62f0674a3c3184c810d5
D9 canonical digest:
50e2c1455534e5b10aeacb79fac42d128317e0e943c817addf47671f599ea206
D9 budget SHA-256:
bb73456a0fcf4bd3b5e0c56fd2ad93931c0d83a2d3c6803ad927e14976308fbf
```

These inputs are not tracked by the lightweight branch. Restore them from local
artifact storage or regenerate D8 and D9 first.

## D10 command

The one-command rerun writes a fresh ignored local output root and JSON receipts:

```bash
scripts/rerun_stage_d_d10_lightweight.sh
```

To choose an output path:

```bash
D10_RERUN_ROOT=evidence/stage_d/d10_recovery_v2_20260906/rerun_lightweight_manual \
  scripts/rerun_stage_d_d10_lightweight.sh
```

The script runs:

```bash
CUDA_VISIBLE_DEVICES= python -m vitnns.stage_d_cli aggregate \
  --repo-root . \
  --allow-formal-root \
  --output-root evidence/stage_d/d8_recovery_v2_20260904/formal_r2 \
  --analysis-root <rerun-root>/analysis \
  --d9-validation-report reports/stage_d_d9_recovery_v2_20260906/d9_formal_validation.json \
  --d9-formal-budget config/stage_d_d7_budget_20260904_r4.json

CUDA_VISIBLE_DEVICES= python -m vitnns.stage_d_cli build-paper-package \
  --repo-root . \
  --allow-formal-package \
  --analysis-root <rerun-root>/analysis \
  --paper-root <rerun-root>/paper

CUDA_VISIBLE_DEVICES= python -m vitnns.stage_d_cli validate-paper-package \
  --repo-root . \
  --allow-formal-package \
  --paper-root <rerun-root>/paper
```

The historical D10 completion anchor remains:

```text
report: reports/stage_d_d10_recovery_v2_20260906/d10_formal_decision.json
report SHA-256: 7eac27b11b9df72a91c6e78cd0d46ea30f7d8701f0d41cbc29e9cb76695b5a08
canonical digest: db9b235cd8cfdeb3caf99445939443b17007ca5225c710c1164e36e2593503e6
output root: evidence/stage_d/d10_recovery_v2_20260906/formal_r2
output root inventory digest: b72d36430d149598ccd9cfd8dd586297e7231be29e61b44309e9971087e46535
```

The lightweight rerun regenerates the D10 analysis and paper package. It does
not pretend to recreate the historical Git completion ref from another branch.

## Network asset checklist for upstream rebuild

These assets are only needed if the D8/D9 inputs above are missing and the
upstream Stage D execution must be rebuilt.

| Role | Local path | Source | Expected identity |
| --- | --- | --- | --- |
| ImageNet validation tar | `assets/imagenet/ILSVRC2012_img_val.tar` | `assets/download_imagenet_val.sh` from `https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar` | MD5 `29b22e2961454d5413ddabcf34fc5622` |
| ImageNet split parts | `assets/imagenet/ILSVRC2012_img_val.parts/` | local split cache or re-split official tar | `assets/imagenet/ILSVRC2012_img_val.parts.sha256` |
| ImageNet devkit | `assets/imagenet/ILSVRC2012_devkit_t12.tar.gz` | ImageNet 2012 download page | SHA-256 `b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953` |
| Synset labels | `assets/imagenet/imagenet_2012_validation_synset_labels.txt` | derive from official devkit or restore audited copy | SHA-256 `313fc3f23c864dde9183c9a368809065765dd1b657a6904fded44e3e46d34604` |
| HF safetensors | `assets/models/model.safetensors` | `https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors` | SHA-256 `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b` |
| HF PyTorch state dict | `assets/models/pytorch_model.bin` | `https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/063c6c38a5d8510b2e57df480445e94b231dad2c/pytorch_model.bin` | SHA-256 `06e516abac6943dbba7b5dddfc39af6d63de8ebe6fa730240acfd32b94e74d19` |
| recovery-v2 runtime state dict | `assets/models/vit_base_patch16_224_weights.semantic_candidate.pth` | restore exact project-derived artifact or run a formal rebind | SHA-256 `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d`; tensor digest `150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd` |
| Stage D reserve JPEGs | `assets/data/ILSVRC2012_val/` | materialize the 500 rows from the official validation tar after D8 authorization | registry digest `514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028`; ordered sample ids SHA-256 `b187ff5cbacc1b595f06d12e309602b09965e01e7ee1461fd8a694354d4b8f50` |

After materializing ImageNet helper files and the Stage C subset, this existing
check should pass:

```bash
scripts/verify_recovery_assets.sh
```

The recovery-v2 model file is a project-derived runtime artifact. Simple
`torch.save` reserialization of the downloaded HF files is not guaranteed to
reproduce its file SHA-256, even though the tensor digest can match. Treat the
file SHA and tensor digest above as the binding that must be audited before any
upstream GPU rerun.

## Machine-readable checklist

The same checklist is tracked as:

```text
config/d10_lightweight_assets.json
```
