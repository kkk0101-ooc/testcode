# ViT-NNS rebuild: Stage A baseline

This directory is the clean, auditable replacement for the legacy experiment
tree. Stage A deliberately uses one model and one numerical forward path:

- `vit_base_patch16_224.augreg2_in21k_ft_in1k`
- the verified local pure `state_dict` only
- timm's real `forward` with non-mutating hooks
- the preprocessing embedded in this checkpoint's timm metadata

The legacy directories are inputs/archives and are never modified here. Large
models, images, tensors, and runtime outputs are excluded from Git.

## Environment

```bash
source ~/.bashrc
conda activate vitnns
cd '/data00/home/yeziqi.831/testcode/vitnns_rebuild'
python -m pip install -e . --no-deps
```

Core versions are pinned in `requirements.lock.txt`. The CUDA wheels for torch
and torchvision come from `https://download.pytorch.org/whl/cu121`.

## Dataset manifests

Synset-folder data obtains its target from the canonical ImageNet-1k WNID
mapping, never directory enumeration order:

```bash
vitnns-stage-a manifest \
  --data-root '../49_vit /datas/imagenet-scanner' \
  --output-dir artifacts/imagenet-scanner
```

Flat ILSVRC validation images require the verified official devkit. Without it,
they can only be emitted with explicit null labels:

```bash
vitnns-stage-a manifest \
  --data-root '../49_vit /datas/ILSVRC2012_val' \
  --devkit /path/to/ILSVRC2012_devkit_t12.tar.gz \
  --output-dir artifacts/imagenet-val-subset
```

## Stage A verification

Use a clean Git commit so that `provenance.json` identifies immutable code:

```bash
CUDA_VISIBLE_DEVICES=4 vitnns-stage-a verify \
  --device cuda \
  --output-dir outputs/stage_a
```

For deterministic CUDA execution, export `CUBLAS_WORKSPACE_CONFIG=:4096:8`; the
runner enables deterministic PyTorch algorithms and disables cuDNN benchmarking.

The command requires all of the following:

- state-dict hash and strict model load pass;
- standard fused forward vs extractor logits maximum error `< 1e-5`;
- hooks do not alter the unfused forward (bitwise-equal blocks/logits);
- recomputed attention vs the attention actually used by timm `< 1e-6`;
- top-1/top-5 and all required provenance are derived from raw logits.

The full-model `.pth` is a Python pickle. It is not a production dependency. A
hash-gated compatibility check exists only behind both
`--compare-legacy-pickle` and an explicit unsafe loader opt-in in code.

## Tests

```bash
python -m pytest -m 'not unsafe_pickle'
VITNNS_ALLOW_UNSAFE_PICKLE=1 python -m pytest -m unsafe_pickle
```

The second command deserializes the known local pickle and should only run when
the artifact is trusted.

