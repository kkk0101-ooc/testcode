# Recovery-v2 machine binding and Stage A verification — 2026-09-01

- Status: `PASS`
- Scope: recovery-v2 machine/model binding and Stage A verification only
- Producer branch: `stage-d-d3-pass`
- Producer commit: `d8c2f84e94b354bc07c23ecc1dcaee1f6d96bcd8`
- Local tracking state at execution: `origin/stage-d-d3-pass` at
  `4a87e4d9cb71fe9c394bb2fdd5d4e71bbfe8aaed`; producer was one local commit
  ahead, zero behind, with the tracking ref verified as an ancestor. No remote
  push is part of this record.

## Environment and machine

The verification ran from the bound Conda environment:

```bash
cd /data01/home/yeziqi.831/nxtrec_wksp/testcode
source .conda/etc/profile.d/conda.sh
conda activate vitnns
cd vitnns_rebuild
```

Recorded environment identity:

| Item | Observed value |
|---|---|
| Conda prefix | `/data01/home/yeziqi.831/nxtrec_wksp/testcode/.conda/envs/vitnns` |
| Python | `3.10.21` |
| torch | `2.2.2+cu121` |
| torchvision | `0.17.2+cu121` |
| timm | `0.9.16` |
| CUDA available / visible device count | `true` / `1` |
| GPU | `NVIDIA H20` (`cuda:0`) |
| torch CUDA runtime | `12.1` |
| cuDNN | `8902` |

`python -m pip check` returned `No broken requirements found.` A separate
`128 x 128` CUDA matrix-multiplication smoke completed on `cuda:0` with sum
`2097152.0`.

## Recovery-v2 model identity

The closed machine binding is `config/recovery_v2_binding.json`, canonical
digest
`e49786a0648df60081419e997c0e5e2ec2fd61e38a30e979532852bb6e6c853e`.
It binds:

```text
path: assets/models/vit_base_patch16_224_weights.semantic_candidate.pth
file_sha256: 489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d
lfs_oid_sha256: 489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d
canonical_tensor_digest: 150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd
identity: RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718
```

The LFS object was hydrated and its OID matched the file SHA-256. The canonical
tensor digest is the identity frozen by the recovery amendment and was
recomputed here over all 152 tensors (86,567,656 parameters). This artifact is
parameter-semantically equivalent to the recovered candidates, but it is not
the missing historical binary with SHA-256
`2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`.
The historical identity remains unrecovered and was not reused.
Before Stage A, `git lfs status` reported no object pending commit or push. At
this local-only handoff, the four Stage A evidence objects are committed with
their LFS OIDs and remain pending a later explicit push; no remote publication
or retrievability claim is made by this record.

## Stage A result

The isolated recovery command was run on the clean producer commit with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `CUDA_VISIBLE_DEVICES=0`. Its immutable
evidence root is `evidence/stage_a/recovery_v2_d8c2f84`; all files are covered
by Git LFS. The terminal receipt is
`evidence/stage_a/recovery_v2_d8c2f84/stage_a_verification.json`, SHA-256
`fc0a6e673be2e6292444074ee0c5daf2d8c15d08ca87b259cd8f357dbb903de5`.

The bound sample was
`assets/data/ILSVRC2012_val/ILSVRC2012_val_00009888.JPEG` (SHA-256
`156ee1e37de4ffc4ef8b9b9642aecbda4e258638ea0cc6d0b67515598d6349f6`,
WNID `n02113978`, ImageNet-1k index `268`). Results:

| Gate | Result | Threshold |
|---|---:|---:|
| fused vs unfused logits max absolute error | `4.32133674621582e-06` | `< 1e-5` |
| hook-induced logits/block error | `0.0` | exact `0.0` |
| attention recomputation max absolute error | `0.0` | `< 1e-6` |
| top-5 parity | `PASS` | exact match |

Conclusion: this machine, environment, model artifact and Stage A path pass the
recovery-v2 verification and may support preparation of a later C7-C10 recovery
replication. This record does not recover the old sealed C10 confirmatory chain,
does not authorize a C7/C8/C9/C10 execution or Stage D D7/D8, and does not
authorize Stage D reserve access. A separate committed recovery launch record
is still required before C7.
