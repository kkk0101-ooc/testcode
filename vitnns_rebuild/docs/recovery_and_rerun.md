# Recovery Assets and Rerun Guide

Last updated: 2026-09-04.

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
- Stage D recovery status: D0-D6 have current recovery-v2 R3 pre-reserve
  records. D0-D2 are formal preparation PASS; D3 is control-plane readiness;
  D4-D6 are implementation/readiness PASS based on metadata/synthetic fixtures.
- Stage D D7 recovery-v2 gate is in progress under
  [`recovery_stage_d_d7_amendment_20260904.md`](recovery_stage_d_d7_amendment_20260904.md).
  The non-reserve synthetic smoke and independent 1,000-unit budget have passed;
  execution remains unstarted and analysis authorization remains unissued.
- Stage C historical summary remains available in the documents. Recovery-v2
  C7, salvage C8, new C9, and new C10 evidence roots are present locally. C10
  build/integrity validation completed; acceptance is `BLOCKED` and its
  primary scientific decision is `NO_GO_NOVELTY`.

The historical Stage D machine-readable files still name historical C10 and
`2e718...` identities and remain immutable audit evidence. The versioned overlay
`config/stage_d_recovery_v2_binding_20260904.json` and corrected R3 D0-D6
records supersede those launch identities without rewriting the scientific
contract. D7 uses a separate versioned producer and authority-only commit; no
historical Stage D record is rewritten.

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

远程推送不是默认动作。默认情况下，每个 C/D recovery 任务结束时只做本地
commit，让 Git/LFS objects 保留在本地待推送状态。只有用户明确要求远程推送时，
才推送 Git refs 和 Git LFS objects。收到推送要求后，先用具备
`kkk0101-ooc/testcode` 写权限和足够 Git LFS 配额的 GitHub credential 完成认证，
再发布当前分支并验证 commit 与 LFS payloads：

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
The resulting recovery-v2 machine binding and Stage A PASS record are indexed in
`docs/recovery_v2_stage_a_verification_20260901.md`; that record is preparation
evidence only and is not a C7 launch authorization.

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

正式 evidence roots 默认保留在本机文件系统，不再整体纳入 Git/LFS 推送范围。
完成 formal run 后，应提交代码、文档、配置、schema、测试和必要的小型索引/报告；
大型 per-unit artifacts、NPY、完整 formal roots、combined manifests 和中间 execution plans
只在用户明确要求时另行打包或加入远端存储。不要运行 `git push` 或 `git lfs push`，
除非用户明确要求远程推送。cache 文件和临时重组 archive 必须放在 evidence root 之外。

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

## Recovery-v2 C7 progress update (2026-09-02)

Recovery-v2 C7 is now `PASS`. The implementation producer is
`026b622577370c50babbec6d1f26342875665c8e`, the separate clean launch and
execution commit is `b05307241337eff42757d7ef0638223034da50be`, and the
complete evidence is sealed by commit
`064a0b8d96e68d78ac652c3677c307cb8b745630` under
`evidence/stage_c/c7_recovery_v2_026b622`. The diagnostic gate is `PASS`; the
formal report is `PASS` with all `40/40` units complete and succeeded. Both
reports record `holdout_accessed=false`; full hashes and scope are in
`阶段C-step-by-step.md`.

This is a recovery replication and audit-chain reconstruction. It does not
recover the historical 15-image confirmatory C10 result and does not make the
historically exposed holdout unseen again. C8 has not been executed: there is
no recovery-v2 C8 root, selection, or PASS report. The existing launch record
is C7-only (`c8_authorized=false`), so C8 requires its own valid handoff and a
fresh empty evidence root before execution.

## Recovery-v2 C8 salvage progress update (2026-09-03)

Recovery-v2 C8 is now PASS through the amendment-authorized three-source
salvage chain in evidence/stage_c/c8_salvage_repair_recovery_v2_de71d55,
sealed by evidence custody commit
82825131a598d4fffc22ca761e4d9a30b5d35666 (root Git tree
74741c1e387c2c83aa6aeb5b90c38d45de53fcee). This supersedes the immediately
preceding C7 note's “C8 has not been executed” status; that older sentence is
retained only as a dated historical snapshot.

The canonical PASS object is reports/c8_salvage_repair_combined_report.json
(file SHA-256 974e9366f240018d6118edc47dfbd623d9690cb407961fcf73437cee07c7898b,
canonical digest 4e31223f9f4e8113b035d9d8ddeb221bec916278864873180593d106684d4612).
It binds 1460 original succeeded units, 569 valid terminal units from the
partial salvage root (568 succeeded plus one retained scientific failure), and
151 newly run succeeded repair units. The combined manifest covers exactly
2180 terminal units and has artifact digest
4a7f1bd35aadfb704196625c709f21c1fd51159dbae8d5b1fb9dae8f95043257.
The old interrupted and partial roots remain read-only source material and are
not themselves PASS roots.

The new selection is selection/stage_c_salvage_repair_selection.json (file
SHA-256 5d26c78bb94278c7f12d7bd88b1a37cfb97352f3d0bde5e1c4e52f8e811c7982,
canonical digest 6915f7c4383c91ed5fcc465a52d3582eebdb6cd325c7bc2107b03c7996c87620).
Its frozen winners are (0,9) primary and (4,2), (6,10), (11,2) exploratory.
The completion reports holdout_accessed=false, and no C9, C10, bootstrap,
Holm, decision, or Stage D work was started.

To keep remote pushes usable, the pushed branch is intentionally lightweight:
it stores the C8 salvage code, contracts, tests, amendments, handoff documents,
and small final report/selection records only. The complete multi-GB C8 roots,
per-unit artifacts, combined manifest payload, and execution plans remain local
source material and are preserved in the local backup branch
`backup/stage-d-d3-pass-full-20260903-before-light-push`.

The next step is a separately authorized recovery-v2 C9. Its authority must
bind the C8 salvage completion, combined manifest, selection, three source
inventories and producer identities, combined artifact digest, and custody
commit; it must reject the old interrupted C8 root as canonical. C10 must later
trace through this salvage C8 and a new C9 root. Stage D must wait for a newly
bound salvage C8 -> C9 -> C10 -> renewed D0-D6 chain. None of those downstream
stages is authorized by the C8 salvage amendment.

## Stage F 展示规划

不依赖未查看 Stage D 结果的论文内容可以与后续 recovery 实验并行推进：
引言、相关工作、方法、已封存的 C7/C8/C10 事实、限制讨论，以及未来图表占位。
依赖 Stage D 结果的结果段落和定性案例必须等待对应 evidence 封存与验证。

版本化展示协议为 `docs/stage_f_display_protocol_v1.md`。它必须在 Stage D 结果访问前冻结。
该协议没有统计决策权限，不改变 C7-C10 recovery，也不改变 Stage D 主实验。
D10 验证完成后，它允许两类展示集合：固定 D1 审计样本，以及按预注册规则从完整
500 图队列中选择的说明性案例。这些案例只用于可视化展示，不能改变 primary 结论。

## Recovery-v2 C9 completion update (2026-09-03)

Recovery-v2 C9 is `PASS` at
`evidence/stage_c/c9_recovery_v2_0d88fbb/formal`. It is explicitly bound to
the C8 salvage-repair PASS root, not to the old interrupted C8 root. Before
the first holdout read, the complete local 2,180-unit three-source C8 union
was source-recomputed and reproduced its completion, manifest, artifact, and
selection digests. The lightweight remote branch alone remains insufficient
for that check; the complete local C8 roots or an equivalent restoration are
required.

The C9 authority producer is `0d88fbb8b71c604d3dab94db19725e01b7519ae5`
and its authority-only execution commit is
`c66e5224e87924253f10a9f5fead006fcc5b5ac7`. The fresh formal run completed
all 320/320 units and all 15/15 primary holdout images passed the C9 validity
checks. The final report has file SHA-256
`ad164a87aa6dedba461e1b19d639d22d955a675d75e0c6e257af4d5846682763`
and canonical digest
`959bc874b04b007f487d05734cf5b85b1eaef75bb8c2ff2ded222c7cf9083248`;
an independent source validation reproduced the exact report bytes. The small
completion index is `reports/recovery_c9_completion_20260903.json`.

The complete C9 root (2,705 regular files, 225,985,972 bytes) and execution
logs remain local and are not committed; its ordered whole-root inventory
canonical digest is
`6ab8ee024ce7d5bfa3c6a408ec118a9eeac7960d046d7001d8019fb8ab6650c8`.
C9 performed no bootstrap, Holm correction, or decision. The following C10
completion update supersedes this dated C9 handoff's statement that C10 still
needed authorization. Stage D remains neither authorized nor started, and no
Stage D reserve image was accessed.

## Recovery-v2 C10 completion update (2026-09-03)

Recovery-v2 C10 was built and independently source-validated at
`evidence/stage_c/c10_recovery_v2_3cf2945/formal`. Processing is complete and
integrity is valid, but Stage C acceptance is `BLOCKED` because the frozen
Holm contract remains `HOLM_BLOCKED_UNDERSPECIFIED`; no raw or adjusted
p-values were invented and Holm was not run. The unique primary scientific
decision is independently `NO_GO_NOVELTY`.

The source chain is the canonical salvage C8 root -> recovery C9 producer
`0d88fbb...` / authority-execution `c66e5224...` -> C10 producer `3cf2945...` /
authority `0d0eab0...`. C9 source recomputation retained 320/320 succeeded
units, no claims/pending/incomplete/missing/tampered or duplicate attempts,
and 15/15 valid primary holdout images. The recovery-v2 model independently
reproduced file SHA-256 `489bfa2f...` and tensor digest `150234d8...`; the old
`d8190...` C9 producer and `2e718...` model are not inputs.

Frozen outcomes are: validity `PASS_VALIDITY`; effect/fidelity `PASS_EFFECT`
(median 1.7534972022995547, 15/15 positive, 95% paired image-bootstrap CI
[1.3317595184568727, 3.3813212483296478]); clean joint spatial
`PASS_JOINT_SPATIAL`; translation and center-crop/resize each
`FAIL_STABILITY`; Gaussian `PASS_STABILITY`; overall spatial `FAIL_SPATIAL`;
cost `PASS_COST`. `PASS_EFFECT` is only one component and does not override
the two stability hard-gate failures. The 10,000-resample bootstrap ran with
image as the cluster; calibration n=5, holdout n=15, and descriptive all-20
remain separate.

The complete 12-file C10 root is 524,798,470 bytes and remains local only; its
ordered whole-root inventory digest is
`b63a32ab2110081aae0a0b033d6a79f6798f611b20f3e6b0a3e769b5ff9b5627`.
The lightweight completion, validation, whole-root inventory, and downstream
handoff indexes are under `reports/`; the lightweight remote branch is not a
complete C8/C9/C10 evidence mirror.

The audited future Stage D scope remains fidelity-only and prospective: one
primary `(0,9)`, NNS versus the matched patch baseline, endpoint
`fine_4x4_cell_delta_fid`, a separate n=500 cohort, and
`run_stability=false`. C10 n=15 and Stage D n=500 must not be pooled. C10
translation/crop failures remain mandatory Study Design, Results, and
Limitations disclosures. Existing Stage D machine-readable files still bind
the historical C10 and `2e718...` model, so renewed D0-D6 versioned bindings
are required before D7 or any reserve access. This C10 record does not modify
Stage D scientific rules, authorize Stage D execution/analysis, start Stage D,
or access/stat/hash/decode/preprocess/infer any Stage D reserve image.

## Recovery-v2 Stage D D0-D6 completion update (2026-09-04)

The corrected current records are under
`reports/stage_d_recovery_v2_20260904_r3/`. They bind canonical salvage C8 ->
recovery C9 -> recovery C10 and model `489bfa...` / tensor digest `150234d8...`.
D0-D2 are preparation PASS; D3-D6 are readiness PASS, with D5/D6 explicitly
not real evidence validation or analysis.

The systematic inventory is `false_pass_audit.json`; the informational future-D7
handoff is `d6_to_future_d7_handoff.json`. R1 is retained for its wrong schema
URI and R2 for incomplete source coverage; neither was overwritten or described
as tamper. The remote branch stores lightweight records, not complete C8/C9/C10
evidence roots.

D7-D10 remain unstarted. No execution, analysis or reserve-access authority was
issued. No registry-referenced reserve image path was checked for existence,
statted, opened, hashed, decoded, preprocessed or used for inference.
