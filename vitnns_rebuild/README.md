# ViT-NNS rebuild: auditable baseline and Stage C primitives

This directory is the clean, auditable replacement for the legacy experiment
tree. Stage A deliberately uses one model and one numerical forward path:

- `vit_base_patch16_224.augreg2_in21k_ft_in1k`
- the verified local pure `state_dict` only
- timm's real `forward` with non-mutating hooks
- the preprocessing embedded in this checkpoint's timm metadata

The legacy directories are inputs/archives and are never modified here. Large
models, images, tensors, and runtime outputs are excluded from Git.

Stage B adds a deliberately narrow, testable method instead of patching the
legacy scanner.  For a selected block it freezes the attention matrix actually
used at the input endpoint and attributes one pre-projection scalar of
`O = A @ V` back to the canonical model input with Integrated Gradients.  It
does not claim to be full Q/K/V attribution or a class-logit explanation.

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

## Stage B

Stage B implements conditional fixed-attention integrated gradients for one
block-internal pre-projection scalar `O_h = A_h @ V_h`. It consumes tensors from
the real timm forward path and does not replay a Transformer block. The complete
method boundary is in `docs/阶段B.md`; formal gates are in
`docs/stage_b_acceptance.md`.

```bash
vitnns-stage-b estimate --config config/stage_b.json

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 vitnns-stage-b attribute \
  --config config/stage_b.json \
  --device cuda \
  --relative-image n03976467/n03976467_9546.JPEG \
  --output-dir outputs/stage_b
```

The default schedule is `[512, 1024, 2048, 4096, 8192]`. The checked-in
configuration caps output at 64 MiB and caps batched forward and backward
model invocations at 30,000 each. A development run is not formal unless saved
provenance explicitly records `formal=true` on a clean committed tree with
every required gate enabled and passing.

## Stage C

Stage C now includes strict config/map/metric contracts, the registered
attention and class-logit methods, fixed-clean-attention head-direction
Value-path methods, C5 evaluation primitives, the C6 auditable execution
pipeline, and the C7 two-stage benchmark gate and report. See
`docs/阶段C_protocol.md`, `docs/stage_c_acceptance.md`, and the live handoff log
`docs/阶段C-step-by-step.md`.

C6/C7 expose non-interactive `estimate`, `run-sample`, `run-batch`,
`validate`, `aggregate`, `c7-diagnostic-gate`, and `benchmark-report`
commands. Every invocation requires both the frozen Stage C
config and an explicit JSON run spec; there are no implicit method, sample,
candidate, fidelity, stability, or repeatability defaults. For example:

```bash
vitnns-stage-c estimate \
  --config config/stage_c.json \
  --run-spec /path/to/explicit-run-spec.json

CUDA_VISIBLE_DEVICES=0 vitnns-stage-c run-sample \
  --config config/stage_c.json \
  --run-spec /path/to/explicit-single-sample-run-spec.json \
  --sample-id ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG \
  --output-root /path/to/new-stage-c-output \
  --device cuda

vitnns-stage-c validate \
  --config config/stage_c.json \
  --run-spec /path/to/explicit-run-spec.json \
  --output-root /path/to/stage-c-output
```

Preflight runs before model/GPU initialization or output creation. Resume is
on by default and skips only a complete attempt whose status, manifest, every
artifact hash/byte count, and current plan/provenance bindings validate. Failed,
incomplete, or tampered attempts are preserved and a retry receives a fresh
directory; `--no-resume` also creates a fresh attempt. `aggregate` emits only
validated L0 rows and never performs C8 selection, bootstrap, or a scientific
decision.

C7 has an explicit two-stage launch boundary. First run the checked-in
single-unit diagnostic spec from a clean commit into a fresh output root, then
write its gate report outside that source tree:

~~~bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
vitnns-stage-c run-sample \
  --config config/stage_c.json \
  --run-spec config/stage_c_c7_diagnostic_run_spec.json \
  --sample-id ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG \
  --output-root /path/to/new-c7-diagnostic-root \
  --device cuda:0 \
  --no-resume

vitnns-stage-c c7-diagnostic-gate \
  --config config/stage_c.json \
  --run-spec config/stage_c_c7_diagnostic_run_spec.json \
  --output-root /path/to/new-c7-diagnostic-root \
  --output /path/to/c7-diagnostic-gate.json \
  --device cuda:0
~~~

The diagnostic is exactly one complete block-0/head-0 NNS workload. Its
validated terminal telemetry must identify an H20 run from the current clean
commit, include lifecycle, phase, call, CUDA-memory, and IG evidence, and have
a synchronized end-to-end wall time strictly below 720 seconds. Equality or a
larger value is `BLOCKED`.

Only after that gate passes may the exact C7 plan start. The CLI requires the
gate before constructing a backend or creating the formal output root:

~~~bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
vitnns-stage-c benchmark \
  --config config/stage_c.json \
  --run-spec config/stage_c_c7_run_spec.json \
  --output-root /path/to/new-c7-formal-root \
  --device cuda:0 \
  --c7-diagnostic-gate /path/to/c7-diagnostic-gate.json

vitnns-stage-c benchmark-report \
  --config config/stage_c.json \
  --run-spec config/stage_c_c7_run_spec.json \
  --output-root /path/to/new-c7-formal-root \
  --output /path/to/stage-c-c7-benchmark-report.json \
  --c7-diagnostic-gate /path/to/c7-diagnostic-gate.json \
  --device cuda:0
~~~

The formal plan is exactly one audit image, 12 fixed Value candidates, seven
methods, and 40 execution units. The report binds the run spec, execution
plan, batch controls, every selected attempt's manifest/status/provenance, the
runtime environment, measured calls/memory/phase timings/bytes/IG attempts,
and the exact C8 (2,180), C9 (320), and Stage D (8,000) projections. Time uses
the slowest validated synchronized unit total and the frozen 1.25 safety
factor. Storage includes every ordinary source-tree file and the frozen 25%
control/provenance overhead. Historical or unselected attempts are never
eligible as benchmark measurements. The report remains an engineering and
budget gate: it does not access holdout data or perform selection, bootstrap,
or a scientific Go/No-Go decision.

### Stage C8 calibration and selection

C8 uses the checked-in `config/stage_c_c8_run_spec.json`. Its exact plan has
five calibration images, all three Value-path methods at every 12x12
block/head candidate, and four global methods: 436 units per image and 2,180
units total. C8 planning does not validate the 15 holdout JPEGs, and its image
loader rejects any sample outside this plan.

Initialize a new empty root from a clean commit and the frozen C7 report:

~~~bash
vitnns-stage-c c8-init \
  --config config/stage_c.json \
  --run-spec config/stage_c_c8_run_spec.json \
  --c7-report /path/to/frozen-c7-report.json \
  --output-root /path/to/new-c8-formal-root
~~~

Run deterministic, disjoint plan-order shards. Every invocation revalidates
the C7 report, measured preflight, clean Git state, plan, and checkpoint before
loading the model. For example, launch shard indices 0 through 5 on six GPUs:

~~~bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
vitnns-stage-c c8-run-shard \
  --config config/stage_c.json \
  --run-spec config/stage_c_c8_run_spec.json \
  --output-root /path/to/new-c8-formal-root \
  --shard-index 0 --shard-count 6 --device cuda:0
~~~

After all shards exit, finalize the root and generate both immutable outputs
from source-validated evidence:

~~~bash
vitnns-stage-c c8-finalize \
  --config config/stage_c.json \
  --run-spec config/stage_c_c8_run_spec.json \
  --output-root /path/to/new-c8-formal-root

vitnns-stage-c c8-select \
  --config config/stage_c.json \
  --run-spec config/stage_c_c8_run_spec.json \
  --output-root /path/to/new-c8-formal-root \
  --selection-output selection/stage_c_selection.json \
  --report-output /path/to/stage-c-c8-selection-report.json
~~~

C8 has no resume or caller-selected unit filter. A claimed, failed, incomplete,
tampered, over-cap, or interrupted root remains evidence; a repair uses another
fresh root. C8 performs no holdout inference, bootstrap, Holm correction, or
Go/No-Go decision.

### Stage C9 frozen pilot

C9 uses the checked-in `config/stage_c_c9_run_spec.json` and the tracked C8
selection. Its exact plan is the registry's 20 images in `selection_order`,
three Value-line methods at the four frozen band winners, and each of the four
global class-line methods once per image: 16 units/image and 320 units total.
The global fidelity pairing context is the frozen primary `(block 0, head 9)`.

Initialize a nonexistent formal-root path only after committing the C9 code.
Initialization revalidates the complete C8 source evidence, both selection
copies, commit lineage, exact plan, and the ten frozen budget caps before it
creates the root or any backend can read a holdout image:

~~~bash
vitnns-stage-c c9-init \
  --config config/stage_c.json \
  --run-spec config/stage_c_c9_run_spec.json \
  --c8-root /data00/home/yeziqi.831/testcode/vitnns_c8_formal_1fd2539_20260823 \
  --c8-report /data00/home/yeziqi.831/testcode/vitnns_c8_selection_report_final_d7f2ef3_20260824.json \
  --c8-recovery-selection /data00/home/yeziqi.831/testcode/vitnns_c8_selection_recovery_d7f2ef3_20260824.json \
  --output-root /path/to/new-c9-formal-root

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
vitnns-stage-c c9-run-shard \
  --config config/stage_c.json \
  --run-spec config/stage_c_c9_run_spec.json \
  --output-root /path/to/new-c9-formal-root \
  --shard-index 0 --shard-count 8 --device cuda:0

vitnns-stage-c c9-finalize \
  --config config/stage_c.json \
  --run-spec config/stage_c_c9_run_spec.json \
  --output-root /path/to/new-c9-formal-root

vitnns-stage-c c9-validate \
  --config config/stage_c.json \
  --run-spec config/stage_c_c9_run_spec.json \
  --output-root /path/to/new-c9-formal-root
~~~

Shard membership is computed internally from plan order. Formal C9 never
resumes or accepts a caller-selected unit. A terminal failed unit remains in
the same batch as failure evidence; any incomplete or tampered root is blocked,
retained, and a repair uses a different, previously nonexistent root path.
Finalization and read-only validation revalidate all 320
status/manifest/artifact/provenance chains and
the complete C8 source evidence, check shared input and perturbation hashes,
recompute all ten frozen caps from C8+C9 measurements, and report primary
15/15 validity as evidence only. They do not aggregate statistics, bootstrap,
apply Holm correction, or emit a scientific decision.

