# Stage Y4-R1 CPU-only mechanism analysis recovery

Date: 2026-09-14

Token: `STAGE_Y4_CPU_ONLY_MECHANISM_ANALYSIS_AND_DECISION`

Status: prospective and outcome-blind at producer freeze.

Y4-R1 is the authorized recovery from the immutable Y4 `BLOCKED_ACCESS`
terminal. The predecessor root, report, commit and blocked archive ref are not
edited. The sole defect repaired here is the startup/publication-validation
path: Y2 and Y3 historical PASS evidence is bound at its own archive commit by
Git blob/LFS identity and source-root recomputation. A historical PASS ref is
not incorrectly required to equal the later current branch tip, and no legacy
validator that hashes JPEG/checkpoint bytes is invoked.

The run is CPU-only and reads no image or checkpoint bytes. It imports no
Torch, torchvision, timm, PIL or cv2, initializes no CUDA context, performs no
model call or empirical attempt, and accesses no network. Only frozen JSON,
CSV, NPY/NPZ scientific records and Git/LFS metadata are eligible. Source
arrays remain in Y2/Y3 roots and are never copied.

The implementation binds the exact Y0 registries, 100 Y2 clean source records,
20 clean perturbation references and 160 Y3 non-clean coordinates. It validates
every array hash, canonical digest, dtype, shape, axes, layout, finiteness,
target, condition and closure before float64 metric reduction. Image is the
only inferential unit. Shallow/middle/deep are blocks 0-3/4-7/8-11. The twelve
registered hypotheses, margins, directions, 20,000-bootstrap and
100,000-sign-flip rules, per-family Holm correction and six claim truth tables
remain byte-bound to Y0.

Before outcome access, the following RNG rule is fixed to remove stream
ambiguity: each registered hypothesis independently initializes
`numpy.random.Generator(numpy.random.PCG64(frozen_seed))`; it consumes one
row-major `(replicate, image)` draw matrix in ascending source/image order. A
two-sided sign-flip endpoint uses that same sign matrix first for `d-margin`
and then for `-d-margin`, without another RNG draw. Every descriptive trajectory
bootstrap uses an independent fresh initialization of the frozen bootstrap
seed and ascending image order. This rule is outcome-independent and frozen
with the producer.

For clean depth metrics, the registered aggregate-total block representation is
the block output complete cut (`bXX_output_cell`, `bXX_output_matrix`, and
`bXX_output_cls`). Within-patch uses aggregate attention-branch cells. Head and
branch endpoints use the registered first- and second-residual components. For
geometric perturbations, patch matrices are compared only after
`W @ T_transformed @ W.T`, with `W[clean_patch, transformed_patch]`, and only
fully-valid source/destination pairs. Brightness/noise use the stored identity
operator. Prediction-flip rows are retained.

The producer emits complete clean/non-clean primitive metrics, image-level
confirmatory aggregates, every block trajectory, every image/block/head row,
patch-mixing, branch-allocation and perturbation tables. Descriptive output is
never promoted. The independent validator ignores producer summaries, rereads
the sealed source records and arrays, regenerates statistics byte-
deterministically, and compares every derived table and decision.

Y4 workflow PASS depends on evidence, computation and publication integrity,
not on favorable claim outcomes. The only authorization after PASS is
`STAGE_Y5_RESULTS_TO_PAPER_HANDOFF_ONLY`; Y5, F3-Final, Stage G and paper writing
remain forbidden in this execution. Any blocker seals the fresh Y4-R1 root
with `next_authorization=NONE`.
