# Stage Y2 clean cohort evidence production

Protocol: `STAGE_Y_VIT_BLOCK_LEVEL_NNS_MECHANISM_V1_20260910`
Task token: `STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY`

Y2 first seals exactly 100 ImageNet-1k ILSVRC2012 validation objects from 100
distinct true classes using release metadata, source-object identity, raw-byte
SHA-256, and the frozen exclusion union. The audit-10 and perturbation-20
identities are sealed from that cohort before any selected JPEG is decoded or
the recovery-v2 model is loaded. Candidate JPEG bytes may be streamed only to
compute SHA-256; candidate image decoding is forbidden.

After the pre-execution seal, exactly eight homogeneous NVIDIA H20 workers,
one process per physical GPU, process independent image units under the clean
condition. Zero-based selection order is assigned by `selection_order mod 8`;
shards 00-03 contain 13 images each and shards 04-07 contain 12 each. Dynamic
rebalancing by class, prediction, runtime, convergence, attribution, or any
outcome is forbidden. Each image is an append-only unit with a PENDING receipt,
retained attempts, and one atomic typed terminal. The implementation reuses
the validated Y1 grouped JVP/VJP mathematical identity without a propagation
change and never materializes a pixel-by-hidden dense Jacobian. Required
complete cuts, native heads, two separate residual decompositions, signed
patch routing, CLS routing, head ablations, and hierarchical positive deletion
controls are retained under the formal root.

No non-clean empirical transform, cohort inference, bootstrap, Holm procedure,
claim decision, best-coordinate selection, display artifact, Y3-Y5 work,
F3-Final work, or Stage G work is part of Y2. `PASS` means only that the frozen
cohort and clean evidence package are complete and valid. It does not support
any scientific claim.

Each worker sees exactly one physical GPU through `CUDA_VISIBLE_DEVICES` and
uses only `cuda:0` internally. It records the physical index, visible-device
string, GPU UUID, PCI bus ID, device name, hostname, PID, and environment
digest in its isolated shard root. The shared cohort seal is immutable before
launch. Reconciliation is CPU-only and reads all eight shard roots; it never
decodes a JPEG, imports the model path, or accesses CUDA.

Formal root: `evidence/stage_y/y2_clean_formal_20260910/formal/`
Report root: `reports/stage_y_y2_20260911/`

On PASS the only next authorization is
`STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY`. Any blocked terminal has
`next_authorization=NONE`.
