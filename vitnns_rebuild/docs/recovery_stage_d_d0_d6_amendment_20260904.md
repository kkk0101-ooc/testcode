# Stage D D0-D6 recovery-v2 pre-reserve amendment — 2026-09-04

- Amendment ID: `STAGE_D_PRE_RESERVE_RECOVERY_V2_20260904`
- Status: `PRE_RESERVE_PREPARATION_ONLY`
- Scientific contract change: none
- Stage D execution authorization: not issued
- Stage D analysis authorization: not issued
- Stage D reserve-access authorization: not issued

This amendment is a versioned identity overlay on the historical D0-D6
records. It does not edit, erase, or backdate them. Historical D0-D2 preparation
PASS records were valid for their original inputs but are stale for recovery-v2
because they bind the historical C10 and `2e718...` model. Historical D3-D6
PASS records are synthetic implementation evidence, not formal Stage D results.

The current source chain is canonical salvage C8 at
`evidence/stage_c/c8_salvage_repair_recovery_v2_de71d55`, recovery C9 at
`evidence/stage_c/c9_recovery_v2_0d88fbb/formal`, and recovery C10 at
`evidence/stage_c/c10_recovery_v2_3cf2945/formal`. The model is
`assets/models/vit_base_patch16_224_weights.semantic_candidate.pth`, file/LFS
SHA-256 `489bfa2f...`, canonical tensor digest `150234d8...`, and is explicitly
not the missing historical `2e718...` artifact. Full identities are closed in
`config/stage_d_recovery_v2_binding_20260904.json`.

The frozen Stage D scope remains prospective fidelity-only replication/extension:
separate `n=500`, primary `(0,9)`, NNS versus matched patch contribution,
`fine_4x4_cell_delta_fid`, image statistical units, `run_stability=false`, no
secondary inference, and no C10/Stage D pooling. Stage D Holm is only
`NOT_APPLICABLE_NO_SECONDARY_INFERENCE`; it does not repair C10
`HOLM_BLOCKED_UNDERSPECIFIED`.

C10 remains build-complete and integrity-valid but acceptance `BLOCKED`, primary
decision `NO_GO_NOVELTY`, effect component `PASS_EFFECT`, translation and
center-crop/resize `FAIL_STABILITY`, overall spatial `FAIL_SPATIAL`, and
downstream authorization false. Stage D is not a C10 reanalysis or redetermination.

D0-D6 may issue only preparation/readiness PASS records after direct-upstream,
exact-source, schema, canonical-digest, Git and access-boundary validation. D4-D6
remain synthetic/implementation-only. D7 is not executed or authorized here; D8
is not initialized. Live reserve identity remains deferred to an independently
authorized D8.0 boundary. No reserve image path may be checked for existence,
statted, opened, hashed, decoded, preprocessed, or used for inference in D0-D6.

Each stage first writes nonterminal completion/validation material and a pending
status, validates it, then atomically replaces the pending filename with a
terminal status. Each completion is anchored by a stage-specific archive ref.
The remote branch contains only lightweight records and is not a complete
C8/C9/C10/D0-D6 evidence mirror.
