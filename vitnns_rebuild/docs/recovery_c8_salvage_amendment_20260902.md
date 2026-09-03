# Recovery C8 salvage amendment — 2026-09-02

- Amendment ID: `RECOVERY-C8-SALVAGE-20260902`
- Status: `APPROVED_FOR_RECOVERY_C8_SALVAGE_ONLY`
- Scientific contract change: none
- Holdout access authorization: none
- C9, C10, and Stage D authorization: none

This amendment is a narrow evidence-custody overlay on
`RECOVERY-C7-RERUN-20260901`.  It exists solely to avoid repeating valid GPU
work after an engineering provenance check treated a mutable remote-tracking
reference as immutable.  It does not change samples, methods, targets,
thresholds, seeds, schedules, perturbations, resource caps, ranking, or any
later decision rule.

## Source incident and immutable boundary

The interrupted source is
`evidence/stage_c/c8_recovery_v2_a342003/formal`, retained by commit
`e0edf3e51906b45feff094174921ca8bfd7eacf8`.  It remains an interrupted,
non-final C8 root and must never be marked PASS, resumed, appended to,
overwritten, relabelled, or used by itself as canonical C8 evidence.  Its
producer is `a342003460bc290bc24c3eb9ace196287e807edb`; its execution/launch
commit is `10633700a268170bd7c6f0ee1c7a52f8d4d97610`.  Neither identity may be
replaced with `4a87e4d...` or a later verifier identity.

The control snapshot records 2,180 expected units: 1,460 succeeded, 76
incomplete, 3 claimed, and 641 pending.  Those counts are candidates for
salvage, not trusted outcomes.  The salvage verifier must derive its own
classification from the complete frozen plan and source bytes.

## Authorized reuse

Only a source unit that is recorded as succeeded and passes an individual,
source-recomputed validation may be inherited.  Validation must cover the
unit identity and plan index; selected attempt and status; exact manifest and
file inventory; every L0/L1/L2/L3 payload byte and SHA-256; result and
provenance; model, config, run-spec, protocol, registry, sample, method,
block/head, target, baseline, seed, schedule and perturbation bindings; runner
telemetry; lifecycle restoration; and all frozen resource caps.

An inherited record must preserve its original relative path, producer and
execution commit, status/manifest/provenance/result hashes, and artifact
inventory digest, and must be labelled
`origin=inherited_from_interrupted_root`.  It must also record the real
salvage verifier commit/tree, verification time, and rule version.  Incomplete,
claimed, pending, missing, tampered, terminal-failed, duplicate, or
validation-failed source units are not inherited.

## Fresh gap execution

The gap is the exact ordered complement of the validated inherited set within
the unchanged 2,180-unit C8 plan.  It is not a subset scientific plan: every
new attempt retains the original full-plan and preflight bindings.  Gap units
run only in a new empty repository-local evidence root and use the current
real salvage launch commit.  They must never claim the old producer or launch
identity.  The interrupted root remains read-only before, during, and after
execution.

Authorization, production, execution, and verification are separate identity
roles.  The implementation commit produces the salvage machinery; its direct
child may be a launch-only commit containing exactly the new authority.  Gap
attempts record the real execution HEAD (normally that launch-only commit),
while audits, combination, selection, and completion record the real verifier
HEAD and source-subtree tree.  A mutable remote-tracking ref is none of these
identities.  Later evidence-only custody commits are permitted only while the
authorized source subtree remains byte-identical.

The experimental root `evidence/stage_c/c8_salvage_recovery_v2_0c855f6` is
superseded because its runtime incorrectly wrote its authorization commit as
the execution identity after runtime source changes.  None of its units may be
inherited, resumed, combined, or selected.  Its preserved diagnosis is failure
evidence, not C8 scientific evidence; a new authority and fresh gap root are
required.

## Canonical salvage evidence

The final C8 object is a virtual, metadata-indexed union, not either physical
root and not a copied or linked single-root illusion.  Its ordered manifest
must cover every frozen plan unit exactly once and identify each entry as
`inherited_from_interrupted_root` or `newly_run_in_salvage_root`; rejected
source candidates are separately disclosed.  The two selected sets must be
disjoint and their union must be the exact 2,180-unit plan.  Validation always
reopens both source roots and recomputes each selected attempt and the combined
digest.  Any missing, extra, overlapping, incomplete, claimed, pending,
tampered, or otherwise invalid required unit prevents PASS.

The combined report binds both root inventories, this amendment, the C7 and
recovery-v2 authorities, the recovery-v2 model identity, the original and new
producer identities, the salvage verifier identity, the frozen contracts, the
ordered unit-origin manifest, and the selection digest.  It records
`holdout_accessed=false`, `bootstrap_performed=false`, `holm_performed=false`,
and `decision_performed=false`.

Selection is computed mechanically from the combined 2,180 records using the
unchanged C8 rules: five calibration images, 144 candidates, one winner per
depth band, one unique primary, six-decimal ranking, NA stability `-1`, NA
residual ratio `0`, and tuple
`(n_pos, m_fid, m_stab, m_res, -block_index, -head_index)`.  No image content,
prediction correctness, display output, C7 result, or holdout outcome may
alter selection.

## Downstream rebinding

C9 may begin only under a new authority that validates this salvage completion
record, combined report, selection, both producer identities, and combined
artifact digest.  The interrupted root is never the canonical C8 root.  C10
must trace through that new C9 root to this explicitly labelled salvage C8; it
must preserve inherited/new origins and may not present the result as a fresh
single-producer C8.  Stage D must renew D0-D6 against the chain `salvage C8 ->
new C9 -> new C10`; old C8/C9/C10 bindings and the historical C10 root cannot
authorize D7 or D8.  Scientific and access boundaries remain unchanged.
