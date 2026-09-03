# Recovery C9 amendment — 2026-09-03

- Amendment ID: `RECOVERY-C9-SALVAGE-C8-20260903`
- Status: `APPROVED_FOR_RECOVERY_C9_ONLY`
- Scientific contract change: none
- C10 and Stage D authorization: none

This amendment authorizes one fresh C9 recovery replication only after a
source-recomputing validation of the complete C8 salvage-repair union.  The
canonical upstream is
`evidence/stage_c/c8_salvage_repair_recovery_v2_de71d55`; the original
interrupted C8 root and the prior partial salvage root are non-canonical
read-only origins.

The C8 gate must revalidate all 2,180 planned units from their three source
roots, retain exactly 2,179 succeeded units and one valid terminal scientific
failure, reject the eight recorded incomplete attempts, and find no active
claim, pending, incomplete, missing, tampered, or duplicate selected unit.  It
must reproduce the bound completion, combined-manifest, artifact-manifest,
selection, and selection-report digests before any of the 15 holdout images is
opened.  A lightweight remote checkout is insufficient unless these complete
local source roots have first been restored from the named custody commit or
equivalent external storage.

C9 keeps the frozen 20-image order and 5 calibration / 15 holdout roles.  Its
four Value-path configurations are `(0,9)`, `(4,2)`, `(6,10)`, and `(11,2)`,
with `(0,9)` primary.  The three Value methods, four global methods, 320-unit
expansion, targets, baseline, IG schedule, seeds, perturbations, fidelity,
retention, and resource caps are unchanged.  C9 must use the recovery-v2
state dict with SHA-256
`489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d`
and canonical tensor digest
`150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd`.

The C9 producer commit must contain all reviewed implementation and contracts.
Its direct child is a launch-only commit containing the C9 authority.  The
formal output root is fresh and immutable: it may not be overwritten, resumed,
or joined to an older C9 root.  Per-unit evidence records the real launch
commit and the recovery-v2 model identity.

C9 performs no bootstrap, Holm correction, or scientific decision.  It does
not authorize C10 or Stage D and grants no access to Stage D reserve images.
Any later C10 authority must bind the chain salvage C8 -> this C9 explicitly.
