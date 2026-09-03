# Recovery C8 salvage repair amendment — 2026-09-03

- Amendment ID: `RECOVERY-C8-SALVAGE-REPAIR-20260903`
- Status: `APPROVED_FOR_RECOVERY_C8_SALVAGE_REPAIR_ONLY`
- Scientific contract change: none
- Holdout access authorization: none
- C9, C10, and Stage D authorization: none

This amendment authorizes one narrow repair of an interrupted C8 salvage run.
It does not authorize ordinary C8, C9, C10, Stage D, holdout access, or a
change to any model, sample, method, target, seed, perturbation, resource cap,
ranking rule, or selection rule.

## Immutable custody inputs

The repair has exactly three evidence origins.

1. `inherited_from_original_interrupted_root`: the original read-only C8 root
   already bound by the original salvage authority.
2. `inherited_from_prior_partial_salvage_root`: individually valid terminal
   attempts preserved under
   `evidence/stage_c/c8_salvage_recovery_v2_ac77b47/gap` by custody commit
   `5e5ea9d01574c4ecf9f9e72f436c7063eedfe665`.
3. `newly_run_in_salvage_repair_root`: the exact complement that must execute
   under a new authority in a fresh repair root.

Both inherited roots are immutable, read-only evidence sources. They must not
be resumed, appended to, overwritten, relabelled, copied into a misleading
single-producer root, or independently presented as complete C8 evidence. The
repair authority binds the original salvage authority and its original source
root, plus the prior partial root's custody commit, Git tree objects, control
file hashes, and complete materialized inventory.

## Counts are observations, not authority

The preserved partial control state reports 568 succeeded units, one terminal
failed unit, and 151 unfinished units (143 pending and eight incomplete). These
figures describe the stopped run only. They are not trusted classifications
and must not be copied into the repair result without re-audit.

The repair verifier must reopen and validate every candidate against the full
2,180-unit plan. It must independently determine whether all 568 reported
successes remain valid, whether the one reported failed attempt is a valid
terminal scientific failure, and whether the exact remaining complement is
151 units. A mismatch blocks repair or changes the derived partition; no
reported count can override the bytes.

A failed attempt that passes the complete structural, provenance, resource,
and failure-semantics audit is terminal evidence. It is inherited with its
failure state and must not be rerun merely to seek a successful result. A
failed attempt that does not pass that audit is not inherited and cannot be
treated as valid evidence.

## Repair execution and final union

Only the source-recomputed complement may execute, and only under fresh
`repair`, `reports`, `selection`, and `logs` roots named in the repair launch
authority. The producer commit must contain the reviewed repair implementation
and contracts. Its direct child must be a launch-only commit containing only
the repair authority. Current runtime HEAD must descend from that launch and
must have an exactly identical `src/vitnns` subtree. Attempts record the real
execution HEAD; it must not be replaced by the producer, authority, custody,
or historical execution identity.

The final C8 repair object is a metadata-indexed three-source union. Every one
of the 2,180 plan units must appear exactly once with one of the three origins
above. The origin sets must be pairwise disjoint, their union must equal the
frozen plan, and each selected attempt must be revalidated from its source
bytes. The final report binds all three root inventories, both authorities,
both custody histories, the actual repair producer/execution/verifier
identities, the unchanged model and contracts, and the ordered unit manifest.

Selection may run only over that fully validated union and must use the
unchanged C8 rules. It records `holdout_accessed=false`,
`bootstrap_performed=false`, `holm_performed=false`, and
`decision_performed=false`. Any downstream stage requires a separate new
authority; this amendment itself grants none.
