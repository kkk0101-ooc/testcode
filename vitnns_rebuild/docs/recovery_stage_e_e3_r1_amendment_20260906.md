# Stage E E3-R1 prospective numerical-recovery amendment — 2026-09-06

- Amendment ID: `STAGE-E-E3-R1-NUMERICAL-RECOVERY-20260906`
- Status: `APPROVED_FOR_E3_R1_GATED_RECOVERY_ONLY`
- Outcome-awareness: prospective after the terminal E3 v1 convergence failure
- Scientific change: NNS IG numerical schedule extension only, after exhaustion
  of the original frozen schedule
- E4 authorization: none until a complete terminal E3-R1 PASS

This amendment authorizes a fresh E3-R1 recovery path. It does not reopen,
replace, relabel, or convert E3 v1. The historical E3 v1 result remains a
terminal `FAIL`, and its receipt, costs, and 516 valid terminal attempts remain
immutable evidence. E3-R1 is not a retry under the E3 v1 identity and is not a
claim that E3 v1 passed.

## Immutable E3 v1 binding

E3-R1 binds the E3 v1 custody commit
`52b0847939a57e54cc72e590fbd9682a95d08444`, tree
`278058f37c28b30ddeb31eadac339812888399c3`, archive ref
`refs/archive/stage-e-e3-v1-terminal-fail`, and the following exact records:

- root: `evidence/stage_e/e3_recovery_v2_20260906/formal`;
- terminal completion: `completion/e3_completion.json`, file SHA-256
  `75355b40fe9df19ae8eedd10ef96bacf0f2eab49989894940350790a0a57b4bd`,
  canonical digest
  `91e76bca8d36173f5fae076159e5e09934ef3c25a902d22bc721787108d82011`,
  status `FAIL`, and `next_authorization=NONE`;
- terminal status: `status.json`, file SHA-256
  `a1e815e57711e88cff022861a4b2d6da65e7bd9ce8fa10b1efd41b396952afb2`,
  canonical digest
  `490f69f83a7b841e32857338d0daa86734e5b618c375b7ccf26365a9c8e89a7a`;
- typed plan: `control/typed_plan.json`, file SHA-256
  `9de140505720d19126b61a8b81c23f5963aef456d06ac7960285e2bca7591843`,
  canonical digest
  `25a89db050917587703903af32e972320339b2447ea21c22da58b2bf7e167430`;
- whole-root inventory: `whole_root_inventory.json`, file SHA-256
  `73858f1bb3a0e66942f7b044d33e935d1eec827b12de62163fecd00dc03e30e5`,
  canonical digest
  `8bf58c71154f88f38d5f213a96d031f061511ecd5b64560da63b5d553fec2671`,
  covering 2,077 non-self-referential files; and
- stop event: `reports/stage_e_e3_20260906/execution_stop_event.json`, file
  SHA-256
  `3d003e0f4fe213c21ba453c2b9db33328320b417b4ecb88b4fc41e374f02a467`,
  canonical digest
  `432f547ddf8e982e9f20a60388606aaf95525f8e32a23d877564b32c38b7d673`.

The source validator must revalidate these bytes and all 516 terminal attempts
before E3-R1 initialization. No write, append, rename, hard-link replacement,
or new attempt is permitted anywhere under the E3 v1 root. E3-R1 uses fresh
`evidence/stage_e/e3_r1_recovery_v2_20260906/formal` and
`reports/stage_e_e3_r1_20260906` roots. Its receipt must be
`completion/e3_r1_completion.json`; it must never replace E3 v1
`completion/e3_completion.json` or `status.json`.

## Eligible complement and immutable union

Eligibility is determined per frozen E3 v1 plan unit by source validation, not
by a reported count or directory existence alone. A structurally,
provenance-, hash-, and contract-valid terminal `VALID`, `FAILED`, or `NA`
attempt is ineligible for E3-R1 computation. Only a unit with no legal terminal
result may run. The current audited partition is 516 valid terminal attempts
and a 2,484-unit complement, but any byte-level disagreement blocks launch.

The 516 inherited attempts remain in place and are referenced read-only by
their unit, attempt, result, manifest, map, and inventory identities. They are
not copied, assigned a new producer, or counted as R1 attempts. Each eligible
unit receives at most one R1 attempt in the fresh root. A valid R1 scientific
failure is terminal evidence and may not be rerun to seek a favorable result.
The final union must be pairwise disjoint, cover all 3,000 frozen new-compute
units exactly once, and retain an explicit origin of `E3_V1_INHERITED` or
`E3_R1_NEW`.

## Frozen numerical recovery

For every eligible NNS unit, R1 must first execute the original schedule
`[512, 1024, 2048, 4096, 8192]` in that order and stop at its first conservation
pass. Only if all five attempts fail may it execute `16384`, and only if that
attempt fails may it execute `32768`. The first passing step count is selected.
Failure at `32768` is terminal scientific invalidity and must be persisted.
Every attempted step, delta, attribution sum, absolute and relative error, pass
flag, resource charge, and selected/final state must be retained.

The conservation rule remains exactly
`abs_error <= 1e-4 + 0.01*abs(delta)`, with `atol=1e-4`, `rtol=0.01`, float32
Gauss-Legendre IG, and chunk size 8. R1 must not change the original global
`IG_STEPS` constant or cause any E3 v1, Stage C, or Stage D validator to
reinterpret historical attempts. The extension is an E3-R1-only overlay.

No threshold may be relaxed; no sample may be removed, replaced, or reordered;
no coordinate, method, target, model, preprocessing, baseline, direction,
ranking, seed namespace/material, tie repeat, intervention, fidelity fraction,
AUC, `delta_fid`, bootstrap, stratum, denominator, missingness rule, or other
statistical contract may change. Coordinates remain `(0,9)`, `(4,2)`,
`(6,10)`, and `(11,2)`, with `(0,9)` read-only. The two methods and all
paper-facing interpretation boundaries remain those frozen by E0 and E3 v1.

## Atomic scientific-failure persistence gate

Before any real pilot access, the producer must fix the E3 v1 defect that
buffered a six-unit image group and lost the coordinate-local failure attempt
when `StageCValuePathConvergenceError` escaped. R1 must publish every unit
independently. A scientific failure must atomically publish a terminal attempt
containing its exact unit/coordinate/method identity, complete IG attempt
history, failure class and code, resource accounting, status, hashes, and
manifest. Any diagnostic attribution tensor is failure-only evidence and may
not be accepted as a valid map or used for fidelity.

Publication is temporary-directory creation, complete writes, file `fsync`,
atomic rename, parent-directory `fsync`, and immediate source validation. Tests
must inject scientific failure at the original-schedule and extended-schedule
boundaries and interruption before and after rename. They must prove that no
partial directory is terminal, no terminal directory is overwritten, the full
failed schedule survives, restart skips every legal terminal result, and a
partial image group cannot enter a summary or PASS. These outcome-free tests
and a clean producer binding must pass before pilot launch.

After that validation, a machine-readable pilot-only execution authorization
must bind the E3 v1 custody commit, this amendment commit, the reviewed
producer commit and tree, the frozen plan/inventory, and both fresh R1 roots.
It may authorize only orders 61 and 72. Bulk continuation requires a separate
machine-readable terminal gate that binds the completed pilot and cumulative
budget decision; this amendment alone is not a bulk-launch receipt.

## Formal billed pilot and continuation gate

The pilot is exactly selection orders `61` then `72`; both are part of the
formal E3-R1 evidence and budget, never disposable smoke data. Their bound
sample IDs and image-group digests are:

- `61`: `ILSVRC2012_val/ILSVRC2012_val_00018234.JPEG`, group digest
  `1594d7d2f10ab19d0ab9a4c50c79898a9ce53f0bf576873fa8b9f7766cb23af9`;
- `72`: `ILSVRC2012_val/ILSVRC2012_val_00026141.JPEG`, group digest
  `0f2cb3e3d83719e2e58d088e50eb5ea5d0b5cbfee080b2021634286e96d543e9`.

All twelve currently eligible units in those two six-unit groups are in the
pilot. Both groups are attempted even if one produces a terminal scientific
failure, so both named failure cases test the recovery path; no non-pilot unit
may start meanwhile. Pilot PASS requires all six NNS units to converge under
the ordered schedule above, all twelve units to have valid terminal results,
atomic-persistence validation to remain PASS, and no contract drift. Pilot
failure is terminal E3-R1 failure and authorizes neither bulk continuation nor
E4. Pilot attempts and costs remain retained in either outcome.

E3 limits remain 8.0 H20 GPU-hours, 16.0 CPU-hours, and 12,884,901,888 retained
bytes. The immutable E3 v1 baseline charged before R1 is 1.0879465173394418
GPU-hours, 4,469.05037553 CPU-seconds, and at least 33,433,845 retained bytes;
it includes the failed and coordinator-interrupted work. No E2 waiver applies.
Every R1 attempt, failure, interruption, pilot operation, retry-equivalent work,
and retained artifact is added to that baseline exactly once.

After pilot completion, a machine-readable gate must show both cumulative
actual cost and a conservative full-campaign projection below every unchanged
limit. Let `C` be the source-recomputed E3 v1 cost plus every E3-R1 cost through
the pilot. Let `g61` and `g72` be the separately recorded, fully charged
per-image-group pilot costs. GPU and CPU projections are respectively
`C + 1.25 * 412 * max(g61,g72)`. The retained-byte projection is the exact
current bytes across the immutable v1 root and all R1 roots/reports, plus
`ceil(1.25 * 412 * max(pilot_group_terminal_attempt_bytes))`, plus a fixed
64 MiB finalization reserve. No overlap, shared setup, interruption, or failed
work may be subtracted from `C`; each pilot group must receive a conservative
share of any shared process cost. The formula is frozen here before pilot
outcome access and may not be tuned afterward.

The remaining 2,472 units may run only if the complete pilot and cumulative
actual and projected total budget views all PASS. A runtime breach stops new
work, retains evidence, and produces no downstream authorization.

| Gate result | Authorized action |
| --- | --- |
| Persistence implementation or tests fail | No pilot |
| Pilot has any nonconverged, failed, NA, missing, or invalid unit | Stop E3-R1 |
| Cumulative actual or projected total budget fails | Stop E3-R1 |
| Pilot and both budget views pass | Run only the validated non-pilot complement |
| Full union, validity, integrity, statistics, and actual budget pass | Issue E3-R1 PASS; authorize E4 only |

## Completion and E4 boundary

E3-R1 PASS requires the exact 516-plus-2,484 union, all 3,000 new-compute units
valid and terminal, the unchanged 1,000 read-only primary method rows, 4,000
total method rows, 2,000 paired rows, twelve frozen coordinate-stratum summary
cells, full source recomputation, no favorable-subset publication, and final
cumulative actual cost within all three E3 limits. Any `FAILED`, `NA`, missing,
duplicate, overwritten, unbound, or contract-drifted unit makes PASS impossible.

The E3-R1 false-PASS audit must independently reject at least: modification or
relabeling of E3 v1; recomputation of any of its 516 terminal attempts; running
a unit that already has a legal terminal result; fallback before exhausting the
original schedule; threshold or contract relaxation; omission of v1 or pilot
cost; exclusion of either pilot; continuation before its gates; favorable-valid
subset summaries; and E4 start before E3-R1 terminal PASS.

Only the new E3-R1 terminal PASS receipt, while explicitly binding and
preserving the E3 v1 terminal FAIL receipt, may authorize E4. E3-R1
initialization, pilot PASS, budget PASS, partial workload progress, or E3 v1
itself authorizes nothing downstream. E4 must not start before that complete
receipt exists and passes independent validation.
