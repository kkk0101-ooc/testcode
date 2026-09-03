# Stage D recovery-v2 D0-D6 completion and future D7 handoff

D0-D6 pre-reserve rebinding is complete in the corrected R2 record root
`reports/stage_d_recovery_v2_20260904_r2`. D0-D2 are current formal preparation
PASS; D3 is control-plane readiness PASS; D4-D6 are implementation/readiness
PASS based only on metadata and synthetic fixtures. D5 is not validation of real
Stage D 500-image evidence, and D6 is not completed analysis or a scientific PASS.

The chain is canonical salvage C8 `82825131...` -> recovery C9
`0d88fbb...` / `c66e5224...` -> recovery C10 `3cf2945...` / `0d0eab0...` /
`f1dc44d...` -> R2 producer `da913e2b...` -> D0 `967102eb...` -> D1
`5f96cb7e...` -> D2 `c65e613e...` -> D3 `b7bed30a...` -> D4 `03ba0d3c...`
-> D5 `32211220...` -> D6 `3898c97a...`. The model is `489bfa...` with
canonical tensor digest `150234d8...`, never the historical `2e718...` artifact.

Historical D0-D2 PASS records remain valid historical preparation evidence but
are stale for recovery-v2. Historical D3-D6 PASS records remain synthetic-only.
The first recovery record set is retained and classified invalid because its
`$schema` URI resolved one directory above `config/`; no file was overwritten.
The corrected R2 records resolve the schema from their own directories and were
freshly generated and validated. The total audit found no forgery, source tamper,
cross-root splice, duplicate selected attempt, or current self-signed PASS without
source recomputation.

The fidelity-only contract is unchanged: independent `n=500`, primary `(0,9)`,
NNS versus matched patch contribution, endpoint `fine_4x4_cell_delta_fid`, image
statistical units, 10,000 paired-median bootstrap, `run_stability=false`, no
secondary inference, and no pooling with C10 `n=15`. C10 remains acceptance
`BLOCKED`, decision `NO_GO_NOVELTY`, effect component `PASS_EFFECT`, translation
and crop `FAIL_STABILITY`, overall spatial `FAIL_SPATIAL`, and Holm
`HOLM_BLOCKED_UNDERSPECIFIED`.

D7, D8, D9 and D10 have not started. No Stage D execution, analysis or reserve
access authorization exists. No registry-referenced reserve image path was
checked for existence, statted, opened, hashed, decoded, preprocessed or used for
inference. D8 was not initialized.

A future D7 remains an independent gate and is not authorized here. It must
revalidate the whole chain, run a non-reserve smoke, derive a fresh budget,
verify Git and the recovery model, freeze the exact plan and shard count, verify
modulo sharding and a fresh formal root, and only then consider a separate
authority-only execution authorization.
