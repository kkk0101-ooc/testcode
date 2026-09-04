# Stage D recovery-v2 D7-R3 false-PASS correction — 2026-09-04

The prior D7 terminal record at
`reports/stage_d_d7_recovery_v2_20260904/status.json` is classified
`UNSUPPORTED_PASS`. Its 422-item synthetic producer/artifact checks were real,
but the receipt's `finalize`, `aggregate`, and `paper_export` PASS fields were
self-authored diagnostic summaries rather than calls to the D5/D6 production
implementations. This is an unsupported overclaim, not evidence forgery or
source tampering. The record and both prior authorities remain immutable
history and must not launch D8.

D7-R3 repairs the defect with a fresh root and records. It runs the actual
`initialize_stage_d_output_root`, `run_stage_d_shard`,
`finalize_stage_d_root`, `validate_stage_d_root`,
`validate_stage_d_analysis_input`, `aggregate_stage_d_analysis`, analysis
package builder/validator, and paper package builder/validator over a complete
500-group/1,000-unit deterministic synthetic fixture. The only acceleration is
batching repeated checkpoint persistence; all attempt artifacts, the final
checkpoint, D5 sealed package and D6 packages are independently source
recomputed. Any synthetic decision is non-formal and excluded from D8-D10.

The first full-chain attempt is retained locally as a failure because D5
correctly rejected an incomplete synthetic L0 envelope. The second fresh attempt
uses the pre-existing closed synthetic `endpoint_parity` envelope and passed the
real API chain. No Stage D reserve path or JPEG was checked, opened, statted,
hashed, decoded, preprocessed or inferred. No model or GPU was used by the
full-chain smoke.

The scientific contract and C10 facts are unchanged. D7-R3 may issue only a
fresh execution authority after its producer is frozen and the full-chain
receipt, recovery model-only identity check, budget, plan, shard mapping and
fresh D8 root all independently validate. Analysis authorization remains
`NOT_EVALUATED`; D8.0 and all D8 shards remain unstarted until D7-R3 completes.
