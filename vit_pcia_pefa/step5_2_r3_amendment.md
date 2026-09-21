# Step 5.2 r3 infrastructure recovery amendment

Status: prospective recovery after the immutable r2 infrastructure failure.

The DeiT-S/16-only scientific protocol, calibration cohort, 1024-to-2048
quadrature rule, thresholds, intervention and claim boundaries remain exactly
as frozen in r2. In r2, all 16 prediction attempts failed after clean numeric
calculation and before artifact creation because the queue stores `task_id` in
the claim envelope while the worker incorrectly read it from the payload. Zero
prediction result was accepted, predictions were not frozen and interventions
did not start. Therefore r2 has no scientific decision.

r3 changes only the task-ID plumbing in both phase workers and adds a regression
assertion for the queue envelope contract. The passed r2 CPU fixture and
DeiT-S/16 P0 result are rebound without recomputation after verifying their
digests, P0-relevant source hashes and equality of all P0 scientific and
numerical protocol fields. The r2 failed-attempt cost remains in the cumulative
Step-5.2 resource ledger.
