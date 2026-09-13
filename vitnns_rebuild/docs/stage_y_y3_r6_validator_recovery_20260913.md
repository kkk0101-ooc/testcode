# Stage Y3-R6 validator-only reconciliation recovery

Status: prospective for Y3-R6 reconciliation. This amendment is frozen after
immutable Y3-R5 `BLOCKED_PROVENANCE` publication and before the R6 validator
accesses scientific payloads.

R5 produced five independently VALID units. Together with the 155 immutable
R4 VALID sources, all 160 frozen non-clean coordinates are scientifically
complete. R5 failed only because its frozen validator compared a path-sorted
manifest against a unit-index-sorted expected list.

R6 performs no empirical computation: no JPEG decode, model load, Torch/CUDA
initialization, GPU use, or new scientific attempt. It source-recomputes R4's
155 VALID units and R5's five VALID units, compares the 160-source manifest by
unique `unit_index`, recomputes the combined resource ledger, and seals the
same frozen Y3 PASS criteria. Arrays remain in their immutable source roots and
are not copied.

All Y0 through Y3-R5 evidence, commits, reports, and refs remain unchanged. Y4,
bootstrap, Holm correction, claim decisions, interpretation, and paper work
remain forbidden in this recovery.
