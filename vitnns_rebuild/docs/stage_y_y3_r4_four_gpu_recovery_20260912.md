# Stage Y3-R4 fixed four-GPU recovery amendment

Status: prospective for all Y3-R4 computation. This amendment is recorded
after immutable Y3-R3 `BLOCKED_RESOURCE_SCHEDULE` evidence and before any
Y3-R4 JPEG decode, model load, CUDA initialization, or new outcome access.

The owner explicitly requires physical H20 devices 0, 1, 2, and 3. Y3-R4 must
lock exactly that ascending roster; any unavailable device blocks startup. The
coordinate owner is `unit_index mod 4`; no hot-add, migration, or dynamic
rebalancing is allowed.

R3 units 0 and 2 are complete VALID evidence and are source-recomputed and
rebound without copying arrays or rerunning them. The three interrupted R3
units 1, 3, and 5 contain only PENDING receipts and no retained arrays, attempt
journals, resource receipts, or terminals. They and the other uncompleted
coordinates are computed once in R4, for 158 new R4 executions. The unified Y3
view remains exactly 160 non-clean units.

R3's missing interrupted-unit resource telemetry is never zero-imputed. For
hard-cap admission, R4 reserves the full three-worker R3 elapsed window as an
upper bound for GPU time and four times that bound for CPU time, in addition to
all measured valid-unit and R4 resources. This conservative prospective
accounting does not raise any Y3 cap. Reports retain both measured values and
the explicit upper-bound reserve.

All scientific, transform, target, retention, failure, and publication
contracts remain unchanged. Y4 and all statistical or claim work remain
forbidden.
