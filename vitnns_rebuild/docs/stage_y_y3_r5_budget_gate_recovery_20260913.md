# Stage Y3-R5 global budget-gate recovery amendment

Status: prospective for all Y3-R5 computation. The owner authorized continued
recovery after immutable Y3-R4 `BLOCKED_BUDGET` evidence and before any R5
JPEG decode, model load, CUDA initialization, or new scientific attempt.

Y3-R4 retained 155 VALID perturbation terminals and five pre-compute
`BLOCKED_BUDGET` terminals at unit indices 141, 145, 149, 153, and 157. All
five missing coordinates belong to immutable logical shard 1. Y3-R5 therefore
locks physical H20 GPU1 only, preserves logical shard 1 ownership, source-
recomputes and rebinds the 155 VALID terminals without copying arrays, and
performs exactly five new scientific computations. No clean unit is rerun.

Root cause: the R4 worker divided the global stage GPU and CPU caps by four and
applied those quarter-caps independently to each shard. This was conservative
but not the frozen global accounting rule, and it blocked shard 1 even though
the independently reconciled global ledger remained within every hard cap.

R5 retains the original caps unchanged. Before each new unit it combines the
R4 accounted global ledger (including the conservative R3 interrupted-work
reserve) with measured R5 initialization and unit resources. It does not divide
the remaining global allowance by worker count. If the combined ledger reaches
a cap, no further unit starts and the stage blocks honestly.

All scientific, transform, target, deterministic-CUDA, retention, failure, and
publication contracts remain unchanged. Y4, bootstrap, Holm correction, claim
decisions, interpretation, and paper work remain forbidden.
