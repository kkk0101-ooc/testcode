# Stage Y3-R3 worker-import recovery amendment

This amendment is prospective and was written before any Y3 scientific unit,
JPEG decode, model load, or model call. Y3-R2 is immutable at commit
`b8f0e6e285845c5e3b755b4d59cc1bc67adf2976` with terminal
`BLOCKED_INFRASTRUCTURE`: every locked worker exited at module initialization
because `stage_y_y3_worker.py` referenced the frozen Y2-R2 contract module as
`r2` without importing it.

Y3-R3 adds exactly that missing module import. It uses a fresh formal/report
root and fresh producer/PASS/blocked refs. The 20-image subset, exact order,
source-derived 17+3 clean provenance, eight conditions, coordinate rule, target,
transforms, numerical tolerances, quadrature, interventions, resource caps,
retention, and failure semantics do not change. The Y3-R2 root and every earlier
Stage Y root/ref remain immutable.
