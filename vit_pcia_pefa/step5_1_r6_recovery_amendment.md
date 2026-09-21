# Step 5.1 r6 reporting-only recovery amendment

`step5_1_r5_20260921` completed all 16 A1 tasks on their first accepted attempts. Every
task passed clean parity, complete-cut closure and zero/tangent controls. A post-execution
finalizer audit, performed before finalization, found two reporting defects: its aggregate
resource formula omitted r4's conservatively sealed 0.25000801771111114 H20-hours, and its
closed-gate terminal label would have called the unrun B1 natural experiment unsupported
rather than explicitly unrun.

r6 inherits every r5 A1 result, the complete queue, launch receipts, A0 and C1 evidence by
byte digest. It changes no input, model, port, score, baseline, threshold, precision or
scientific result. The finalizer now includes r4 time and emits
`NOT_RUN_A1_GATE_CLOSED` for `NATURAL_NONLINEARITY_LOCALIZATION`. Because the observed A1
result is known, r6 is a transparent reporting-only recovery, not a new prospective
experiment. Evaluation remains unread and Step 6 remains unauthorized.
