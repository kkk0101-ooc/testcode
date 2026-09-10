# Stage Y Y1 implementation validation and cost pilot

Protocol: `STAGE_Y_VIT_BLOCK_LEVEL_NNS_MECHANISM_V1_20260910`
Authorization: `STAGE_Y1_IMPLEMENTATION_VALIDATION_AND_COST_PILOT_ONLY`

Y1 implements and validates the frozen complete-cut identity, source-patch routing, native-head and two-residual decomposition, transform transport, head interventions, hierarchical within-patch deletion, and resource accounting. It may decode only `ILSVRC2012_val_00002287.JPEG`, under `clean` and `horizontal_flip`, on one H20. It is implementation and cost evidence, never cohort evidence or a scientific claim decision.

The pre-outcome machine contract is `config/stage_y/y1_run_spec.json`. Source-patch routing uses grouped forward-mode JVPs from disjoint 16x16 input-patch directions and native reverse cotangents at each frozen cut. It never constructs a pixel-by-hidden-coordinate dense Jacobian. The first residual is identity versus twelve native pre-concat head outputs carried through their matching output-projection columns; the second is post-attention residual versus MLP. Layer decompositions remain alternatives and are never summed.

Formal evidence is written only to `evidence/stage_y/y1_validation_pilot_20260910/formal`; reports are written to `reports/stage_y_y1_20260910`. The run begins with an fsynced `PENDING` receipt and ends through atomic replacement by `PASS`, `BLOCKED`, or `BLOCKED_BUDGET`. A PASS authorizes only `STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY`; every blocked terminal authorizes `NONE`. No Y2 materialization or later-stage work occurs here.
