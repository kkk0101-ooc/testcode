# Stage X X2-R2-R1 Control Initialization Recovery Amendment

This prospective amendment authorizes only `X2_R2_R1_CONTROL_INITIALIZATION_RECOVERY_ONLY`.

X2-R2-R1 may repair the X2-R2 initializer, worker startup preflight, and publication-control classification that incorrectly exposed control-layer startup failures as unit-level `NA` terminals. It does not alter the X2, X2-R1, or X2-R2 roots, terminals, receipts, schemas, run specs, implementation seals, route roster, checkpoint bindings, NNS-v2 definition, cuts, schedule, tolerances, comparators, gamma grid, endpoints, transformations, repeats, margins, futility rule, resource caps, or budget gates.

The exact 95% clean coverage policy remains frozen: 200 expected routes, at least 190 valid routes overall, and at least 38 valid routes per model with 40 expected routes per model. Schema, source, identity, atomic publication, and data-isolation failures have zero tolerance.

X2-R1 and X2-R2 terminal records are immutable. The 197 X2-R1 clean `VALID` routes may be reused only after source validation of the terminal envelope, files, arrays, model/sample/checkpoint bindings, selected steps, 135-cut status, and IG identity control. The X2-R1 failed route must not be relabelled as valid. The original failed route and the two never-started routes must be newly executed under X2-R2-R1.

The two X2-R2 `NA` unit terminals are retained as historical control-only diagnostics. Their `actual_execution_attempted=true` field is inaccurate for scientific-route classification. X2-R2-R1 binds them as `CONTROL_PREFLIGHT_ONLY_NO_SCIENTIFIC_ROUTE_EXECUTION`: a model may have been loaded, but the journal preflight failed before JPEG decode and before prediction, NNS-v2, IG, cuts, comparators, or ROAD started.

Before any X2-R2-R1 model load or JPEG decode, the fresh root must publish startup, amendment, X2-R1 failure, X2-R2 blocked, sample manifest, unit manifest, reuse/new route CSVs, model materialization binding, mutation suite, run spec, implementation seal, and a CPU-only, model-free, JPEG-free worker preflight over an explicit required-files list. Same-root retry is forbidden. X3, X4, X5, X6, F3-Final, and Stage G remain unauthorized.
