# Step 5.1 r5 A1 claim-envelope recovery amendment

`step5_1_r4_20260921` created the closed A1 queue and selected four eligible H20s. The
first four DeiT tasks each failed twice after computing the first condition but before
writing an accepted result. The worker used `task['task_id']` for the deterministic random
namespace, while the queue deliberately keeps `task_id` in the immutable claim envelope
and only scientific payload fields inside `task`. Four later tasks were in-flight when the
repeated common exception was detected; all workers were stopped, and r4 has zero accepted
scientific results. It is terminal `TERMINAL_INFRASTRUCTURE_FAIL`.

r5 changes exactly that read to `claim['task_id']`. It preserves every scientific field,
the exact A1 manifest, seed namespace, model, sample, target, head assignments, controls,
path, reference, 512-node quadrature, scores, thresholds, precision and resource caps. A0
and C1 are inherited by byte digest. r4 time through the conservative failure seal is
charged to the cumulative resource ledger. Evaluation remains unread and Step 6 remains
unauthorized.
