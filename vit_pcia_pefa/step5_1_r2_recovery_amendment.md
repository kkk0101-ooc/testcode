# Step 5.1 r2 prospective recovery amendment

`step5_1_r1_20260921` is immutable and terminal `FAIL_A0_TASK_EXECUTION`. Both real-model
A0 workers stopped before model deserialization or scientific calculation because the
frozen A0 task rows omitted the already frozen `block_index=5` field consumed by the
worker. The common exception was `KeyError: 'block_index'`; this is an infrastructure
schema failure, not a numerical or scientific outcome.

r2 adds only `block_index: 5` to both A0 manifest rows. It preserves the models, discovery
image, class target, real consumer use-port, selected head, wrapper formula, strengths,
controls, path, reference, 512-node quadrature, precision, tolerances, A0 gate, A1 sample,
all scientific thresholds, and all stage boundaries. The r1 CPU fixture result is not
inherited because the source binding changes; it is recomputed in r2. The conservative r1
launch time on two selected H20s is added to the cumulative resource ledger. Evaluation
remains unread and Step 6 remains unauthorized.
