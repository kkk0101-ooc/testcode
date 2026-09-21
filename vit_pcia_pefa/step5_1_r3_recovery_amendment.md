# Step 5.1 r3 prospective A1 clarification

`step5_1_r2_20260921` passed the CPU-float64 fixture and both registered-model A0 tasks.
Those development-only results are immutable and establish A1 eligibility. Before any A1
task was launched or any A1 outcome read, an internal consistency audit found that the
protocol prose requires a same-strength decoy and evaluates it at each primary strength,
while the structured control list accidentally named only strength `1.0`. The same
pre-execution audit also found that the level-only comparator implementation read the
modified rather than clean arrays, despite the frozen prose specifying clean Endpoint,
PCIA, gradient and activation levels.

r3 resolves this ambiguity in the stricter direction: the predeclared decoy port is run at
both primary strengths `0.5` and `1.0`. Nothing else changes: A1 samples, models, targets,
true and decoy head selection, path, reference, block, cut, 512-node quadrature, methods,
metrics, thresholds, numeric tolerances and resource limits are identical. r3 inherits A0
by validating the canonical r2 A0 decision and its three result digests; it does not rerun
or reinterpret A0. Neither clarification changes any A0 gate quantity. r1 and r2 resource
use remains cumulative. Evaluation remains unread
and Step 6 remains unauthorized.
