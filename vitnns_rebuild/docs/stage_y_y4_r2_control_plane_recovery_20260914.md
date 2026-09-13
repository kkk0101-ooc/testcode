# Stage Y4-R2 prospective control-plane recovery

Y4-R1 froze its outcome-blind numerical implementation at commit
`4cbcf763f81455b09f09d7e489161225f62c8237` and created the remote producer
ref. Its implementation-seal command then stopped before creating a formal root
or reading any scientific outcome because it compared the pre-producer ref
snapshot with a post-producer snapshot without excluding its own newly-created
producer ref. No empirical or numeric access occurred.

Y4-R2 inherits the frozen Y4-R1 numerical engine byte-for-byte. Its only change
is prospective control-plane logic that excludes the new Y4-R2 producer ref
itself when comparing the preflight historical-ref snapshot. All prior refs,
roots, metrics, aggregations, seeds, replicate counts, hypotheses, thresholds,
Holm rules, truth tables, access restrictions and reporting requirements remain
unchanged. Y4-R1's producer ref remains immutable.

The fresh formal root is
`evidence/stage_y/y4_r2_mechanism_analysis_20260914/formal/`. The only PASS
authorization remains `STAGE_Y5_RESULTS_TO_PAPER_HANDOFF_ONLY`; this run does
not execute Y5.
