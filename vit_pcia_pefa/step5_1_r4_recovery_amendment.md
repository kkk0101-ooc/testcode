# Step 5.1 r4 A1 queue-schema recovery amendment

`step5_1_r3_20260921` froze successfully, inherited the valid A0 gate, and completed C1.
Its A1 launcher then stopped before creating the SQLite schema, selecting GPUs, decoding
images, loading models, or producing any A1 outcome. The cause was that the reusable base
queue requires rows shaped as `{task_id, payload}`, whereas the frozen Step-5.1 manifest
stores an equally closed flat row. No A1 task was claimed.

r4 changes only the queue adapter: each flat frozen manifest row is deterministically
normalized to `{task_id, payload=<all other fields>}` before calling the already tested
base queue. The queue then remains closed and digest-bound. A new unit test freezes this
normalization. r4 keeps the complete r3 protocol, task manifest content, A0 and C1 results,
science methods, thresholds, precision, resource limits and stage boundaries unchanged.
It inherits A0 and C1 by byte digest. Evaluation remains unread and Step 6 unauthorized.
