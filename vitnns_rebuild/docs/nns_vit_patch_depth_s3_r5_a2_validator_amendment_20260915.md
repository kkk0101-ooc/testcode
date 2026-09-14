# S3-R5-A2 prospective numeric-validator logical-key amendment

The A1 validator correctly identified the unique corrected scalar but retained its physical unit-NNNNN.npy basename when adding the logical control_mse.npy key, so exact inventory comparison saw one extra alias. No artifact or numeric check failed.

A2 removes the unique physical basename from the temporary in-memory lookup before adding the logical control_mse.npy key. It changes no evidence, array, weight, access record, or scientific contract.
