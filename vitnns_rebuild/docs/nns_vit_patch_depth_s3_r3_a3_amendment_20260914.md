# S3-R3-A3 prospective test-access publication check amendment

Before weight sealing or any D2 test access, review found that requiring a test-access-gate JSON to contain the commit hash that first contains that same file is self-referential and cannot be satisfied. A3 replaces that impossible check with three concrete publication checks: the recorded weight-seal input commit must be an ancestor of local HEAD; local HEAD must equal the live remote branch tip; and the test-access-gate path must exist in that exact HEAD Git tree. The gate's content digest, PASS status, weight seal, and archive rehash gates remain independently verified.

This changes only the control-plane publication proof. It does not change weights, alpha, normalization, cohort, preprocessing, test data, predictions, missingness, or any scientific contract. D2 test access remains zero at amendment freeze.
