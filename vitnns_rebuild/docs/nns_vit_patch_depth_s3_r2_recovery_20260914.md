# S3-R2 prospective recovery amendment

S3-R1 corrected the outer Git-tree path but then failed in metadata-only startup because S2-R8 evidence is stored in Git as LFS pointers while the worktree contains hydrated bytes. Comparing pointer bytes with hydrated bytes is invalid. S3 and S3-R1 roots remain empty and immutable; all formal image, mask, feature, prediction, and outcome access counts remain zero.

S3-R2 uses fresh evidence, payload, and schedule paths. For LFS-backed files it verifies the Git-tree pointer oid SHA-256 and size against the hydrated file; ordinary blobs remain byte-for-byte verified. No scientific or execution contract changes.
