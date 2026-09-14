# S3-R3-A4 prospective evidence storage publication amendment

The first two pushes of the unpublicized weight-seal commit were interrupted because a local Git info attribute sent every small S3 JSON evidence file through LFS, producing 21,809 independent LFS uploads. The remote branch remained at the prior A3 implementation commit and D2 test access remained zero.

A4 changes only Git storage for S3-R3 JSON evidence: S3-R3 JSON files are ordinary Git blobs while large NumPy arrays remain ignored local payloads and NPZ/NPY rules remain LFS if explicitly committed. Existing JSON file bytes, canonical digests, schedules, terminal identities, weights, access gates, and scientific contracts do not change. The unpublicized weight-seal commit is amended before publication so the remote receives a single coherent Git tree.
