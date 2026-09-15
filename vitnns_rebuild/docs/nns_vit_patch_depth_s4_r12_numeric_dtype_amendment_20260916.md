# S4-R12 numeric dtype validation amendment

The first post-implementation producer attempt failed before creating the R12 evidence root because the generic array validator applied `numpy.isfinite` to the frozen string-valued `p1_seed_digests` array. No R12 evidence file or scientific result was written. Prospectively, finiteness is required for numeric dtypes; nonnumeric arrays retain exact bytes, SHA-256, shape, dtype, and canonical-array-digest validation. This changes no source, endpoint, threshold, numerical value, or decision rule.
