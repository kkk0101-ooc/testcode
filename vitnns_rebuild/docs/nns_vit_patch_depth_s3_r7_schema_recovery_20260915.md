# S3-R7 prospective closed-schema publication amendment

Post-publication review found that S3-R6 summary envelopes used a run ID not enumerated by the older general S3 schema, and the general schema did not close every summary payload even though the independent publication validator did. S3-R6 remains immutable but is not the final terminal root.

S3-R7 adds a dedicated publication schema with an exact document-type enum and additionalProperties=false for every summary payload. After this schema and the R7 finalizer are committed and pushed, R7 performs a metadata-only rebind of the same S3-R5 terminals and arrays and emits byte-equivalent scientific and resource summaries under the closed schema. No formal data, features, predictions, weights, alpha choices, or outcomes are accessed or changed.
