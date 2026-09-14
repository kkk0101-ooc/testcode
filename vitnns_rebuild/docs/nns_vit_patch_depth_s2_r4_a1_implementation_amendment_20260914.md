# S2-R4-A1 implementation amendment

This amendment is prospective to the S2-R4 terminal and does not alter any effect threshold, endpoint, candidate grid, target power, precision criterion, missingness rate, family, or scientific claim.

Two implementation defects were found after the first power report was materialized. First, the optional pre-power protocol schema incorrectly combined nested `additionalProperties: false` with no nested property declarations. It is retained unchanged because it is bound by the pre-power seal; the closed signed-artifact envelope in `s2_r4_artifact_envelope.schema.json` supersedes it for artifact validation, while the independent validator checks every exact protocol key and value against the pre-power seal. Second, the D2 small-pilot rule is clarified and corrected to use the maximum of (a) the largest empirical endpoint standard deviation and (b) `sqrt(2.5 * pilot_D2_variance)`. This can only increase the D2 design variance.

The original power and sample-size reports remain immutable and are marked superseded only by reference from new `*_a1.json` reports. The A1 reports must recompute the frozen grid with the corrected conservative D2 scale. A1 cannot authorize S3 if any candidate through 872 classes fails a frozen scientific gate.
