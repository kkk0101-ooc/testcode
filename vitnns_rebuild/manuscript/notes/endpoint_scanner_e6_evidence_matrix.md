# Endpoint Scanner E6 claim-evidence matrix

| Claim | Evidence | Decision | Boundary |
|---|---|---|---|
| Base Endpoint identity | EP1-R11 implementation/regression | ESTABLISHED_IMPLEMENTATION_IDENTITY | Same scalar target and normalized input coordinates |
| Structured cut correctness | E3-R2: 76,800 positive closure checks; 28,800 negative controls | SUPPORTED | Correctness, not utility |
| Local directional correctness | E4-R3: 6,144 condition-scaled analytic JVP checks | SUPPORTED | Local differential result |
| C4a finite-replacement association | E6 12 families and 12 Holm tests | NOT_SUPPORTED | All registered families required |
| C4b spatial information gain | E6 4 families and 16 mandatory-baseline Holm tests | NOT_SUPPORTED | All four baselines in every family required |
| Endpoint-Path comparison | EP2-R1 | NOT RELEASED | ViT-B/16 calibration failed under the frozen candidate range |
