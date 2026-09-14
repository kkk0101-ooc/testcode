# S2-R8 prospective low-cost hierarchical gatekeeping recovery

S2-R8 is owner-authorized after S2-R7 design power and ICC were known. It is therefore a prospective recovery before formal outcome access, not the original blind preregistration. The only changes are target power 0.80, serial gatekeeping, and the outcome-blind D2 cap ladder `m in {2,3,4}`. Effect thresholds, design alternatives, test directions, nuisance parameters, class set, missingness, and estimands do not change.

The first family is P1, P2, and D1 under Holm FWER 0.05. D2 is tested two-sided at alpha 0.05 only if all three first-family nulls are rejected. If any first-family null is true, every familywise error event requires a false first-family rejection and is contained in the Holm familywise error event, whose probability is at most 0.05. If all first-family nulls are false, the only possible true null is D2 and its gated level-0.05 test has type-I error at most 0.05. This proves strong FWER at most 0.05 for every nonempty true-null configuration.

The three exact candidate manifests are generated and sealed before real power computation. Each contains the original two confirmatory images per class and then takes the first additional R7 records under the already frozen selection-key order, capped at m. They are strictly nested. Extra D2 images have `P1=false`, `P2=false`, `D1=false`, `D2=true`, `S4_NNS=false`, and `S4_occlusion=false`.

Power uses 1,000,000 fixed-seed Monte Carlo replicates. PASS requires every first-family marginal power, gate-open power, conditional D2 power, unconditional joint power and its one-sided 95% Wilson lower bound, and the correlation-independent Frechet lower bound to satisfy the frozen rules. The joint point estimate and lower bound must both be at least 0.80. Calibration covers all 15 nonempty true-null configurations and supplements, but does not replace, the analytic strong-FWER proof.

The maximum S3 source-image forward count is 10,458, an increase of 1,738 over 8,720, below 20%. S4 remains exactly 1,744 images. S2-R8 does not execute S3, S4, or S5 and makes no scientific claim.
