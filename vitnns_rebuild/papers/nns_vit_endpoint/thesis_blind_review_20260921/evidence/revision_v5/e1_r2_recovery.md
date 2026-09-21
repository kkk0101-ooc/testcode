# E1-R2 recovery record

The first E1-R2 execution root is reports/nns_vit_revision_v5_e1_r2_20260921. Four workers completed inference but failed before receipt registration because the environment-supplied run root was relative while the receipt code required a path under the absolute repository root. The queue records four failed tasks. That root was not edited or reused.

The implementation resolves environment-supplied manifest and run roots to absolute paths before task execution. The fresh root reports/nns_vit_revision_v5_e1_r2_r1_20260921 then executed all 1,000 tasks with no failures. The final E1-R2 analysis and manuscript use only the fresh root. This recovery changes path handling, not the frozen cohort, WordNet matching, targets, or estimands.
