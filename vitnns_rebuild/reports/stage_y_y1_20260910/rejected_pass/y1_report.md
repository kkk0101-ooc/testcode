# Stage Y Y1 terminal report

Y1 completed implementation validation and the frozen one-image cost pilot. Status: **PASS**. This is implementation/cost evidence, not a Stage Y scientific claim decision.

- Empirical scope: `ILSVRC2012_val_00002287.JPEG`, prior-access order 44, clean and horizontal flip only, two complete repeats each, one H20.
- Coverage: blocks 0-11, heads 0-11, seven frozen cut families, signed 196x196 routing matrices, CLS vectors, 56x56 maps, 144 single-head ablations and all top/bottom-k/deletion controls per repeat.
- Numeric status: closure, endpoint parity, convergence and byte-identical repeatability PASS; clean selected 512 steps and flip selected 256 steps.
- Resources: {'cpu_hours': 4.258968718303611, 'devices': 1, 'download_bytes': 0, 'h20_gpu_hours': 1.2318711631944446, 'peak_reserved_gpu_gib_per_process': 13.7578125, 'retained_storage_gib': 0.13808077573776245, 'wall_hours': 1.232424861373923}. Y1 hard-cap status PASS.
- Proposed Y2/Y3 budgets use the frozen maximum-repeat formula and 1.5 multiplier: Y2 {'h20_gpu_hours': 61.400000000000006, 'wall_hours': 62.0, 'cpu_hours': 213.0, 'storage_gib': 6.0}; Y3 {'h20_gpu_hours': 98.2, 'wall_hours': 99.0, 'cpu_hours': 340.0, 'storage_gib': 9.0}; both below owner ceilings.
- Tests: 29 passed. False-pass audit: 23/23 rejected.
- Formal outcome access: none. Y2-Y5, F3-Final, and Stage G were not started.

The only next authorization is `STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY`.
