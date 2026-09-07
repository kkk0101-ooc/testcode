# Stage F F3C4-R1 pre-outcome access recovery addendum (2026-09-08)

F3C4 v1 passed its startup gate, but a validator-development test then used one real F3C3 formal terminal as a mutation fixture before the F3C4 pre-outcome plan seal existed. The test read that terminal's `result.json` and numerical `.npy` payload. It decoded no JPEG, rendered no map, ran no inference, and produced no statistical result. This violates the required ordering, so the v1 root is permanently `BLOCKED`; its startup gate, pending receipt, and failure event must remain immutable.

F3C4-R1 is authorized only to recover that ordering defect:

1. use fresh roots `evidence/stage_f/f3c4_analysis_r1_20260908/formal/` and `reports/stage_f_f3c4_analysis_r1_20260908/`;
2. replace the real-terminal mutation fixture with a fully synthetic terminal fixture;
3. re-run the complete startup gate and atomically seal the exact cohort/input/statistical/render/access/resource/output/code plan before any further F3C3 `result.json`, numerical array, or formal JPEG access;
4. execute the unchanged frozen F3C1 statistical method and unchanged F3C4 rendering contract after that seal;
5. preserve all F3C0-F3C3 inputs and every earlier sealed root/ref unchanged.

This recovery does not authorize model/checkpoint/Torch/timm/CUDA access, empirical recomputation, pilot/order-44 access, sample or method changes, additional endpoints, F4, or reuse/overwrite of the v1 root.
