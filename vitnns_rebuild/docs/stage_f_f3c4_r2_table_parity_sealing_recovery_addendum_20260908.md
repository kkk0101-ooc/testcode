# Stage F F3C4-R2 table-parity sealing recovery addendum (2026-09-08)

F3C4-R1 completed the sealed-plan CPU reconciliation, frozen statistics, paper-data export, exhaustive 50x5 atlas, signed audit, 50 formal JPEG hash-before-decode checks, and 23/23 manual visual QC. Its independent final validator then failed before terminal completion because canonical JSON sorts object keys while the validator incorrectly required each decoded row object's iteration order to equal the separately frozen `columns` list. All table row counts and values were present; this was a sealing-validator ordering defect, not a scientific or data-integrity failure. The R1 pending receipt, payload, and failure event remain immutable.

F3C4-R2 is limited to CPU-only terminal sealing:

1. use fresh roots `evidence/stage_f/f3c4_analysis_r2_terminal_seal_20260908/formal/` and `reports/stage_f_f3c4_analysis_r2_terminal_seal_20260908/`;
2. fix CSV/JSON parity validation to compare the frozen explicit column sequence with CSV headers and compare row values by column name, independent of canonical JSON object-key serialization order;
3. validate and bind the immutable R1 paper-data, renders, render reports, manual QC, resource/access reports, tests, and 47-item false-pass audit by file hash;
4. publish new R2 validation, payload/inventory/status, and pending-to-terminal completion receipts without copying or changing R1 scientific payload;
5. perform no new F3C3 result/array access, JPEG decode, statistics, rendering, or scientific interpretation.

This recovery does not authorize model/checkpoint/Torch/timm/CUDA access, forward/backward/attribution/fidelity execution, pilot/order-44 access, sample/method/statistical/display changes, upstream modification, or F4.
