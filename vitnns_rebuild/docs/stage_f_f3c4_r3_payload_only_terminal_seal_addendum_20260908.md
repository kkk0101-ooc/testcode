# Stage F F3C4-R3 payload-only terminal-seal addendum (2026-09-08)

F3C4-R2 stopped before validation because its proposed validator would have rehashed upstream F3C3 arrays and formal JPEGs, exceeding the R2 payload-only recovery boundary. R2 performed no new F3C3 array access, JPEG access, statistics, rendering, model work, or terminal publication; its startup records, pending receipt, and failure event remain immutable.

F3C4-R3 is the final CPU-only payload validation and terminal sealing wrapper. It may read only the already generated R1 evidence/report roots. It must:

1. use fresh roots `evidence/stage_f/f3c4_analysis_r3_terminal_seal_20260908/formal/` and `reports/stage_f_f3c4_analysis_r3_terminal_seal_20260908/`;
2. bind the complete R1 payload inventory and prove it remains byte-identical throughout R3;
3. validate CSV/JSON parity by the explicit frozen `columns` sequence, independent of canonical JSON object-key order;
4. revalidate R1 per-image tie means, frozen contrast direction, descriptive summaries, bootstrap, sign-flip, and Holm outputs solely from R1 paper-data;
5. validate render module counts, exhaustive formal-position coverage, recorded shared scales, R1 output hashes, and the 23/23 R1 manual-QC receipt without opening any F3C3 source array or JPEG;
6. validate the R1 resource/access boundary, 47-item false-pass audit, and test receipt;
7. publish R3 validation, resource/access, payload/inventory/status, and atomic completion receipts.

R3 must not open or hash F3C3 terminal payload files, formal JPEGs, pilot data, or order 44; must not recompute an empirical endpoint, render, or change any R1 scientific value; must not import Torch/timm, load model/checkpoint, access CUDA, or start F4.
