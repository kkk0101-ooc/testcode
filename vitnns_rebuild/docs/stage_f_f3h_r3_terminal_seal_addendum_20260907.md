# Stage F F3H-R3 atomic terminal sealing addendum（2026-09-07）

F3H-R2 completed reconciliation, rendering, audits, tests, and 20/20 manual visual review, but its finalizer signed an already signed validation result a second time. The resulting validation file failed its own canonical-digest check and the pending receipt was not replaced by a terminal completion. R2 therefore remains an immutable blocked attempt even though its scientific/display payload passed validation.

R3 is a CPU-only atomic sealing wrapper. It must:

1. require the R2 blocked archive ref and failure receipt;
2. rerun the existing R2 output validator without writing into R2;
3. recompute and verify the R2 payload manifest, whole-root inventory, 60 exports, 20 review copies, and 20-entry manual QC;
4. require the R2 pending receipt to remain present and R2 terminal completion to remain absent;
5. write only small R3 source-binding, validation, false-pass, test, inventory, status, and pending-to-terminal completion receipts.

R3 must not copy or rerender R2 figures, load a model, decode a JPEG, import or use Torch/timm, access CUDA/GPU, execute attribution/fidelity, define head weights, start empirical F3C, or start F4. A terminal PASS may authorize only `F3C0_PROTOCOL_FREEZE_ONLY`.
