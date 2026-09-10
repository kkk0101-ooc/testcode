# Stage X X2 provenance closeout

Date: 2026-09-10. Status: `NO_GO_BROAD_CLAIM_DEVELOPMENT`.

This record closes X2 without changing any scientific result or retrying any
route. The implementation/configuration producer is commit
`d7d8ca177caac0b81efc3d051093d84bc464cf0a`, tree
`bf7d1aef55df7c04adb693011af9b9bb7993ee8d`, archived as
`refs/archive/stage-x-nns-v2-x2-producer`.

The authoritative terminal root is
`evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal/`. Independent
source-recomputing validation returned the expected terminal status, 200 unit
terminals, 2,431 inventory entries, and zero pending publications. The final
denominator is 200: 199 clean-valid routes and one immutable route-local
numerical failure. The root records 50/115 absolute stability cells passing,
failed Grad-CAM stability noninferiority, a non-triggered ROAD futility test,
and failed X2/formal budget gates. It makes no formal superiority claim and
does not authorize X3, X4, or X5.

The earlier X2, X2-R1, X2-R2, and X2-R2-R1 terminal failures remain immutable
and are bound by file SHA-256 and canonical digest in
`reports/stage_x_x2_provenance_closeout_20260910.json`. No failed root was
overwritten or reclassified.

The five local X2 roots contain 11,407 regular files and 1,506,461,563 bytes.
They remain the complete scientific evidence. The lightweight Git publication
contains the implementation, configurations, schemas, prospective amendments,
tests, historical failure receipts, final control/aggregate/terminal records,
validation report, and full inventory/hash indexes. It intentionally does not
copy every per-route payload to Git; the remote repository is therefore not a
complete evidence mirror.

Verification performed for closeout: 57 focused Stage X tests passed; all X2
Python modules compiled; `git diff --check` passed; the final source validator
validated 2,434 artifacts and 398 formal journals. The next authorization is
only `F3_FINAL_FIGURE_PACKAGE_ONLY` on the original scoped-paper route.
