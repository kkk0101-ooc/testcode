# ENDPOINT-E4 R3 completion audit

Date: 2026-09-18. Objective: execute the E4 recovery required by
`NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md`.

| Requirement | Evidence | Result |
|---|---|---|
| Preserve R1 failure and use a fresh recovery root | R1 terminal SHA-256 `7bb4e796...`; R2/R3 configs bind immutable source roots; R3 root is `reports/nns_vit_endpoint_e4_intervention_r3_20260918` | PASS |
| Freeze the revised numerical contract before new outcomes | R3 contract and runner were committed and pushed at `808d827acdc2f72e6a9dd94bcf4161884eebf6f3`; startup snapshot binds the same local, tracking and remote commit | PASS |
| Diagnose R2 rather than silently relax its gate | Published diagnostic SHA-256 `22a8fdc...` identifies one failed component and records the condition scale; R2 terminal remains `ENDPOINT_E4_R2_FAIL_ANALYTIC_JVP_CONTRACT` | PASS |
| Keep the frozen 400-image evaluation cohort and both models | Combined result index has 800 unique image-model records; independent validator binds every row to the E2 materialization | PASS |
| Re-run the complete local subset under the R3 gate | 32/32 new tasks completed, covering both models and evaluation orders 0--15; 6,144/6,144 condition-scaled analytic JVP checks passed | PASS |
| Do not select only the prior failed sample | R3 task manifest contains all 32 local tasks, including `vit_b16__009` | PASS |
| Reuse only validated non-local R2 results | 768 records with evaluation orders 16--399 were revalidated for payload/result digests, identity, finite values, metrics, zeroing and determinism | PASS |
| Exclude R1 and R2 local partial results | Combined index origin counts are exactly 768 `inherited_from_r2_source_revalidated` and 32 `newly_run_in_r3`; no R1 or R2 local origin is present | PASS |
| Exact complete union | 800 records, 0 missing, 0 duplicate, 0 invalid; independent validator passed | PASS |
| Frozen finite-replacement counts | 249,600 main, 9,216 secondary and 16,896 zeroing evaluations | PASS |
| Numerical and no-op gates | Maximum JVP error `2.648076753197781e-05`; maximum error/absolute-dot-mass `6.948755071487438e-06`; maximum no-op difference `2.09808349609375e-05` | PASS |
| Deterministic H20 execution and resource caps | Startup snapshot records H20 eligibility and existing PIDs; report records CUBLAS determinism; combined 0.6299475847480777 GPU-hours, 287.32127761713497 wall-seconds, and 17,168,290 retained bytes | PASS |
| No out-of-scope analysis | Report defers bootstrap, Holm and C4a decisions to E6; E5 spatial intervention and Path IG/input fidelity/stability were not run | PASS |
| Independent verification covers the objective | `scripts/validate_endpoint_e4_recovery_r3.py` reopens manifest entries and all 800 result files, recomputes digests, per-image metrics, counts, provenance, resource and next-authorization gates | PASS |
| Code regression | 65 focused Endpoint tests passed; `py_compile`, `git diff --check`, unified-plan and E0--E4 predecessor validators passed | PASS |
| Next-stage boundary | Terminal records `ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST`; E5 was not started | PASS |

Completion decision: `ACHIEVED`.

## Defensive-writing audit

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| R3 result, scope paragraph | `本 PASS 只表示...不表示...` | D12 | Minor | Yes | The sentence marks the registered stage boundary once | Clear scope rather than prebuttal | Preserve | No change |
| R3 contract, recovery boundary | `不得进入 R3 联合结果` | CLEAN | None | Yes | It defines provenance eligibility | Reproducible exclusion rule | Preserve | No change |
| Unified plan, next step | `不得由本次 E4 恢复授权自动启动` | CLEAN | None | Yes | It states the explicit stage gate | Clear authorization boundary | Preserve | No change |

### Paper-level defensive-writing score

`1/10` (clean). The reviewed R3 documents state measurements first and use one concise scope
boundary for the deferred C4a decision.

### Highest-priority removals

None.

### Necessary caveats to preserve

- R3 PASS is an execution/correctness/coverage result, not the E6 C4a decision.
- R1 and R2 failure receipts remain immutable.
- E5 requires a separate explicit request.

### Repeated phrase audit

No repeated reviewer-facing prebuttal, promotional compensation, or redundant non-claim
disclaimer was found in the R3 contract or result document.
