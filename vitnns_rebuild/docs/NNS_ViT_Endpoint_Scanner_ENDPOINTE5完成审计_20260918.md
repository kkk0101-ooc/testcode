# ENDPOINT-E5 completion audit

Date: 2026-09-18. Objective: execute ENDPOINT-E5 from the unified Endpoint Scanner plan.

| Requirement | Concrete evidence | Result |
|---|---|---|
| Explicit E5 authorization and prospective freeze | User objective names ENDPOINT-E5; config, contract, implementation, worker, validator and tests were committed and pushed as c876ea2170a4287bd9bd15baa0a933fa00eb31bd before formal outcome access | PASS |
| Valid E4-R3 predecessor | E4-R3 terminal is ENDPOINT_E4_R3_PASS_EXECUTION_CONDITION_SCALED_JVP_AND_FINITE_REPLACEMENT; startup reran the E4-R3 and unified-plan validators | PASS |
| Published code and reproducible provenance | Startup snapshot binds local HEAD, upstream and remote to c876ea217; implementation bindings cover eight named source files | PASS |
| Frozen 400-image shared cohort and two models | Queue contains 800 unique tasks; each result is bound to E2 materialization ID, WNID, class, bytes and SHA-256 | PASS |
| Block 5 and three registered heads per model | Every result contains ViT-B 0/6/11 or DeiT-S 0/3/5; validator rejects other block/head sets | PASS |
| 4×4 row-major 56×56 regions and budgets 0/1/2/4 | Every result binds 16 independently reconstructed mask hashes, budget counts and actual areas 0/1/16/2/16/4/16 | PASS |
| Two frozen references | Every task binds normalized-zero and blur tensor identities to the corresponding E4 result; model and checkpoint identities also match E4 | PASS |
| Primary component-map RGB absolute-mass ranking | All result ranks are recomputed; 32 retained raw-map archives independently reproduce their 16 regional masses and ranks | PASS |
| Mandatory total G×I and raw-gradient baselines | Both shared rankers are present for every task/reference and their ranks are independently recomputed | PASS |
| Mandatory regional activation-space Taylor | 25,600 single-region forward results; every head-specific rank is recomputed from its saved 16 scores | PASS |
| Twenty frozen random permutations | 96,000 random curves; validator derives every PCG64 seed/permutation from task/model/reference/head/repeat | PASS |
| No finite-effect region selection | Deterministic rankings bind saved pre-effect score vectors; random rankings bind hash-derived permutations; every selected union is a frozen rank prefix | PASS |
| Isolated head effect and descriptive full-forward effect | 345,600 isolated-head evaluations and 345,600 corresponding full-forward curve points; validator recomputes both effects from raw logits | PASS |
| Per-image primary score and random aggregation | Validator recomputes every trapezoidal AUC, three-head mean, random signed-curve mean and four Endpoint-minus-baseline differences | PASS |
| Exact queue completion | 800 completed first attempts, 0 missing, 0 duplicate, 0 invalid, 0 retry; artifact manifest hashes all runtime files | PASS |
| Deterministic elastic H20 execution | Eight H20 workers; startup records eligible devices, UUIDs and existing PIDs; worker records show pre-import CUBLAS config, matmul/backward probe, deterministic algorithms, TF32 off and autocast off | PASS |
| Resource gates | E5 0.397993 GPU-hours and 179.097 formal wall-seconds; E4+E5 1.027941 GPU-hours and 466.418 wall-seconds; 378,075,471 retained bytes, 2,290,089,984 peak reserved GPU bytes and 1,859,092,480 peak CPU RSS | PASS |
| Scope boundary | Report and terminal contain no bootstrap, p value, Holm result or C4b decision; these remain assigned to E6 | PASS |
| Independent validation covers the objective | Validator reads all 800 results and all manifest records, recomputes hashes, ranks, seeds, curves, AUCs, aggregates, resource accounting and next authorization | PASS |
| Validator failure was governed explicitly | Initial RANKER_SCHEMA false negative is recorded; amendment commit binds the corrected validator and leaves formal result bytes unchanged | PASS |
| Regression and syntax checks | 10 focused E4/E5 tests passed before execution; E5 focused tests, py_compile and diff checks passed after freeze | PASS |
| Remote publication | Producer and validator-amendment commits are on origin/stage-d-d3-pass; final result seal is committed and pushed separately | PASS |

Completion decision: ACHIEVED.

## Defensive-writing audit

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Result scope | E5 only generated per-image C4b inputs and did not run global inference | D12 | Minor | Yes | It marks the registered stage boundary once | Clear separation of execution from inference | Preserve once | No change |
| Descriptive table interpretation | C4b requires the frozen common set, effect threshold, bootstrap precision gate, 16 sign tests and 28-test Holm family | CLEAN | None | Yes | It specifies the pending inferential procedure | Prevents overinterpretation of medians | Preserve | No change |
| Validator amendment | Formal result bytes were unchanged | CLEAN | None | Yes | It states the provenance consequence of a validator-only correction | Verifiable and concise | Preserve | No change |

### Paper-level defensive-writing score

1/10 (clean). The result is reported first; one concise boundary assigns inferential decisions to E6.

### Highest-priority removals

None.

### Necessary caveats to preserve

- E5 PASS establishes execution and data validity, not C4b support.
- The table is descriptive until the frozen E6 analysis is run.
- The first validator failure changed only validator schema interpretation, not formal result bytes.

### Repeated phrase audit

No repeated reviewer-facing prebuttal, apology, or promotional compensation was found.
