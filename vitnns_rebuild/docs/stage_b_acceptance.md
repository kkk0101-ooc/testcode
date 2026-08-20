# Stage B acceptance contract

Stage B implementation acceptance combines automated tests, a full 12-block development audit, and an independent real-input repeatability/sanity run. A retained paper artifact is formal only when a clean committed run records `formal=true`; dirty-tree outputs remain development evidence.

## Method and input gates

1. Block, head, query token and head channel unambiguously identify one pre-projection `O_h = A_h @ V_h` scalar. Class metadata is not confused with the Value-path conservation target.
2. Attribution consumes real timm-forward tensors and does not replay a hand-written Transformer Block.
3. The method is `conditional_fixed_attention_value_path_ig`: attention is fixed at the real endpoint. A separately named `full_model_ig` covers LayerNorm, GELU, residual, QK and softmax through the real autograd graph, without being presented as NNS or Value-path attribution.
4. NNS and direct use the identical image tensor, model, preprocessing and target context; Stage A endpoint parity is recorded.
5. Native raw attribution is signed RGB float32 `3 x 224 x 224`; metric-ready attribution is signed float32 2D without display transforms. Resize cannot establish native pixel resolution.

## Numerical gates

6. Linear, LayerNorm, GELU, residual and attention primitives each require unit tests before inclusion in the claimed propagation path. Out-of-scope primitives must remain explicitly documented.
7. Attention intervention and fixed-attention endpoint behavior have real-block tests; Stage A continues to cover Q/K/V order, scale and extraction.
8. Gauss-Legendre IG uses a strictly increasing schedule; development used `[512, 1024, 2048, 4096, 8192]`. Invalid/unknown config fails closed.
9. All 12 blocks record endpoint values, sums, absolute/relative conservation errors, step counts and convergence. The formal threshold is preregistered; current development evidence has every block below `1%`.
10. Repeatability satisfies configured tolerances. Independent real block-0 evidence observed max-abs `0`.
11. Sanity meets its preregistered threshold. Independent real block-0 evidence observed relative L1 `0.08975 >= 0.05`; the 12-layer development audit itself had sanity disabled.

## Retention and budget gates

12. `debug`, `paper` and `audit` are explicit profiles. `paper` emits no PNG, per-pixel or per-dimension files and retains only preregistered compact 2D maps.
13. `audit` retains float32 raw RGB and float32 signed 2D metric-ready maps sufficient to reproduce paper metrics and measure quantization error. Display is never a metric input.
14. Static preflight estimates bytes and batched model invocations; exact writes enforce the byte cap. The checked-in config caps both forward and backward calls at 30,000.
15. Actual paper output remains below its declared single-sample budget; the audit subset reproduces paper metrics and reports storage quantization error.
16. Outputs do not overwrite; incomplete runs cannot finalize; provenance inventories kind, dtype, shape, size and hash.

## Provenance and formal gates

17. Provenance records commit/clean state; model, image, config and lock hashes; preprocessing; runtime/device/dtype/seed; CUBLAS setting; target/baseline/schedule; compute counts; validations; profile; estimated/actual bytes; and artifacts.
18. `formal=true` requires a clean commit plus enabled and passing repeatability/sanity, parity, conservation, and finalized artifacts within storage/compute budgets. Phase-level acceptance additionally requires a separate all-12-block audit; a selective paper profile is not falsely relabeled as full-depth coverage.
19. Any disabled required gate forces `formal=false`; CLI `COMPLETE` means execution completed, not formal acceptance.

## Current development evidence — not formal

- HEAD `1854838`, dirty tree, NVIDIA H20 GPU 1, `CUBLAS_WORKSPACE_CONFIG` set.
- Input `n03976467/n03976467_9546.JPEG`; no unverified common name asserted. Stage A parity errors all `0`.
- Schedule `[512, 1024, 2048, 4096, 8192]`.
- Block 0–11 relative errors: `0.000716`, `0.000437`, `0.000002`, `0.0000535`, `0.000554`, `0.0000566`, `0.000642`, `0.001498`, `0.000836`, `0.002175`, `0.004749`, `0.004924`; all below `1%`.
- Actual `9,164,616` < estimate `9,748,480` < cap `67,108,864` bytes.
- 12 raw RGB float32, 12 metric-ready signed 2D float32, 0 PNG; development path `/tmp/vitnns_stage_b_audit`.
- The 12-layer audit disabled repeatability/sanity, so `formal=false`. Independent block-0 evidence: repeat max-abs `0`, perturbation relative-L1 `0.08975 >= 0.05`.

The implementation gate is satisfied by the combined evidence above. After commit, a clean-tree `formal=true` run should be used whenever retaining a formal paper artifact tied to the final revision.
