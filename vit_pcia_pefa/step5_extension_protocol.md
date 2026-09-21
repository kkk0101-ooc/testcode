# Step 5 mechanism extension: prospective protocol

## Status and boundary

This extension was proposed after the terminal result of `step5_pilot_r8_20260921` was known. It is a new, prospective Step-5 pilot and cannot alter that root's `NOT_SUPPORTED_EMPIRICAL_VALUE` decision. It does not authorize Step 6, does not read the evaluation split, and does not promote a positive extension result to confirmatory evidence.

The priority is fixed as: local circulation/curl correspondence, nonlinear-condition amplification, then source-resolution PEFA. All targets remain the frozen ground-truth raw logit on the same eight calibration-prefix images and two registered models.

## E1: local loop and path-order difference

For each image and model, two deterministic orthonormal directions `v1,v2` are generated in normalized-input coordinates. For each registered head port at block 5, the signed local prediction is

```text
kappa_e = v1^T (J_e^T - J_e) v2.
```

This is compared with the finite path difference between `v1 then v2` and `v2 then v1`, equivalently the circulation around the oriented parallelogram

```text
x -> x + eps*v1 -> x + eps*v1 + eps*v2 -> x + eps*v2 -> x.
```

The endpoint is identical by construction; ordinary Endpoint or logit differences at that endpoint are not used. Port-field line integrals use eight-node Gauss-Legendre quadrature on every edge. Frozen scales are `eps = 0.25, 0.5, 1.0` in normalized-coordinate L2 units. The local prediction is `eps^2*kappa_e`.

The claim is `SUPPORTED` only when both models meet all of: median within-image absolute-rank correlation at least 0.50; median signed cosine at least 0.80; median eligible-port sign agreement at least 0.75; median relative slope error at most 0.50; median circulation-norm scaling exponent in `[1.5,2.5]`; and median rank advantage over the deterministic random baseline at least 0.10. Complete-cut circulation cancellation and identical endpoints are validity checks, not empirical wins.

## E2: stronger local nonlinearity

The same loop is evaluated at a deterministically generated boundary-directed point. Starting from the clean input, the ground-truth-versus-best-other margin gradient defines one raw-RGB L-infinity descent direction. The fixed candidates are `1,2,4,8 / 255`; the first candidate crossing the decision boundary is selected, otherwise the candidate with minimum absolute dynamic margin is selected. No sample is removed. This condition is an evaluation perturbation and does not modify the primary PEFA identity path.

The clean eight-image set is also split within each model into the four lowest and four highest absolute clean margins by a frozen ordering rule. The claim is `SUPPORTED` only if, in both models: at least 75 percent of boundary probes reduce absolute margin or cross; median boundary/clean curl norm and circulation-density ratios are each at least 1.25; boundary signed cosine remains at least 0.50; and the low-margin half has larger clean circulation density than the high-margin half. Otherwise the valid result is `NOT_SUPPORTED`.

## E3A: source-resolution PEFA staircase

The first two calibration-prefix images in both models form a four-task cost-bounded pilot. At block 5, head-port source-resolved `R` and `T` are computed on the straight zero-reference path with sixteen-node Gauss-Legendre quadrature for spatial source partitions `4x4` and `7x7`. Each source cell is matched to zero-baseline occlusion of the same cell. Comparators are Endpoint mass, input norm, PCIA direct mass, and a deterministic random score. The primary intervention outcome is Endpoint-map L1 change; target-logit absolute change is secondary. Source-group sums must close to independently contracted `R` and `T` under the same quadrature; terminal-`T` convergence is reported separately and cannot be relabelled as exact closure.

Advancement to a fresh `14x14` root requires, in both models, the `7x7` PEFA-R mean within-image Spearman to exceed every simple comparator by at least 0.05, exceed its `4x4` value by at least 0.05, and improve in every registered image. Failing this gate stops the staircase without treating unrun `14x14` or pixel-level work as a scientific failure. Pixel-level work additionally requires a fresh root after a successful `14x14` gate and a prospective cost projection within the remaining resource cap.

## Shared audit contract

- Seed namespace: `ENDPOINT_PCIA_PEFA|step5_extension_r1`.
- Dynamic global queue; at most four eligible H20 workers; no termination of existing processes.
- Per-worker GPU cap 40 GiB, CPU RSS cap 32 GiB, task timeout 3600 seconds, at most two attempts.
- Extension cap 24 H20-hours; cumulative Step-5 cap remains 64 H20-hours; retained cap 128 GiB.
- Float32 model/derivatives and float64 reductions. TF32 and autocast disabled; deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` required before CUDA import.
- Claims use only `SUPPORTED / NOT_SUPPORTED / INVALID`. Resource or numerical invalidity is not converted into a scientific negative.
- `step5_pilot_r8_20260921` remains immutable, and `step6_started=false` is required in every receipt.
