# M5: Method semantics and reference dependence

For a registered scalar target $F$ and the normalized input $x$, Endpoint is
$E(x)=x\odot\nabla_xF(x)$. It requires no externally chosen intervention reference. This is not coordinate-free: if $x=(r-\mu)/\sigma$, then
$x_i\partial F/\partial x_i=(r_i-\mu_i)\partial F/\partial r_i$. The normalization mean is therefore the raw-RGB origin of canonical Endpoint. For a logit target, Endpoint is Input × Gradient in normalized-input coordinates.

We reused the sealed M1 records for two references. Within every image and block, all attention heads were ranked separately; the 12 block-level Spearman coefficients were averaged per image and summarized over 400 images. Endpoint absolute mass and signed sum were exactly identical across references in all 163,200 paired component records. By contrast, Taylor magnitude and absolute finite effect changed in every paired record. Exact head rankings agreed across references in only 46/9,600 Taylor comparisons and 64/9,600 finite-effect comparisons.

| Model | Taylor cross-reference rho | Effect cross-reference rho | Endpoint rho: zero / blur | Taylor rho: zero / blur | Taylor−Endpoint: zero / blur |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 | 0.3782 | 0.3759 | 0.2815 / 0.2727 | 0.6742 / 0.6856 | 0.3878 / 0.4082 |
| DeiT-S/16 | 0.3000 | 0.3071 | 0.1952 / 0.1476 | 0.6095 / 0.6238 | 0.4048 / 0.4738 |

The low cross-reference concordance shows that the intervention reference changes component ordering, not merely score scale. Endpoint retains one observed-input ranking and its association is evaluated against each reference-specific outcome. Activation-space Taylor includes the corresponding displacement and had higher behavioral association in all four model-reference combinations. The sealed M2 arrays provide the same task-level pattern for spatial selection: regional Taylor had higher median AUC in all 24 model-reference-block-outcome groups, with the original Holm decisions retained.

| Method | Primary question | Internal scalar | Input map | External reference | Local-Jacobian closure | Finite replacement |
|---|---|---:|---:|---:|---:|---:|
| Endpoint | observed-input local dependence | yes | yes | no | yes | empirical, limited |
| Activation-space Taylor | specified replacement prediction | yes | regional score | yes | not primary | strongly matched |
| Integrated Gradients | baseline-to-input path attribution | yes | yes | yes | extensible | baseline difference |
| Attention rollout | across-layer attention flow | usually no | yes | no | no | not primary |
| Grad-CAM | class-related coarse localization | partial | yes | no | no | not primary |
| Chefer/AttnLRP | Transformer relevance propagation | partial | yes | usually no | implementation-dependent | not unified |
| Attention weights | token allocation | attention object | patch/token | no | no | not by weights alone |

The comparison concerns estimands and information conditions. Output localization methods answer a different question from registered internal-target scanning, and this study makes no output-localization superiority claim. Path IG remained outside the cross-model release because ViT-B passed only 41/64 ground-truth and 40/64 clean-top1 calibration cases at the frozen maximum of 512 points, below the 61/64 gate.

Observed package costs were 0.079 GPU-hours for the new M1 acquisition and 5.982 GPU-hours for the joint M2 intervention package. These packages share forwards and combine multiple predictors, controls, and counterfactuals, so they do not identify comparable per-method runtimes. Operationally, Endpoint uses an observed-input forward and target backward/VJP; Taylor additionally uses a reference displacement; IG evaluates multiple path nodes; and finite replacement requires candidate counterfactual forwards.
