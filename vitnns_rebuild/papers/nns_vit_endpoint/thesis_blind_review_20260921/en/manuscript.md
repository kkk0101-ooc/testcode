# NNS Structural Path Decomposition and Systematic Validation on Vision Transformer Computation Graphs

## Abstract

Vision Transformers (ViTs) represent an image as a token sequence and transform it through multi-head self-attention, feed-forward networks, and residual connections. Input attribution methods can show which pixels a class output depends on, while representation and attention analyses can show how tokens and channels respond. A structural representation that preserves input coordinates, follows the actual ViT computation edges, and admits itemwise validation remains useful for connecting these two views. Starting from the learning image of the Neural Network Scanner (NNS), this thesis studies its mathematical identity, structural path decomposition on ViT computation graphs, depth-wise patterns, and local approximation range for finite internal replacement.

First, this thesis proves that a scalar NNS learning image equals Gradient × Input for the same target when every layer uses the exact local Jacobian at the current forward state and no normalization, truncation, or relevance redistribution is introduced. This identity specifies the base object. The method then registers ViT token-channel units, channel aggregates, attention heads, residual edges, and class outputs in one input-coordinate system. Relative to a specified graph and valid computational cut, an attention differential is separated into query-, key-, and value-induced paths, and a residual differential into identity and branch paths. A function-preserving linear reparameterization counterexample shows that the total input map can remain unchanged while its components change; the structural decomposition therefore depends on the graph, cut, and parameterization.

Experiments use two fixed ImageNet pretrained parameter states, ViT-B/16 and DeiT-S/16. Numerical validation includes 5,000 NNS formula regressions, 76,800 positive closure tests, 28,800 omission or duplication negative controls, 6,144 analytic Jacobian-vector product checks, and 88 operator-direction checks. With the image as the inferential unit, all 30 model-sample-metric combinations from five independent token-channel samples reproduce increasing entropy, decreasing locality, and increasing distance. Population statistics further show distinct depth trajectories for query, key, and value paths. In multi-amplitude attention-head-output interpolation, absolute error increases monotonically in all four model-reference combinations and all 1,600 image-reference curves; head-ranking correlation decreases from 0.9564–0.9810 at the smallest amplitude to 0.6095–0.6856 under complete replacement. Under complete replacement, local path maps have broad but non-universal associations with component effects. For reference-specified regional replacement, regional Taylor estimates that explicitly use the displacement outperform local maps computed only at the clean input.

The new contribution is a unified implementation of NNS for heterogeneous ViT targets and actual computation edges, graph-relative path accounting, and a systematic body of evidence that separately evaluates mathematical identity, numerical correctness, structural patterns, and finite-intervention utility. The conclusions apply to the two fixed parameter states, frozen samples, and prescribed replacement operators evaluated here.

**Keywords:** Vision Transformer; Neural Network Scanner; learning image; local Jacobian; structural path decomposition; finite intervention

## Contents

1. Introduction
2. Related Work
3. Mathematical Foundations and Boundaries of NNS
4. Structural Path Decomposition for ViTs
5. Research Design and Statistical Analysis
6. Depth-wise Patterns of Structural Paths
7. Local Approximation and Ranking under Finite Interventions
8. General Discussion and Conclusion

# Chapter 1 Introduction

## 1.1 Background

A Vision Transformer partitions an image into patches, projects them linearly, and processes the resulting tokens together with a class token through Transformer encoders [1-3]. Its internal objects are heterogeneous: token-channel units describe responses at specific positions, attention matrices control information mixing across tokens, projected multi-head outputs combine multiple streams, and residual connections add an identity route to a nonlinear branch. A final saliency map or a last-layer attention map alone does not identify which actual computation edges carry a local response.

NNS associates an internal neuron with a learning image that preserves input position [4]. Extending this representation to a ViT is not a matter of renaming an existing input-attribution formula. It requires answers to three structural questions: how heterogeneous targets can share an input-coordinate representation; how query, key, value, head, and residual paths can be separated without omission or duplication; and over what range the resulting local differential quantities approximate finite internal replacements.

The study therefore follows a representation → contribution → intervention evidence chain. The representation stage identifies the mathematical object. The contribution stage builds additive path terms on a specified graph. The intervention stage tests their range with actual internal replacements. Numerical closure establishes implementation correctness but does not substitute for finite-effect evaluation.

![Figure 1 NNS–ViT study framework](../../publication_r2_20260920/figures/png/fig_01_concept_overview.png)

**Figure 1-1 NNS–ViT study framework.** Normal forward computation produces a prediction; each registered internal or output scalar target is associated with a learning image in input coordinates.

## 1.2 Research Questions

This thesis addresses four research questions.

1. **RQ1: What is the mathematical identity and scope of an NNS learning image?** This question relates layerwise local-scale propagation to the Jacobian chain rule, Gradient × Input, the input-coordinate origin, and function-value completeness.
2. **RQ2: How can NNS be structured on a specified ViT computation graph?** This question concerns heterogeneous target registration, valid computational cuts, query/key/value, head, and residual paths, closure, and non-uniqueness.
3. **RQ3: How do structural paths vary with depth in two fixed pretrained parameter states?** This question studies the spatial extent of sampled units, query/key/value paths, routing/value paths, residual identity/branch paths, independent samples, and parameter-state controls.
4. **RQ4: What is the range of local path quantities under prescribed finite replacements?** This question studies first-order error as an attention-head output moves from the clean state toward a reference state, together with component ranking and regional selection under complete replacement.

The four questions concern mathematical identity, structural representation, empirical patterns, and task utility. RQ1 and RQ2 establish the method and its correctness; RQ3 and RQ4 provide empirical evaluation and cannot replace the first two forms of evidence.

The four questions are not parallel hypotheses at one level. RQ1 is a definition and mathematical-identification question that determines the object used downstream. RQ2 is a construction and correctness question that allocates this object to actual ViT structural edges. RQ3 is a descriptive empirical question about depth patterns in fixed models and sampled targets. RQ4 is a finite-task utility question about how much numerical and ranking information remains when a local quantity moves away from the current point. The first two questions provide the measurement basis for the latter two; the latter two do not redefine the mathematics of the first two.

This hierarchy also determines the required evidence. Derivation and numerical regression support the identity; valid-cut conditions and closure negative controls support structural decomposition; image-level intervals and independent target samples support depth patterns; and actual replacements support finite utility. Each level uses evidence matched to its question, avoiding substitution of formula correctness for empirical generality or visual examples for intervention results.

## 1.3 Research Scope and Technical Route

The study first expresses NNS as Jacobian propagation on a directed acyclic computation graph and obtains Gradient × Input for a scalar target. It then registers targets and actual edges in frozen ViT implementations, constructing cut decompositions for query, key, value, attention heads, identity routes, and branches. Formula regression, closure, negative controls, analytic Jacobian-vector products, and finite differences jointly assess correctness.

The empirical study separately examines units, attention, and residual paths. Depth analyses use absolute mass, spatial entropy, locality, normalized distance, cancellation, relative scale, and approximation error. Finite interventions use normalized-zero and blurred-input references and evaluate head-output interpolation, complete component replacement, and fixed-area regional replacement. Population intervals are computed after within-image aggregation, with images as the inferential units.

The technical route has four stages. The first audits the mathematical object by mapping each original layerwise scale rule to the Jacobian chain rule while retaining the actual derivative choice at nonsmooth points. The second implements structure through a target registry for heterogeneous tensor shapes, defines cuts by actual use edges rather than tensor names, and constructs omission and duplication controls. The third estimates population patterns on images, tokens, channels, and class targets fixed in advance. The fourth places local differentials, reference displacements, and actual effects in one finite-replacement framework, separately studying amplitude error and ranking tasks.

Each stage retains raw arrays, summaries, and validation receipts. This organization locates an error at the level of object definition, graph accounting, numerical implementation, or statistical interpretation, rather than mixing infrastructure or implementation problems into scientific conclusions.

## 1.4 Contributions

Each contribution is stated through prior work, the present increment, and direct evidence.

1. NNS, Gradient × Input, and hidden-unit path attribution already provide learning images, pointwise input-weighted gradients, and internal attribution along a reference path [4,6-9]. This thesis adds unified target registration and input-coordinate implementations for ViT token-channel units, channel aggregates, attention heads, residual edges, and class targets. Direct evidence comprises 457 registered scalar targets per ViT-B/16 image and 385 per DeiT-S/16 image, 5,000 formula regressions, and 88 operator-direction checks.
2. GlobEnc, DecompX, AttnLRP, and LibraGrad already propagate Transformer token contributions or relevance [26,28-30]. This thesis keeps the actual local Jacobians and accounts for query, key, value, multi-head, and residual edges on specified valid cuts of frozen ViT graphs. Direct evidence comprises 76,800 positive closure tests, 28,800 negative controls, and 6,144 analytic directional checks.
3. Activation patching and edge attribution study finite internal effects and their first-order approximations [31-34]. This thesis brings structural depth statistics, five independent target samples, multi-amplitude interpolation, and complete-replacement ranking into one design across two fixed ViT parameter states, while reporting correctness, descriptive patterns, and finite-intervention utility separately.

The Gradient × Input identity and chain rule specify the object and are not proposed as a new foundational attribution theory. The structural increment lies in target registration, cut definitions, actual-edge accounting, and systematic evaluation for frozen ViT implementations.

## 1.5 Thesis Organization

Chapter 2 reviews input attribution, Transformer explanation, and internal intervention. Chapter 3 derives the mathematical identity and boundaries of NNS. Chapter 4 defines ViT structural paths, closure, and a non-uniqueness counterexample. Chapter 5 specifies samples, models, estimands, and statistical analyses. Chapter 6 reports depth-wise structural patterns. Chapter 7 reports multi-amplitude and complete-replacement results. Chapter 8 integrates the findings, limitations, and conclusions.

## 1.6 Chapter Summary

This chapter limits the study to local input dependencies and structural components on specified ViT computation graphs. The four research questions connect mathematical identity to finite intervention. The thesis contribution is based on structural implementation, graph-relative decomposition, and tiered validation rather than a new name for the base formula.

# Chapter 2 Related Work

## 2.1 Input Gradients, Path Attribution, and Completeness

Input gradients directly measure local output sensitivity [5], while Gradient × Input weights those gradients by the current input coordinates [6,11]. Integrated Gradients averages gradients along a path from a reference input and satisfies corresponding completeness axioms [7]. DeepLIFT propagates contribution differences relative to reference activations [6], and conductance extends path integration to hidden units [8]. FullGrad combines the input-gradient term with bias-gradient terms from all layers to form a function-value-complete representation [9], while layer-wise relevance propagation and gradient smoothing provide distinct backward-allocation and denoising strategies [10,12].

These methods first differ in their objects. Gradient × Input is local; path integration and DeepLIFT require references; FullGrad completeness includes bias terms. The present study examines closure of local input differentials on valid computational cuts, which is distinct from function-value decomposition and reference-path completeness.

“Completeness” is not one concept across this literature. Integrated Gradients completeness sums path attribution to an output difference; FullGrad completeness includes input and bias terms; layer-wise relevance propagation emphasizes relevance conservation. Closure in this thesis requires only that one local input differential be partitioned without omission or duplication on a valid cut. A method comparison must first align the quantity being summed, because the same word can otherwise conceal different equalities.

Pointwise and path methods also use different information budgets. Gradient × Input requires only the current input and its gradient, whereas Integrated Gradients and conductance require a reference and gradients along a path. The former is direct and suited to local structure; the latter can incorporate nonlinear variation along the path. Chapter 7 turns this distinction into a multi-amplitude experiment instead of inferring relative performance from definitions alone.

## 2.2 Visualization, Perturbation Evaluation, and Reliability Checks

Grad-CAM uses gradient-weighted convolutional features to produce class-related regions [13], while deconvolution and occlusion inspect intermediate features or input perturbations [14]. LIME and SHAP explain predictions through local surrogate models and game-theoretic allocation [15-16]. Similar-looking visualizations need not represent the same statistical object. Parameter-randomization tests, deletion and retention benchmarks, and audits of modified backward rules show that visual readability alone does not establish faithfulness [17-20].

This study consequently separates display from statistics. Blue-base learning images are used to read spatial structure; every conclusion is computed from raw signed arrays, absolute mass, or actual finite replacement. Parameter-state controls assess dependence on a pretrained state without interpreting a contrast between one pretrained model and random initializations as a causal training effect.

Saliency reliability has at least three requirements. The first is parameter sensitivity: an explanation should measurably change on the corresponding object when model parameters change [17,19]. The second is task relation: highly ranked positions should have corresponding effects in a prescribed deletion, retention, or replacement task [18]. The third is numerical and implementation stability: conclusions should not be driven primarily by display normalization, interpolation, or an accidental backward rule. Parameter-state controls, finite internal replacement, and recomputation from raw arrays address these three requirements, respectively.

These requirements also specify the role of case figures. A case can reveal the pixel-space form of a path and help identify implementation errors or hypotheses worth testing; a population claim requires prespecified statistics on the full sample. Display examples are therefore selected for readability and never enter effect-size or interval calculations.

## 2.3 Transformer and ViT Explanation

Attention flow and rollout compose attention matrices across layers [21]. Chefer et al. extend relevance propagation to Transformer attention and skip connections [22]. Counterexamples and interventions challenge the sufficiency of “attention as explanation” [23-24], while vector norms can change contribution estimates derived from attention weights alone [25].

GlobEnc incorporates the full encoder layer into token-contribution propagation [26]; ALTI analyzes contextual mixing [27]; DecompX propagates token representation components to predictions [28]; AttnLRP and LibraGrad construct Transformer attribution through relevance conservation and gradient-flow balancing [29-30]. These methods address token contributions, relevance, or input attribution. The present method retains the model's actual local derivatives and groups one local input differential along specified actual computation edges.

Transformer explanation also requires separating forward mixing from target-conditioned dependence. Attention weights and ALTI-like quantities mainly describe token relations through forward information mixing; class-related gradients and relevance methods include a downstream target in backward propagation. A head can have sharp forward weights yet have a small local path for a class target because its downstream cotangent is small. This thesis carries the output projection and every derivative after the target into each head's cotangent to preserve that target-conditioned meaning.

Token contribution and input-pixel dependence also have different native resolutions. The former is commonly defined on a $14\times14$ patch grid or token sequence; the latter additionally maps through patch-projection weights into pixel coordinates. This thesis reports native pixel maps and patch statistics and excludes display interpolation from numerical analysis, so a difference in spatial resolution is not presented as method superiority.

## 2.4 Internal Representations, Activation Replacement, and Edge Attribution

Studies of internal neurons and feed-forward layers link representational units to model behavior [8,35]. Activation patching replaces an internal variable from a clean state with the value produced by a reference input and measures the output change [31]; causal tracing applies this operation to factual associations [34]. Attribution patching and edge attribution patching approximate finite replacement with an inner product between activation displacement and a local gradient [32], and AtP* further treats attention saturation and cancellation [33].

Such methods directly concern finite effects, but their results depend on the reference, replaced variable, operator, and output target. This thesis retains the clean local map, the reference-aware Taylor quantity, and the actual replacement effect as separate objects and observes first-order error along a multi-amplitude interpolation.

“Causal” in activation replacement denotes a model-internal effect under a prescribed operation on a computation-graph variable. Replacing one head while keeping other heads and the identity route clean creates a hybrid forward state. It answers how that operation changes the target logit, not the causal contribution of a natural image region in a data-generating process. The variable, reference, and unchanged edges are included in the experimental contract so that the conclusion can be reproduced for the same operation.

The computational advantage of first-order edge attribution is that one backward pass can provide local predictions for many components. Its error can arise from gradient change along the displacement, softmax saturation, nonlinear gate changes, and path cancellation [32-33]. Chapter 7 therefore reports absolute error, relative error, and head-ranking correlation. The first two evaluate numerical approximation; the third evaluates ordering information in a screening setting.

## 2.5 Method Positioning

**Table 2-1 Comparison by object and evaluation**

| Method | Primary object | External reference | Internal-path decomposition | Completeness or closure | Finite internal replacement |
|---|---|---|---|---|---|
| Gradient × Input | Input coordinates at one point | No | No | Generally no function-value completeness | Not intrinsic |
| Integrated Gradients | Input-path attribution | Yes | No | Path-attribution completeness | Usually input perturbation |
| Conductance | Hidden-unit path attribution | Yes | Hidden units, not graph-edge cuts | Complete for its path object | May evaluate neuron ablation |
| FullGrad | Input and bias terms | No | Layerwise bias terms | Full-gradient representation completeness | Usually input perturbation |
| Attention rollout | Cross-layer attention flow | No | Attention matrices | No class-score closure | Usually input-token perturbation |
| GlobEnc / ALTI | Encoder token contribution | No | Token or component propagation | Defined by each decomposition | Usually input tokens |
| DecompX | Token representation components of prediction | No | Encoder components | Specified token decomposition | Token erasure |
| AttnLRP | Input and latent relevance | No | Attention and skip connections | Relevance conservation | Input perturbation |
| LibraGrad | ViT input attribution | No | Modified backward paths | Targets FullGrad completeness | Input perturbation |
| Activation patching | Finite effects of internal variables | Yes | Selected nodes or edges | Not an additive-completeness objective | Direct evaluation |
| Edge attribution approximation | First-order estimates of edge effects | Yes | Yes | Not an additive-completeness objective | Compared with replacement |
| NNS–ViT in this thesis | Local input maps for ViT targets and edges | No for maps; yes for replacement evaluation | Specified Q/K/V, heads, and residual cuts | Local-differential closure on valid cuts | Prescribed replacement tasks |

## 2.6 Literature Synthesis and Research Gap

Existing work covers input gradients, path attribution, Transformer token decomposition, and internal replacement. The missing connection addressed here is to map heterogeneous ViT targets into common input coordinates without changing their actual local Jacobians, form verifiable differential cuts along actual edges, and then evaluate correctness, depth-wise descriptions, and finite-replacement utility separately.

Table 2-1 reveals three differences from the closest method families. Relative to Gradient × Input, this thesis adds internal target registration and structural cuts rather than changing the endpoint formula. Relative to Transformer token decomposition, it studies components of the actual local Jacobian in input-pixel coordinates. Relative to activation patching, it first constructs local structural maps that require no external reference and then evaluates them with explicitly referenced replacement tasks. These three distinctions organize the comparisons in later chapters.

## 2.7 Chapter Summary

This chapter organized related methods by object, reference, internal path, and utility evaluation. NNS–ViT does not replace foundational attribution methods; it adds unified target registration, actual-edge decomposition, and tiered evidence for a specified ViT computation graph.

# Chapter 3 Mathematical Foundations and Boundaries of NNS

## 3.1 Chapter Question

This chapter answers RQ1: what layerwise NNS learning-image propagation represents on a general computation graph, under what conditions it equals Gradient × Input, and how the coordinate origin, non-homogeneous terms, and nondifferentiable points bound the interpretation.

## 3.2 Method: Learning Images on a Computation Graph

Let a network be a directed acyclic graph $G=(V,E)$ with column-vector input $x\in\mathbb R^d$ and scalar target $s(x)$. For an intermediate variable $u(x)\in\mathbb R^m$, define the learning-image matrix $\mathcal L(u;x)\in\mathbb R^{m\times d}$. Input initialization and a multi-input operator propagate as

**Equation (3-1)**

$$
\mathcal L(x;x)=\operatorname{diag}(x),\qquad
\mathcal L(v;x)=\sum_{a=1}^{r}J_{f,u_a}(x)\mathcal L(u_a;x),\quad
v=f(u_1,\ldots,u_r).
$$

Equation (3-1) evaluates every local Jacobian at the actual forward state. Linear layers, convolutions, and active ReLU gates use their usual derivatives; max pooling uses the current forward winner; fixed framework choices apply at ReLU zeros and max ties.

**Proposition 3-1 (scalar-target identity).** If all local Jacobians are evaluated at the same current state and propagation introduces no normalization, truncation, or relevance redistribution, then

**Equation (3-2)**

$$
\mathcal L(u;x)=J_{u,x}(x)\operatorname{diag}(x),\qquad
E_s(x)=\mathcal L(s;x)^\top=\operatorname{diag}(x)\nabla_xs(x).
$$

**Proof.** At the input, $J_{x,x}=I$. Assume the statement holds for all predecessors in topological order. Substitution into Equation (3-1), followed by the multi-input chain rule, gives $J_{v,x}\operatorname{diag}(x)$. Induction to the scalar target and transposition yield $E_s(x)=x\odot\nabla_xs(x)$.

The quantity is the first-order response along coordinate-scaling directions. For $x_i\mapsto(1+\epsilon)x_i$,

**Equation (3-3)**

$$
s(x+\epsilon x_i e_i)-s(x)=\epsilon E_{s,i}(x)+o(\epsilon).
$$

A general network satisfies

**Equation (3-4)**

$$
\sum_{i=1}^{d}E_{s,i}(x)=x^\top\nabla_xs(x)\neq s(x).
$$

Euler's equality applies to first-degree homogeneous functions at differentiable points. A general ViT with biases, class tokens, positional embeddings, LayerNorm, softmax, and GELU does not satisfy this condition.

If raw pixels $r$ are normalized as $x=(r-\mu)/\sigma$, then

**Equation (3-5)**

$$
x_i\frac{\partial s}{\partial x_i}=(r_i-\mu_i)\frac{\partial s}{\partial r_i}.
$$

The local input map therefore requires no separately selected intervention reference, but depends on the model-input coordinate origin.

## 3.3 Validation Method

Formula validation covers a fully connected network, a convolutional network, a convolutional network with max pooling, a residual multilayer perceptron, and a residual convolutional network. Each class uses 100 inputs in two numerical precisions, giving 5,000 comparisons between layerwise propagation and automatic-differentiation Gradient × Input. Eleven operator classes provide 88 direction checks, supplemented by finite differences and analytic Jacobian-vector products.

Validation uses three complementary mechanisms. Formula regression compares two independently organized computations: explicit learning-image propagation and direct automatic differentiation followed by multiplication with the input. Operator-direction checks decompose a complex network into linear, convolutional, pooling, normalization, and nonlinear objects and test the directional response of each local Jacobian. Finite differences do not reuse the analytic backward formula and test first-order predictions through changes in target values under small perturbations.

Floating-point precision affects both target differences and relative-error denominators. When the target response in a direction is small, a float32 finite difference may be dominated by rounding. The protocol retains every float32 exceedance and repeats the same input, coordinate, and direction in float64 instead of dropping difficult cases or widening tolerance. This separates numerical conditioning from the formula identity.

## 3.4 Results

**Table 3-1 Maximum relative L2 error in NNS formula regression**

| Compatible network | Float64 | Float32 | Comparisons |
|---|---:|---:|---:|
| Fully connected | $4.371\times10^{-16}$ | $1.964\times10^{-7}$ | 1,000 |
| Convolutional | $4.289\times10^{-16}$ | $2.189\times10^{-7}$ | 1,000 |
| Convolutional plus max pooling | $4.769\times10^{-16}$ | $2.427\times10^{-7}$ | 1,000 |
| Residual multilayer perceptron | $2.686\times10^{-16}$ | $2.155\times10^{-7}$ | 1,000 |
| Residual convolutional | $2.923\times10^{-16}$ | $1.692\times10^{-7}$ | 1,000 |

The float64 maximum is at most $4.769\times10^{-16}$ and the float32 maximum at most $2.427\times10^{-7}$. All 88 operator-direction checks pass. Of 57,600 float32 finite-difference checks in the ViT atlas, 36 exceed the prespecified tolerance; all 36 same-coordinate float64 reviews pass, with the original failures retained. These results validate the correspondence between Equation (3-2) and the implementation. They are not a reproduction of unavailable original-author software.

## 3.5 Discussion

Proposition 3-1 clarifies the mathematics of NNS propagation. A scalar endpoint is not a new foundational input-attribution quantity, and the learning image can be computed through unified automatic differentiation. Equations (3-4) and (3-5) give two material boundaries: the local input map generally does not reconstruct the function value and changes with the input-coordinate origin. The later contributions build on this known local quantity and concern ViT targets and structural paths.

For a vector internal target, each row of the learning-image matrix corresponds to one scalar coordinate. Channel energy and other aggregate targets first map the vector to a scalar and then use the same propagation rule, avoiding direct addition of coordinates with different downstream cotangents. For a multi-input operator, Equation (3-1) explicitly sums Jacobian terms for each input edge, so repeated uses of a shared tensor must be distinguished by computation edge. The valid cut in Chapter 4 is this multi-input chain rule applied structurally to a ViT.

Map sign must also be separated from display. A positive entry is a first-order increase in the target along the current coordinate-scaling direction, and a negative entry is a decrease. Absolute mass is used to study spatial extent and discards direction while retaining magnitude. A blue-base positive view shows only the former morphology. All statistics retain the raw signed arrays and explicitly convert to absolute mass only for spatial distributions.

## 3.6 Chapter Summary

The proposition, proof, and 5,000 numerical regressions answer RQ1. With exact pointwise local derivatives, a scalar NNS learning image equals Gradient × Input; it is a coordinate-dependent local first-order quantity rather than a general function-value-complete decomposition.

# Chapter 4 Structural Path Decomposition for ViTs

## 4.1 Chapter Question

This chapter answers RQ2: how heterogeneous ViT targets can be mapped into input space, how attention and residual paths can be separated along actual computation edges, and which conditions guarantee closure. A reparameterization counterexample then bounds uniqueness.

## 4.2 Method: Internal Scalar-Target Registration

The ViT patch projection maps local pixels to token vectors while learning images remain in pixel coordinates. Four target classes are registered: individual token-channel scalars, channel aggregates across spatial tokens, structural-edge targets, and class outputs. For patch-token activations $h_{t,c}$ of channel $c$, define an energy aggregate

**Equation (4-1)**

$$
s_{\mathrm{eng}}=\frac12\sum_{t=1}^{196}h_{t,c}^{2},\qquad
E_{\mathrm{eng}}=\sum_{t=1}^{196}h_{t,c}E_{h_{t,c}}.
$$

Equation (4-1) is the Jacobian form of activation-weighted channel aggregation, but the energy is an aggregate scalar rather than one neuron. Other targets include patch-token sums, attention-head internal coordinates, final class-token channels, ground-truth logits, and frozen clean-top1 logits. A channel index shared by two models is not treated as the same semantic unit.

A target registry stores target semantics, tensor location, aggregation rule, and downstream scalar together. Token-channel targets retain position and channel indices; channel aggregates record the 196 participating patch tokens; head targets record block, head, and the edge before output projection; class targets record the class index and its freezing rule. The registry allows one executor to process objects with different shapes and prevents a channel chosen for a display figure from being confused with a channel in the population sample.

Target types answer different questions. Individual token-channels support locality and input-distance analyses, energy aggregates connect forward channel responses to input dependence, ground-truth logits support structural path comparisons, and clean-top1 targets support target-sensitivity controls. The types share input coordinates but are not numerically pooled.

## 4.3 Method: Attention Paths

For one attention head, let

**Equation (4-2)**

$$
Q=XW_Q+b_Q,\quad K=XW_K+b_K,\quad V=XW_V+b_V,\quad
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right),\quad O_h=AV.
$$

Its local differential is

**Equation (4-3)**

$$
\mathrm dO_h=(\mathrm dA)V+A\,\mathrm dV,\quad
\mathrm dA=J_{\mathrm{softmax}}\mathrm dS,\quad
\mathrm dS=\frac{\mathrm dQK^\top+Q\mathrm dK^\top}{\sqrt{d_h}}.
$$

The first two terms in Equation (4-3) define query- and key-induced paths and jointly form the routing path; $A\mathrm dV$ defines the value-induced path:

**Equation (4-4)**

$$
E^{\mathrm{attn}}=E^Q+E^K+E^V
=E^{\mathrm{route}}+E^{\mathrm{value}}.
$$

The head edge lies before concatenation and output projection. Each downstream cotangent includes the output projection and every derivative after the target, giving

**Equation (4-5)**

$$
E^{\mathrm{MHA}}=\sum_{h=1}^{H}E_h.
$$

Equations (4-4) and (4-5) group one local differential. Query, key, and value share upstream inputs and may overlap in input space; norm ratios are not mutually exclusive causal percentages.

![Figure 4-1 Attention differential paths](../../publication_r2_20260920/figures/png/fig_06_attention_differential.png)

**Figure 4-1 Attention differential paths.** Query and key change attention weights, while value transports content under the current weights. The three terms reconstruct the full attention differential on the specified cut.

## 4.4 Method: Residual Paths and Valid Computational Cuts

For $y=u+f(u)$ and a downstream scalar $F$, let $g_y=\nabla_yF$. The identity and branch paths are

**Equation (4-6)**

$$
E^{\mathrm{id}}=\operatorname{diag}(x)J_{u,x}^{\top}g_y,\qquad
E^{\mathrm{branch}}=\operatorname{diag}(x)J_{u,x}^{\top}J_{f,u}^{\top}g_y.
$$

The identity term must use $g_y$ at the residual output. Using $\nabla_uF$, which already contains the branch effect, and then adding a branch term would double count that branch.

For a cut-edge set $C=\{z_k\}$ of target $F$, define the path term on input coordinate $i$ by

**Equation (4-7)**

$$
E^C_{k,i}=x_i\left\langle g_{z_k},\frac{\partial z_k}{\partial x_i}\right\rangle.
$$

**Proposition 4-1 (valid-cut closure).** If every relevant input-to-target differential path crosses the cut exactly once, all terms use the same forward state, each cotangent lies on its actual use edge, and no bypass is omitted or path duplicated, then the terms in Equation (4-7) satisfy

**Equation (4-8)**

$$
\sum_{k:z_k\in C}E^C_{k,i}=x_i\frac{\partial F}{\partial x_i}.
$$

**Proof.** Expand the chain rule into Jacobian products over every input-to-target path. A valid cut partitions those paths without omission or duplication. Each group factors into an upstream Jacobian to $z_k$ and the downstream cotangent on that edge. Summing all groups and multiplying by $x_i$ yields Equation (4-8). Cutting only the branch omits the identity bypass; adding a branch term to a complete input gradient duplicates the branch.

A valid cut can be constructed in four steps. First, fix one scalar target and its current forward state. Second, traverse backward from the target to enumerate the actual use edges to be explained and identify any bypass around them. Third, store the upstream input-Jacobian action and downstream cotangent for every edge. Fourth, compare the elementwise sum with the full local input map and construct at least one omitted-bypass and one duplicated-path control. Positive closure and rejection of both controls are required to exclude a vacuous pass in which all terms are near zero.

Multi-head attention particularly requires cuts by use edge rather than tensor storage location. Concatenation combines head dimensions, and output projection rotates head coordinates into model dimensions. Each head is intercepted before projection while its downstream cotangent passes through the same output projection, making Equation (4-5) consistent with the original forward computation. A residual cut analogously includes both the identity input edge and branch output edge.

![Figure 4-2 Residual paths](../../publication_r2_20260920/figures/png/fig_07_residual_principle.png)

**Figure 4-2 Residual paths.** The identity and branch edges use the same output cotangent; their signed path maps sum to the residual output's local input map.

## 4.5 Method: Graph Dependence and Non-Uniqueness

Consider $x=(x_1,x_2)^\top$, hidden variable $h=x$, and output $s=(1,1)h=x_1+x_2$. A cut by hidden coordinate gives

**Equation (4-9)**

$$
E_1=(x_1,0)^\top,\qquad E_2=(0,x_2)^\top.
$$

Let

**Equation (4-10)**

$$
R=\begin{bmatrix}1&1\\0&1\end{bmatrix},\qquad
h'=Rh,\qquad w'=R^{-\top}(1,1)^\top=(1,0)^\top.
$$

By Equation (4-10), the same function is $s=w'^\top h'$, with new hidden-coordinate terms

**Equation (4-11)**

$$
E'_1=(x_1,x_2)^\top,\qquad E'_2=(0,0)^\top.
$$

Equations (4-9) and (4-11) have the same total but different components. Path maps are therefore defined relative to a graph, cut, and parameterization. Closure establishes complete accounting for the specified representation; it does not make components unique functions of the end-to-end mapping.

The counterexample is a general identification issue for internal contribution decompositions rather than an implementation defect. If a research question concerns an end-to-end functional invariant, it should use the total input gradient, function value, or an object with a proved invariance property. If it concerns which edges carry information in a concrete implementation, a graph-relative decomposition is appropriate, provided the implementation and cut are reported. This thesis chooses the latter question and freezes model weights, software paths, and target registration as reproducibility conditions.

Cut granularity also changes explanatory resolution. Treating an attention sublayer as one term supports only a sublayer-versus-identity comparison; cutting by head compares heads; cutting within a head by query, key, and value compares routing and content transport. Finer cuts improve diagnostics but increase cancellation and multiplicity. Results are reported only at these three granularities, all defined through actual edges and a common cotangent and all directly closure-tested.

## 4.6 Method: Spatial and Scale Metrics

For a three-channel local map at pixel $p$, define absolute mass and the patch-mass distribution by

**Equation (4-12)**

$$
M(p)=\sum_{c\in\{R,G,B\}}|E_{p,c}|,\qquad
m_j=\sum_{p\in P_j}M(p),\qquad
q_j=\frac{m_j}{\sum_km_k}.
$$

If total mass is zero, $q_j$ is missing. Normalized spatial entropy, target-patch neighborhood locality, and normalized distance measure extent, near-target concentration, and distance from a target location. The Q/K/V population analysis instead uses the expected distance between two independent draws from $q$, requiring no selected target patch. Signed-map cancellation is

**Equation (4-13)**

$$
C_{\mathrm{map}}=1-\frac{|\sum_iE_i|}{\sum_i|E_i|},\qquad
C_{\mathrm{path}}=1-\frac{\|E^{(1)}+E^{(2)}\|_1}{\|E^{(1)}\|_1+\|E^{(2)}\|_1}.
$$

Routing relative scale is

**Equation (4-14)**

$$
S_{\mathrm{route}}=\frac{\|E^{\mathrm{route}}\|_1}
{\|E^{\mathrm{route}}\|_1+\|E^{\mathrm{value}}\|_1},
$$

with an analogous residual identity scale. Equation (4-13) reports cancellation within signed maps and between paths. The error from retaining only $A\mathrm dV$ is

**Equation (4-15)**

$$
\varepsilon_V=\frac{\|E^{\mathrm{attn}}-E^V\|_2}
{\|E^{\mathrm{attn}}\|_2}.
$$

## 4.7 Structural Closure Results

Across the two models, all 76,800 positive Q/K/V, routing/value, multi-head, and residual closure tests pass; all 28,800 omission or duplication controls are rejected; and all 6,144 analytic Jacobian-vector product checks pass. Independent recomputation gives a maximum Q/K/V closure absolute error of $1.574\times10^{-5}$ and a maximum residual closure absolute error of $2.480\times10^{-5}$. These results support structural-accounting correctness on the specified frozen graphs.

Closure validation spans all 400 evaluation images and 12 blocks in both models, thereby testing input variation and numerical conditioning from shallow to deep blocks. Negative controls omit the identity bypass, duplicate a branch, or break cut completeness; their rejection under the same tolerance shows that positive passes result from the correct edge set rather than a permissive threshold. Analytic Jacobian-vector products directly compare local directional responses and complement input-space component summation.

Compute cost is dominated by scalar-target backward passes and the intermediate edges retained for analysis. A full local map for one scalar requires one backward pass. Path components can reuse upstream states and downstream cotangents at the same forward state, but large token-channel registries still require chunking. The implementation freezes target registration, distributes work by image, streams summaries, and retains selected complete raw maps for independent recomputation and visualization.

## 4.8 Chapter Summary

Target registration, actual edges, and valid cuts answer RQ2. Query, key, value, attention-head, and residual terms close on the specified graph, while the counterexample shows that components change with internal coordinates and cuts. Every subsequent empirical conclusion refers to these explicitly defined objects in the frozen implementations.

# Chapter 5 Research Design and Statistical Analysis

## 5.1 Models, Data, and Sample Construction

Experiments use ImageNet-1K pretrained ViT-B/16 and DeiT-S/16 classifiers [1-2]. Both contain 12 encoder blocks; the former has 12 attention heads and hidden dimension 768, while the latter has 6 heads and hidden dimension 384. Images are resized to a short side of 248 and center-cropped to $224\times224$, then normalized by model-specific parameters. Dropout and stochastic depth are disabled for inference.

Samples follow a deterministic hash rule. Classes with at least one nonexcluded image are ranked by the SHA-256 of namespace, class, and WordNet identifier, and the first 528 are retained. Within each class, nonexcluded filenames are ranked by the SHA-256 of namespace, image, and filename, with one image selected. The first 64 ranked images form the discovery split, the next 64 the calibration split, and the remaining 400 the evaluation split. The splits do not overlap and each class contributes at most one image. Both models share the same 400 evaluation images in the main population analyses.

This rule prioritizes class coverage and makes split assignment a function of outcome-independent metadata. The discovery split supports implementation debugging and figure design, the calibration split fixes numerical tolerances and display rules, and the evaluation split supports primary statistics. Evaluation images are not chosen from model outputs or learning-image appearance after protocol freeze. One image per class reduces class-frequency imbalance in image-level summaries, while making the target population closer to one image from each selected class than to the natural within-class frequency of ImageNet.

The stratified unit study uses a frozen 100-image subset, selecting the class token, 15 spatially stratified patch tokens, and 32 channels per block. Each model-image contains 6,144 post-block scalars, giving 1,228,800 records. Five independent replications retain the same images but resample 16 tokens and 32 channels per block using separate namespaces frozen before outcome access. The parameter-state control uses 32 shared images per model, three independently initialized states, and the architecture-matched pretrained state.

Token sampling covers the class token and spatial patch strata, while channels are generated independently for each block. The five replications change token and channel targets while holding images fixed and therefore test dependence on a particular internal-target sample. Image-level bootstrap within each group quantifies uncertainty over images. These are different sources of variation, so additional bootstrap replicates cannot substitute for new token-channel samples.

**Table 5-1 Models and major samples**

| Item | ViT-B/16 | DeiT-S/16 |
|---|---:|---:|
| Encoder blocks | 12 | 12 |
| Attention heads | 12 | 6 |
| Hidden dimension | 768 | 384 |
| Registered scalar targets per image | 457 | 385 |
| Main evaluation images | 400 | 400 |
| Stratified-unit images | 100 | 100 |
| Parameter-state control images | 32 | 32 |
| Randomly initialized states | 3 | 3 |

## 5.2 Finite-Replacement Design

The multi-amplitude study interpolates an attention-head output $z$ from the clean state to a reference state:

**Equation (5-1)**

$$
z_\alpha=(1-\alpha)z_{\mathrm{clean}}+\alpha z_{\mathrm{ref}},\qquad
\alpha\in\{0.05,0.10,0.25,0.50,1.00\}.
$$

The actual finite effect and first-order prediction are

**Equation (5-2)**

$$
e_\alpha=F(z_{\mathrm{clean}})-F(z_\alpha),\qquad
p_\alpha=\alpha\left\langle\nabla_zF,z_{\mathrm{clean}}-z_{\mathrm{ref}}\right\rangle.
$$

References are produced by normalized-zero and blurred inputs. The experiment fixes the same 400 images, ground-truth logits, all attention heads in all 12 blocks, and both references. It reports absolute actual and predicted effects, absolute error, relative error, and within-image head-ranking Spearman correlation.

The normalized-zero input maps the model-coordinate origin to a reference state, whereas the blurred input preserves low-frequency color and contour while reducing detail. They create different internal displacements and test whether local ranking depends on reference choice. The amplitude study takes five positions along each particular clean-to-reference direction; it is a distance continuation of the same replacement task rather than a generic random-noise sweep.

Complete component replacement uses

**Equation (5-3)**

$$
e_k=F(x)-F(x;z_k\leftarrow z_k^{\mathrm{ref}}),\qquad
p_k=\left\langle\nabla_{z_k}F,z_k(x)-z_k^{\mathrm{ref}}\right\rangle.
$$

The regional task fixes three heads in block 5, partitions the input into $4\times4$ candidate regions, and replaces 1, 2, or 4 regions, up to 25% area. Each curve takes the absolute isolated effect, integrates over 0–25% area, divides by 0.25, and averages the three heads within image. The candidate oracle selects only among evaluated regions and is not a global optimum over all possible regions.

Component ranking and regional selection use different candidate sets. The component task compares heads or residual edges within one structural family, whereas the regional task compares spatial subsets within fixed heads. The former uses Spearman correlation, top-k capture, and regret to evaluate component screening; the latter uses absolute-effect AUC to evaluate accumulated gain under an area budget. Both concern ranking, but a threshold from one task cannot be transferred directly to the other.

## 5.3 Statistical Principles

All population inference first aggregates tokens, channels, heads, or blocks within each image and then uses the image as the sole inferential unit. Intervals use image-level bootstrap [36]; the original confirmatory 28-test family uses Holm correction [37]. Component-correlation threshold 0.10 and regional-gain threshold 0.05 are prespecified practical-effect thresholds in the frozen protocol, not values optimized against current outcomes. Unless stated otherwise, intervals are 95% intervals.

Within-image aggregation avoids treating many units that share one input and model state as independent observations. Medians reduce sensitivity to extreme gradient norms, and resampling complete images preserves within-image dependence. Between-group standard deviations for independent channel-token samples are reported separately because repeated bootstrap on one group estimates image uncertainty but cannot simulate selection of new internal targets.

Multiplicity families are frozen by research question and analysis stage. The original confirmatory family retains its thresholds and Holm decisions. New Q/K/V population statistics and multi-amplitude experiments use their own frozen successor contracts. Post-hoc spatial total-variation and saturation analyses are interpreted separately and do not alter original composite decisions. This separation prevents supplementary findings from changing prior pass or failure labels.

Confirmatory analysis freezes samples, targets, estimands, thresholds, and multiplicity families before outcome access. Supplementary analysis tests parameter states, targets, and method relationships without changing original decisions. Post-hoc analysis explains observed patterns in separate families and does not upgrade confirmatory claims.

## 5.4 Research-Question and Estimand Table

**Table 5-2 Research question–estimand–unit–sample–analysis–conclusion map**

| Research question | Main estimand | Statistical unit | Sample size | Analysis type | Supported conclusion |
|---|---|---|---|---|---|
| RQ1 identity | Relative L2 and direction errors | Network-input comparison | 5 network classes, 1,000 each | Method validation | Implementation agrees with Equation (3-2) |
| RQ2 closure | Maximum absolute/relative error and negative-control rejection | Image-block-path | 400 images/model; 76,800 positives | Confirmatory correctness | Accounting closes on specified graph and cuts |
| RQ3 unit extent | Late-minus-early entropy, locality, and distance | Image | 100 images/model; 5 independent samples | Confirmatory successor replication | Stable broadening across sampled targets in two fixed states |
| RQ3 Q/K/V | Block-11-minus-block-0 mass, entropy, pair distance, cancellation, relative scale | Image | 400 images/model | Confirmatory strengthening | Distinct population depth trajectories |
| RQ3 routing/value and residual | Relative scales, approximation error, absolute mass | Image | 400 images/model | Confirmatory description | Routing share and value-only error rise; residual sublayers differ |
| RQ3 parameter state | Pretrained-minus-random depth contrasts | Image and random seed | 32 images/model, 3 random states | Supplementary | Parameter-state differences, not a causal training effect |
| RQ4 amplitude | Absolute/relative error and head-ranking Spearman | Image | 400 images/model, 2 references | Confirmatory successor experiment | First-order error grows with amplitude on the prescribed interpolation |
| RQ4 component ranking | Spearman, top-k capture, regret | Image | 400 images/model | Original confirmatory and frozen supplementary | Broad but non-universal association; generally stronger Taylor ranking |
| RQ4 regional selection | Absolute-effect AUC, paired difference, normalized regret | Image | 400 images/model | Original confirmatory; cross-layer supplementary | Original local-map composite unsupported; regional Taylor better suited |
| Saturation and spatial sensitivity | Partial rank correlation and spatial total variation | Image | 400 images/model | Post-hoc | Interpretation of attention and residual patterns only |

## 5.5 Visualization and Evidence Management

Blue-base main figures show the positive part of the signed RGB sum and use a fixed quantile and square-root display transform to reveal weak structure. The transform is excluded from every statistic. Raw signed maps, absolute maps, and display parameters are retained in the evidence package. Examples are selected for readability; population conclusions use the full samples in Table 5-2. Numerical files, figure sources, and validation receipts are bound by SHA-256.

Evidence management follows a raw quantity → image-level summary → manuscript number map. Raw files contain path components and actual replacement effects, analysis files recompute image-level statistics, and manuscript tables take only sealed fields. An independent validator recomputes selected raw quantities, checks sample counts and formulas, and compares numerical rows across languages. Figure-pixel hashes bind presentation assets; scientific conclusions remain bound to raw numerical files.

## 5.6 Chapter Summary

This chapter specified deterministic sample construction, models, finite-replacement operators, and image-level inference. Table 5-2 binds each research question to an estimand, unit, sample size, and analysis type, keeping theoretical identities, numerical correctness, descriptive patterns, and intervention utility separate.

# Chapter 6 Depth-wise Patterns of Structural Paths

## 6.1 Chapter Question

This chapter answers RQ3: whether local input dependencies of internal targets broaden with depth; whether query, key, and value paths share one trajectory; how routing/value and residual identity/branch paths change; and whether these findings are stable across independent target samples and parameter states.

## 6.2 Method

The unit analysis defines blocks 0–3, 4–7, and 8–11 as early, middle, and late. Sampled tokens and channels are first aggregated within image, followed by the late-minus-early contrast. Five independent samples retain the same 100 images but resample tokens and channels; directional consistency is assessed across model-sample-metric combinations.

The Q/K/V population analysis uses 400 images per model, all 12 blocks, and frozen ground-truth logits. Absolute mass, entropy, normalized pairwise distance, cancellation, and relative scale from Equations (4-12)–(4-14) are computed separately for the three paths, with image-level median block-11-minus-block-0 differences and intervals. Routing/value and residual analyses use the same 400 images and all attention heads, and Equation (4-15) quantifies value-only approximation error.

## 6.3 Results: Unit Learning Images and Independent Samples

In the original frozen sample, ViT-B/16 has late-minus-early changes of 0.2201 in entropy, −0.3558 in locality, and 0.1312 in normalized distance; the corresponding DeiT-S/16 values are 0.2250, −0.3424, and 0.1219. All 100 images have the same direction for each of the three contrasts in each model.

All 30 model-sample-metric combinations in the five independent samples reproduce increasing entropy, decreasing locality, and increasing distance. Table 6-1 gives the image-level median differences.

**Table 6-1 Late-minus-early differences in five independent token-channel samples**

| Model | Metric | Group 1 | Group 2 | Group 3 | Group 4 | Group 5 |
|---|---|---:|---:|---:|---:|---:|
| ViT-B/16 | Entropy | 0.2351 | 0.2273 | 0.2179 | 0.2313 | 0.2247 |
| ViT-B/16 | Locality | −0.3685 | −0.3646 | −0.3537 | −0.3634 | −0.3618 |
| ViT-B/16 | Distance | 0.1500 | 0.1473 | 0.1432 | 0.1445 | 0.1450 |
| DeiT-S/16 | Entropy | 0.2246 | 0.2301 | 0.2244 | 0.2271 | 0.2189 |
| DeiT-S/16 | Locality | −0.3408 | −0.3453 | −0.3357 | −0.3413 | −0.3401 |
| DeiT-S/16 | Distance | 0.1311 | 0.1344 | 0.1294 | 0.1347 | 0.1320 |

Across-group standard deviations for entropy, locality, and distance are 0.00653, 0.00546, and 0.00270 in ViT-B/16 and 0.00409, 0.00342, and 0.00225 in DeiT-S/16. The evidence advances the conclusion from one target sample to five independent target samples in both fixed parameter states.

Between-group variation in Table 6-1 is small relative to the median depth contrast for each metric. For example, the five ViT-B/16 entropy differences range from 0.2179 to 0.2351 with a between-group standard deviation of 0.00653. DeiT-S/16 locality differences range from −0.3453 to −0.3357 with a standard deviation of 0.00342. Stability is visible in both direction and magnitude rather than being defined only by whether an interval crosses zero.

Entropy, locality, and distance describe complementary aspects of one spatial change. Higher entropy means that absolute mass covers the patch grid more evenly; lower locality means that less mass remains in the target patch's neighborhood; greater distance places more mass farther from the target location. Their joint direction excludes two single-metric explanations: redistribution only within the original neighborhood and a few distant hotspots while most mass remains highly local.

![Figure 6-1 Five independent samples](../../publication_r2_20260920/evidence/revision_v3_step2/independent_sampling.png)

**Figure 6-1 Five independent samples.** Each point follows within-image aggregation; each group uses an independently frozen token-channel sample.

## 6.4 Results: Query, Key, and Value Paths

**Table 6-2 Image-level median block-11-minus-block-0 Q/K/V differences and 95% intervals**

| Model | Path | Absolute mass | Entropy | Pair distance | Cancellation | Relative scale |
|---|---|---:|---:|---:|---:|---:|
| ViT-B/16 | Query | 208.8022 [193.7022, 229.1715] | 0.0268 [0.0217, 0.0314] | 0.0108 [0.0071, 0.0143] | 0.0099 [0.0083, 0.0122] | 0.0427 [0.0220, 0.0549] |
| ViT-B/16 | Key | 308.5203 [285.6579, 347.7998] | −0.0586 [−0.0636, −0.0528] | −0.0109 [−0.0150, −0.0073] | 0.0104 [0.0093, 0.0123] | 0.2553 [0.2449, 0.2675] |
| ViT-B/16 | Value | 255.5471 [235.9237, 295.1770] | −0.0530 [−0.0591, −0.0471] | −0.0082 [−0.0106, −0.0052] | 0.0018 [0.0015, 0.0023] | −0.3088 [−0.3249, −0.2882] |
| DeiT-S/16 | Query | 91.9272 [86.3993, 98.2857] | −0.0038 [−0.0080, 0.0005] | −0.0021 [−0.0046, 0.0022] | 0.0192 [0.0177, 0.0223] | 0.1292 [0.1169, 0.1376] |
| DeiT-S/16 | Key | 140.0030 [128.9647, 148.3989] | −0.0669 [−0.0736, −0.0590] | −0.0100 [−0.0140, −0.0059] | 0.0119 [0.0106, 0.0144] | 0.2581 [0.2507, 0.2649] |
| DeiT-S/16 | Value | 73.3232 [63.7414, 84.2214] | −0.0669 [−0.0748, −0.0609] | −0.0054 [−0.0094, −0.0021] | 0.0027 [0.0021, 0.0032] | −0.3915 [−0.4044, −0.3840] |

Key and value paths increase in absolute mass and cancellation and decrease in entropy and pairwise distance in both models. Key gains relative scale while value loses relative scale. Query also gains mass, cancellation, and relative scale in both models, but its spatial direction differs: entropy and distance increase in ViT-B/16, whereas the small DeiT-S/16 estimates decrease with intervals spanning zero. The three paths cannot be summarized by one common expansion or contraction.

Joint reporting of absolute mass and relative scale prevents a ratio-only interpretation. Value absolute mass increases in both models while its relative scale decreases because query- and key-induced routing grows faster. Key gains both absolute mass and relative scale, indicating an increasing share of the full local attention differential. The model-dependent spatial direction for query suggests that architecture, parameter state, or normalization can affect its spatial organization.

Increasing cancellation means that total magnitude before addition grows faster than the signed net magnitude. A visually bright absolute-mass map therefore cannot be read directly as output support in one direction. Absolute mass answers where the distribution lies; signed sums and finite replacements answer direction and effect. The results keep these questions separate.

![Figure 6-2 Population Q/K/V trajectories](../../publication_r2_20260920/evidence/revision_v3_step2/qkv_population.png)

**Figure 6-2 Population Q/K/V trajectories.** Each model uses 400 images and all 12 blocks; bands are image-level 95% bootstrap intervals.

## 6.5 Results: Routing, Value, and Residual Paths

Median routing relative scale increases from 0.3092 in block 0 to 0.5154 in block 11 for ViT-B/16 and from 0.2041 to 0.4808 for DeiT-S/16. Value-only error increases from 0.4407 to 0.8022 in ViT-B/16 and from 0.2774 to 0.8488 in DeiT-S/16.

**Table 6-3 Median absolute mass of attention paths**

| Model | Block | Routing mass | Value mass | Full-attention mass |
|---|---:|---:|---:|---:|
| ViT-B/16 | 0 | 2.6596 | 7.6041 | 9.8604 |
| ViT-B/16 | 11 | 40.7842 | 39.0241 | 54.6768 |
| DeiT-S/16 | 0 | 4.5294 | 18.4261 | 20.1427 |
| DeiT-S/16 | 11 | 35.9257 | 38.8767 | 44.5653 |

Deep-block analytic-direction discrepancies are 0.9516 and 0.9636; the corresponding full-response norms are 14.9711 and 1.5155, with absolute discrepancies of 13.8064 and 1.4228. Ratios and absolute values jointly show that $A\mathrm dV$ under fixed attention weights is a poor approximation of the full deep attention response.

Residual paths differ. Late-minus-early attention-residual identity scale is −0.1242 for ViT-B/16 and −0.0449 for DeiT-S/16, while multilayer-perceptron residual values are 0.1798 and 0.1780. Spatial total variation between output and identity increases by 0.0520 and 0.0282 for attention but changes by −0.0187 and −0.0462 for the multilayer perceptron. These four values come from a separately corrected post-hoc spatial analysis.

The opposite attention and multilayer-perceptron directions do not reduce to a single statement that deeper blocks depend more on branches. Decreasing identity scale in attention is consistent with growing routing terms, whereas increasing identity scale in the multilayer perceptron indicates a different balance between skip and branch paths. Spatial total variation adds another dimension: a scale change and a spatial-distribution change may coincide or separate.

Value-only error rises from approximately 0.28–0.44 to 0.80–0.85, making attention-weight-change terms material to the target-conditioned local response in deep blocks. This result concerns the differential response and leaves the forward role of value vectors intact. It shows that freezing attention weights yields a poorer approximation of full input sensitivity in deeper blocks.

## 6.6 Supplementary Results: Parameter State, Target, and Forward Features

The three randomly initialized states also show spatial broadening, at different magnitudes. ViT-B/16 pretrained entropy late-minus-early is 0.2288 versus a random-state mean of 0.0940, a difference of 0.1348 [0.1232, 0.1466]. DeiT-S/16 values are 0.2256, 0.1718, and 0.0539 [0.0473, 0.0602]. Several attention-routing and residual directions also differ by parameter state. This comparison establishes parameter-state differences and does not identify a causal effect of training.

Across 100 images per model and blocks 0, 5, and 11, ground-truth, clean-top1, and top1-margin targets preserve increasing routing scale and value-only error, decreasing attention identity scale, and increasing multilayer-perceptron identity scale. The incorrect subsets contain only 8 ViT-B/16 and 14 DeiT-S/16 images and do not support stable population conclusions for errors.

Image-level mean Spearman correlation between forward channel energy and same-channel local-input absolute mass is 0.2663 for ViT-B/16 and 0.2811 for DeiT-S/16. Both exceed wrong-channel, same-channel cross-layer, spatial-permutation, and random-channel controls. The quantities are associated but remain distinct objects.

A correlation near 0.27 means that forward-response position explains only part of the local-input-dependence ranking. Forward energy is controlled by current activation magnitude; the local input map also includes the target's downstream gradient and the patch-projection Jacobian to pixels. Correct pairs exceeding four controls show that channel identity is informative, while the distance from unit correlation shows that a forward feature map alone does not recover target-conditioned input dependence.

## 6.7 Discussion

Five independent samples make spatial broadening independent of any single token-channel sample, while retaining scope to the two fixed parameter states and prescribed target distribution. Population Q/K/V statistics further show that one path cannot summarize full-attention depth changes. Greater spatial extent is not semantic improvement: entropy, locality, and distance describe only the distribution of absolute local mass over the input grid.

The main RQ3 result is consequently not that one deep heatmap appears more complete. Multiple image-level estimands give the same direction across independently sampled internal targets. At the same time, the cross-model query exception, parameter-state differences, and modest feature–local-map correlation define the range of inference. This combination of a stable main pattern and named exceptions supplies stronger model-internal evidence than isolated examples.

## 6.8 Chapter Summary

RQ3 receives three layers of evidence. Sampled internal targets consistently depend on broader and more distant regions across five independent samples; query, key, and value have distinct population trajectories; and deeper attention has higher routing relative scale and value-only error, with different trends for attention and multilayer-perceptron residuals. Parameter-state controls support difference statements, not causal training language.

# Chapter 7 Local Approximation and Ranking under Finite Interventions

## 7.1 Chapter Question

This chapter answers RQ4: how first-order error changes with attention-head-output intervention amplitude and how clean local maps and reference-aware Taylor estimates rank components and regions under complete replacement.

## 7.2 Method

The amplitude study follows Equations (5-1) and (5-2), using the same 400 images per model, all heads in all 12 blocks, ground-truth logits, and both references. Spearman correlation is computed across heads within each image and block and then averaged over 12 blocks. Absolute error is $|e_\alpha-p_\alpha|$; relative error follows the frozen denominator rule and propagates zero denominators as missing.

Complete component replacement follows Equation (5-3). Local-map ranking uses only the clean state; Taylor ranking explicitly uses the reference displacement. The regional task compares the local map, total Gradient × Input, raw gradient, regional Taylor, and 20 random rankings. The methods have different information budgets, so the comparison concerns fit to the prescribed task.

## 7.3 Results: Multi-Amplitude First-Order Boundary

**Table 7-1 ViT-B/16 multi-amplitude head-output interpolation**

| Reference | Amplitude | Absolute actual effect | Absolute prediction | Absolute error | Relative error | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Normalized zero | 0.05 | 0.002922 | 0.002903 | 0.000069 | 0.0272 | 0.9564 |
| Normalized zero | 0.10 | 0.005795 | 0.005807 | 0.000266 | 0.0524 | 0.9205 |
| Normalized zero | 0.25 | 0.014466 | 0.014517 | 0.001638 | 0.1216 | 0.8531 |
| Normalized zero | 0.50 | 0.028453 | 0.029035 | 0.006064 | 0.2231 | 0.7794 |
| Normalized zero | 1.00 | 0.056748 | 0.058069 | 0.021058 | 0.3913 | 0.6742 |
| Blur | 0.05 | 0.002471 | 0.002479 | 0.000057 | 0.0262 | 0.9639 |
| Blur | 0.10 | 0.004912 | 0.004957 | 0.000224 | 0.0514 | 0.9289 |
| Blur | 0.25 | 0.012021 | 0.012393 | 0.001341 | 0.1212 | 0.8633 |
| Blur | 0.50 | 0.023885 | 0.024786 | 0.005077 | 0.2219 | 0.7922 |
| Blur | 1.00 | 0.046413 | 0.049572 | 0.017925 | 0.3883 | 0.6856 |

**Table 7-2 DeiT-S/16 multi-amplitude head-output interpolation**

| Reference | Amplitude | Absolute actual effect | Absolute prediction | Absolute error | Relative error | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Normalized zero | 0.05 | 0.004951 | 0.005035 | 0.000105 | 0.0259 | 0.9762 |
| Normalized zero | 0.10 | 0.010104 | 0.010071 | 0.000415 | 0.0511 | 0.9524 |
| Normalized zero | 0.25 | 0.025327 | 0.025177 | 0.002521 | 0.1227 | 0.8810 |
| Normalized zero | 0.50 | 0.052060 | 0.050354 | 0.010036 | 0.2342 | 0.7810 |
| Normalized zero | 1.00 | 0.105217 | 0.100707 | 0.037632 | 0.4345 | 0.6095 |
| Blur | 0.05 | 0.004384 | 0.004419 | 0.000093 | 0.0241 | 0.9810 |
| Blur | 0.10 | 0.008877 | 0.008838 | 0.000364 | 0.0477 | 0.9571 |
| Blur | 0.25 | 0.021907 | 0.022095 | 0.002371 | 0.1168 | 0.8905 |
| Blur | 0.50 | 0.044622 | 0.044191 | 0.008908 | 0.2244 | 0.7952 |
| Blur | 1.00 | 0.091581 | 0.088381 | 0.033841 | 0.4186 | 0.6238 |

Cohort-median absolute error increases monotonically in all four model-reference combinations, and all 1,600 image-reference curves are individually monotone. Ranking correlation decreases with amplitude. On this prescribed interpolation, relative-error medians are 0.0241–0.0272 at $\alpha=0.05$ and 0.3883–0.4345 at complete replacement; corresponding head-ranking correlations decline from 0.9564–0.9810 to 0.6095–0.6856.

The amplitude pattern has both absolute and relative meaning. Absolute error grows with actual-effect scale, so the deviation is not fixed numerical noise. Relative error rises from approximately 2.4%–2.7% to 38.8%–43.5%, showing that error also grows relative to the intervention effect. Agreement across all four model-reference combinations reduces the possibility that one input origin or blur operator alone drives the pattern.

Ranking correlation decays more slowly than numerical accuracy. Even when relative error is near forty percent under complete replacement, head-ranking Spearman remains 0.61–0.69. This distinction matters in use: a local first-order quantity is unsuitable for precise magnitude prediction at large replacement but retains screening information. Section 7.4 therefore evaluates which head is selected through capture and regret rather than relying only on pointwise error.

![Figure 7-1 Multi-amplitude finite intervention](../../publication_r2_20260920/evidence/revision_v3_step2/multi_amplitude.png)

**Figure 7-1 Multi-amplitude finite intervention.** Absolute and relative errors increase with amplitude while head-ranking correlation decreases.

## 7.4 Results: Complete Component Ranking

Eleven of the 12 model-reference-component-family combinations meet the frozen composite criterion in the original confirmatory replacement analysis; the joint all-family condition is not met. ViT-B/16 normalized-zero routing/value correlation is 0.0996 [0.0748, 0.1226] and misses the 0.10 point-estimate threshold. The other 11 point estimates range from 0.1065 to 0.3063. The result is a broad but non-universal association of small-to-moderate magnitude.

In the supplementary within-layer head comparison, Taylor Spearman values of 0.6095–0.6856 exceed local-map values of 0.1476–0.2815. Taylor top-1 capture is 0.8028–0.8426 versus 0.5075–0.5609 for the local map; Taylor top-1 regret is 0.0346–0.0860 versus 0.1163–0.2381. Explicit reference displacement improves head ranking for the prescribed complete replacement.

The three metrics correspond to different uses. Spearman evaluates the full ordering of heads, top-1 capture measures how much of the best available effect is obtained when only one replacement can be run, and regret measures the gap between the selected and candidate-best head. Taylor improves all three, so the result is not induced by one correlation definition. Zero capture differences in some cross-layer residual tasks also show that improvement varies with candidate structure and metric.

The 11-of-12 original family result does not imply high predictive accuracy. Median correlations range only from 0.0996 to 0.3063 and primarily support weak-to-moderate ranking association. ViT-B/16 normalized-zero routing/value supplies a named exception. The text therefore reports family counts, point estimates, intervals, and the joint decision together.

## 7.5 Results: Regional Selection

**Table 7-3 Absolute-effect area under the curve for regional replacement**

| Model | Reference | Local map | Regional Taylor | Random | Candidate oracle |
|---|---|---:|---:|---:|---:|
| ViT-B/16 | Normalized zero | 0.0309 | 0.0558 | 0.0339 | 0.0654 |
| ViT-B/16 | Blur | 0.0276 | 0.0518 | 0.0307 | 0.0747 |
| DeiT-S/16 | Normalized zero | 0.0617 | 0.0961 | 0.0611 | 0.1474 |
| DeiT-S/16 | Blur | 0.0395 | 0.0770 | 0.0411 | 0.1063 |

Paired median local-map-minus-regional-Taylor differences are −0.0234, −0.0237, −0.0299, and −0.0356. Local-map-minus-random differences are −0.0023, −0.0031, 0.0009, and −0.0010, showing no stable advantage. Normalized regret relative to the candidate oracle is 0.5303–0.6366 for the local map and 0.1442–0.3352 for regional Taylor. None of the four regional families meets the original composite requirement that the local map exceed every baseline by 0.05.

Absolute AUC is small relative to the original target-logit scale: 0.0031–0.0072 for the local map and 0.0059–0.0107 for regional Taylor. Regional replacement within the prescribed budget therefore changes only part of the output. The candidate oracle is also limited to the $4\times4$ candidate set and 25% maximum area. Normalized regret compares each method with the reachable upper bound in the same candidate set and is therefore more suitable than raw AUC alone for comparing selection efficiency across models.

The corrected random baseline takes absolute isolated effect and computes AUC within each random repeat before averaging 20 repeats. This order avoids cancellation caused by averaging signed curves first. After correction, local-map-minus-random differences are near zero, showing that an earlier apparent random advantage was mainly an estimator-order artifact rather than stable regional selection ability.

The cross-layer all-head supplement covers blocks 2, 5, and 10. Eleven of 12 full-input-replacement families and all 12 isolated-head families significantly favor regional Taylor. This supplementary result does not alter the original confirmatory decision for the three-head block-5 study.

![Figure 7-2 Regional intervention boundary](../../publication_r2_20260920/figures/png/fig_21_intervention_boundary.png)

**Figure 7-2 Regional intervention boundary.** The left panel shows regional replacement curves for one image; the right panel shows absolute-effect AUC distributions across 400 images.

## 7.6 Discussion

The amplitude and complete-replacement studies answer adjacent but distinct questions. The first fixes a head-output displacement direction and varies distance, establishing an empirical increase in error. The second compares rankings of components or regions under complete replacement. High ranking correlation at a small amplitude does not entail itemwise numerical accuracy at complete replacement.

In a Taylor expansion, Equation (5-2) omits higher-order terms generated as the gradient changes along the displacement. The amplitude study directly measures the empirical growth of this remainder with distance. Complete replacement additionally combines heterogeneous displacement scales and directions across components. The former is therefore a local-approximation diagnostic, whereas the latter evaluates screening under a resource constraint.

The local map uses only the clean input and local derivative, whereas regional Taylor also uses the actual reference displacement. The latter has information better matched to a reference-specified replacement task. The local map remains useful for describing pointwise input structure, while its original regional-utility composite claim is unsupported.

The result separates two use cases. When the question is which input positions and actual edges the current target locally depends on, a reference-free local map provides a common structural representation. When the question is how much a variable will change after replacement by a specified reference, a displacement-aware Taylor quantity matches the estimand more closely. The difference concerns the estimated object rather than an unconditional method ranking.

## 7.7 Chapter Summary

RQ4 has three parts. First-order error increases monotonically with amplitude and ranking correlation decreases on the prescribed interpolation. Local path maps have broad but non-universal small-to-moderate associations with complete component effects. For prescribed regional replacement, reference-aware Taylor estimates outperform clean-only local maps.

# Chapter 8 General Discussion and Conclusion

## 8.1 From Representation to Contribution to Intervention

The evidence chain has three levels. Equation (3-2) specifies representation: a scalar NNS learning image is pointwise Gradient × Input. Equations (4-4), (4-6), and (4-8) specify contribution: a local differential can be grouped along query, key, value, attention-head, and residual edges on specified valid cuts. Equations (5-1)–(5-3) specify intervention: local first-order quantities are more accurate for small displacements, and their ranking utility under complete replacement depends on the object, reference, and metric.

These levels answer different questions. Formula identity does not make path components unique; structural closure does not establish finite-replacement utility; and superiority on one replacement task does not establish general explanatory superiority. Their separation allows positive findings and boundary findings to coexist in the conclusion.

## 8.2 Theoretical Implications

The Jacobian interpretation removes an apparent difference between layerwise NNS propagation and automatic differentiation and permits heterogeneous ViT targets to share input coordinates. A valid cut turns a visual notion of “path” into a checkable condition: each relevant differential path crosses exactly once. Negative controls show that omitting an identity bypass or adding a branch twice breaks closure.

The theoretical framework is compositional. Any internal target that can be written as a scalar on the current graph receives an input-coordinate representation through Equation (3-1). Any actual-edge set that forms a valid cut receives an additive relation through Equation (4-8). Adding a target or cut does not require changing the base propagation rule; it requires an explicit scalarization, edge set, and closure check.

The non-uniqueness counterexample supplies the corresponding boundary. Structural paths are not canonical decompositions determined solely by the end-to-end function; they are relative to an explicitly specified implementation graph, cut, and internal coordinate system. Their scientific use is to analyze actual edges in a concrete implementation.

Path names should therefore be read as operational coordinates in a model implementation. Query, key, value, and residual edges are clear, intervenable, and recomputable objects in the frozen implementation. Cross-architecture comparison would first require a common cut or aggregation to a coarser invariant; names of internal coordinates alone do not provide that alignment.

## 8.3 Empirical Implications

Five independent samples show that sampled internal targets depend on broader and more distant input regions with depth in both fixed pretrained states. Query, key, and value do not share one spatial trajectory: key and value become more concentrated in both models, whereas query direction differs by model. Increasing routing relative scale and value-only error show that content transport under fixed attention weights does not represent the full local response of deep attention.

Finite intervention gives a more direct boundary. At small head-output amplitudes, first-order estimates have low error and high ranking correlation; error rises monotonically with amplitude. Under complete regional replacement, a Taylor estimate containing the actual reference displacement clearly exceeds the clean local map. Local differentials describe structure around the current point, whereas reference-aware displacement estimates fit a specified finite replacement.

Method evaluation must consequently report both what was computed correctly and on which task it is useful. Structural closure is strong, whereas the regional selection result is negative; the two findings are compatible. The former shows that components faithfully reconstruct a local differential, and the latter shows that this local object lacks sufficient information for the prescribed finite regional replacement. Retaining the negative result in the main conclusion prevents mathematical correctness from being extrapolated into intervention utility.

## 8.4 Limitations

The model scope contains one fixed pretrained state each for ViT-B/16 and DeiT-S/16. Three random initializations support parameter-state differences but do not identify a training trajectory or consistency across multiple trained states. Data come from deterministically selected ImageNet validation images, prioritizing class coverage over within-class variation; other tasks, architectures, and data distributions remain outside the evidence.

Internal-unit conclusions cover five frozen token-channel samples rather than every unit in either model. Incorrect subsets contain only 8 and 14 images and do not support stable population conclusions for errors. Pixel-scale structure may also depend on the $16\times16$ patch projection and normalization origin; within-patch permutation and uniformization controls were not executed.

Finite interventions cover two references, prescribed attention-head-output and residual variables, three principal blocks, and at most 25% regional area. A hybrid internal state produced by replacement need not correspond to the full forward state of a natural image. The amplitude study establishes an empirical first-order boundary along the prescribed interpolation directions and does not automatically generalize to arbitrary perturbation directions.

## 8.5 Conclusion

This thesis reaches four conclusions.

First, with exact current-point local derivatives and no additional redistribution, a scalar NNS learning image equals Gradient × Input. The identity specifies the base quantity and its coordinate-dependent, non-function-complete boundary.

Second, on a specified ViT computation graph and valid cuts, an attention local differential separates into query, key, and value paths, and a residual local differential separates into identity and branch paths. Their sums recover the full local input map, while components depend on the graph, cut, and parameterization.

Third, five independent samples reproduce spatial broadening of internal targets in two fixed pretrained parameter states. Query, key, and value have distinct population trajectories; deeper attention has higher routing relative scale and value-only error; and attention and multilayer-perceptron residuals change differently.

Fourth, first-order error increases monotonically with amplitude in prescribed head-output interpolation. Local path maps have broad but non-universal associations with complete component effects, while reference-specified regional replacement is better served by Taylor estimates containing the actual displacement.

NNS is thereby extended into a unified implementation and evaluation framework for actual ViT computation edges. Its added value lies in graph-relative structural decomposition, verifiable implementation, and empirically bounded conclusions.

## 8.6 Chapter Summary

This chapter synthesized the theoretical and empirical findings along the representation → contribution → intervention chain and consolidated the limitations on models, sampling, pixel structure, and replacement tasks. The abstract, main text, and conclusion use the same claim strength.

# References

[1] Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16×16 words: Transformers for image recognition at scale[C]//International Conference on Learning Representations. 2021.

[2] Touvron H, Cord M, Douze M, et al. Training data-efficient image transformers and distillation through attention[C]//Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021, 139: 10347-10357.

[3] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. 2017, 30.

[4] Dou H, Shen F, Zhao J, Mu X. Understanding neural network through neuron level visualization[J]. Neural Networks, 2023, 168: 484-495.

[5] Simonyan K, Vedaldi A, Zisserman A. Deep inside convolutional networks: Visualising image classification models and saliency maps[C]//ICLR Workshop. 2014.

[6] Shrikumar A, Greenside P, Kundaje A. Learning important features through propagating activation differences[C]//Proceedings of the 34th International Conference on Machine Learning. PMLR, 2017, 70: 3145-3153.

[7] Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks[C]//Proceedings of the 34th International Conference on Machine Learning. PMLR, 2017, 70: 3319-3328.

[8] Dhamdhere K, Sundararajan M, Yan Q. How important is a neuron?[EB/OL]. arXiv:1805.12233, 2018.

[9] Srinivas S, Fleuret F. Full-gradient representation for neural network visualization[C]//Advances in Neural Information Processing Systems. 2019, 32.

[10] Bach S, Binder A, Montavon G, et al. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation[J]. PLoS ONE, 2015, 10(7): e0130140.

[11] Ancona M, Ceolini E, Öztireli C, Gross M. Towards better understanding of gradient-based attribution methods for deep neural networks[C]//International Conference on Learning Representations. 2018.

[12] Smilkov D, Thorat N, Kim B, et al. SmoothGrad: Removing noise by adding noise[EB/OL]. arXiv:1706.03825, 2017.

[13] Selvaraju R R, Cogswell M, Das A, et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 618-626.

[14] Zeiler M D, Fergus R. Visualizing and understanding convolutional networks[C]//European Conference on Computer Vision. Springer, 2014: 818-833.

[15] Ribeiro M T, Singh S, Guestrin C. Why should I trust you? Explaining the predictions of any classifier[C]//Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016: 1135-1144.

[16] Lundberg S M, Lee S-I. A unified approach to interpreting model predictions[C]//Advances in Neural Information Processing Systems. 2017, 30.

[17] Adebayo J, Gilmer J, Muelly M, et al. Sanity checks for saliency maps[C]//Advances in Neural Information Processing Systems. 2018, 31: 9505-9515.

[18] Hooker S, Erhan D, Kindermans P-J, Kim B. A benchmark for interpretability methods in deep neural networks[C]//Advances in Neural Information Processing Systems. 2019, 32.

[19] Sixt L, Granz M, Landgraf T. When explanations lie: Why many modified BP attributions fail[C]//Proceedings of the 37th International Conference on Machine Learning. PMLR, 2020, 119: 9046-9057.

[20] Montavon G, Samek W, Müller K-R. Methods for interpreting and understanding deep neural networks[J]. Digital Signal Processing, 2018, 73: 1-15.

[21] Abnar S, Zuidema W. Quantifying attention flow in transformers[C]//Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020: 4190-4197.

[22] Chefer H, Gur S, Wolf L. Transformer interpretability beyond attention visualization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 782-791.

[23] Jain S, Wallace B C. Attention is not explanation[C]//Proceedings of NAACL-HLT. 2019: 3543-3556.

[24] Serrano S, Smith N A. Is attention interpretable?[C]//Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019: 2931-2951.

[25] Kobayashi G, Kuribayashi T, Yokoi S, Inui K. Attention is not only a weight: Analyzing transformers with vector norms[C]//Proceedings of EMNLP. 2020: 7057-7075.

[26] Modarressi A, Fayyaz M, Yaghoobzadeh Y, Pilehvar M T. GlobEnc: Quantifying global token attribution by incorporating the whole encoder layer in Transformers[C]//Proceedings of NAACL-HLT. 2022: 258-271.

[27] Ferrando J, Gállego G I, Tsiamas I, Costa-jussà M R. Measuring the mixing of contextual information in the Transformer[C]//Proceedings of EMNLP. 2022: 8698-8714.

[28] Modarressi A, Fayyaz M, Aghazadeh E, et al. DecompX: Explaining Transformers decisions by propagating token decomposition[C]//Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics. 2023: 2649-2664.

[29] Achtibat R, Hatefi S M V, Dreyer M, et al. AttnLRP: Attention-aware layer-wise relevance propagation for Transformers[C]//Proceedings of the 41st International Conference on Machine Learning. PMLR, 2024, 235: 135-168.

[30] Mehri F, Soleymani Baghshah M, Pilehvar M T. LibraGrad: Balancing gradient flow for universally better Vision Transformer attributions[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2025: 67-78.

[31] Zhang F, Nanda N. Towards best practices of activation patching in language models: Metrics and methods[EB/OL]. arXiv:2309.16042, 2023.

[32] Syed A, Rager C, Conmy A. Attribution patching outperforms automated circuit discovery[EB/OL]. arXiv:2310.10348, 2023.

[33] Kramár J, Lieberum T, Shah R, Nanda N. AtP*: An efficient and scalable method for localizing LLM behaviour to components[EB/OL]. arXiv:2403.00745, 2024.

[34] Meng K, Bau D, Andonian A, Belinkov Y. Locating and editing factual associations in GPT[C]//Advances in Neural Information Processing Systems. 2022, 35: 17359-17372.

[35] Geva M, Schuster R, Berant J, Levy O. Transformer feed-forward layers are key-value memories[C]//Proceedings of EMNLP. 2021: 5484-5495.

[36] Efron B, Tibshirani R J. An Introduction to the Bootstrap[M]. New York: Chapman & Hall/CRC, 1993.

[37] Holm S. A simple sequentially rejective multiple test procedure[J]. Scandinavian Journal of Statistics, 1979, 6(2): 65-70.
