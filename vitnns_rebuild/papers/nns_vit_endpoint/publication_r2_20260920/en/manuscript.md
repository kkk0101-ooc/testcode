# From NNS to Vision Transformers: Auditable Jacobian Scanning and Structural Decomposition of Internal Targets

- Authors: [AUTHOR REQUIRED]
- Affiliations: [AFFILIATION REQUIRED]
- Corresponding author: [CORRESPONDING AUTHOR REQUIRED]

## Abstract

Neural Network Scanner (NNS) propagates local scales layer by layer to construct position-preserving learning images for internal network targets. When every layer uses the exact local derivative at the observed forward state without normalization, truncation, or relevance redistribution, a scalar NNS learning image equals $x\odot\nabla_xs(x)$, i.e. same-target Gradient × Input. We turn Gradient × Input into a structure-aware and auditable scanner for heterogeneous internal targets in Vision Transformers, enabling exact differential accounting and systematic analysis of how local input dependence is reorganized across depth, attention paths, and residual branches. The framework registers heterogeneous internal targets, defines valid differential cuts on actual computational edges, decomposes the attention differential into Q-induced, K-induced, and V-induced paths, decomposes residual differentials into identity and branch paths, and tests additive closure at each cut. Its terms are defined relative to a specified graph, cut, and parameterization. Experiments on two fixed ImageNet-pretrained checkpoints, ViT-B/16 and DeiT-S/16, cover numerical correctness, path-depth statistics, differences between trained and random parameter states, and ranking performance in prescribed finite-replacement tasks. Frozen target samples depend on broader and more distant patches at depth; routing-relative norm and Value-only approximation error increase in deep attention, while attention and MLP residuals exhibit different depth trajectories. Local path maps have broad but non-universal association with finite component effects; in reference-specified regional replacement, displacement-aware Taylor ranking outperforms clean-only Endpoint.

Keywords: Vision Transformer; Neural Network Scanner; learning image; local Jacobian; structural path decomposition

## 1. Introduction

Vision Transformers (ViTs) divide images into patch tokens and build representations through self-attention and feed-forward layers (Dosovitskiy et al., 2021). This architecture models relationships across positions while making internal computation difficult to interpret. Token-channel units represent responses at particular positions, attention heads transmit information across positions, and residual connections combine inputs with branch outputs. Understanding their relationships calls for a representation that preserves input locations and applies to different internal objects.

Existing visualizations commonly show feature maps, attention weights, or class-related regions. Each view serves a purpose, but their resolutions, targets, and propagation rules differ. A forward feature map shows a channel's response on the token grid, whereas attention weights describe allocation between tokens. Neither is the same object as an internal scalar's dependence on the original image. Mapping internal targets to common input coordinates provides a shared basis for comparing modules.

Neural Network Scanner (NNS) associates a neuron with a learning image that preserves input positions (Dou et al., 2023). With exact local derivatives, its layerwise scale propagation is an application of the Jacobian chain rule; for a scalar target, the endpoint learning image is Input × Gradient. This identity fixes the local measurement primitive. A total input gradient does not distinguish whether a local response travels through Q, K, V, individual heads, or the identity and branch edges of a residual addition. We therefore build a structure-aware, closable, and auditable scanner for heterogeneous internal ViT targets, decompose complete local input dependence over actual graph edges, and study how that dependence is reorganized across depth and structural paths.

![Figure 1](../figures/png/fig_01_concept_overview.png)

**Figure 1.** The same input produces a prediction through the ordinary forward computation. NNS-ViT associates internal and output targets with learning images in the original input coordinates. Dashed lines link targets to their images.

The overarching question is how NNS can represent the local input dependence of an internal ViT target as computable, decomposable, and verifiable structural-path maps, and how these maps vary with depth and rank candidates in prescribed finite-replacement tasks. We address four questions:

1. **RQ1: What is the mathematical object represented by an NNS learning image?** We relate layerwise scale propagation to the Jacobian chain rule and Input × Gradient, including the effects of non-homogeneous operations, fixed tokens, and the input-coordinate origin.
2. **RQ2: How can NNS be extended to a structured ViT computational graph?** We define internal scalar targets, actual graph edges, complete computational cuts, and additive local differentials for attention, multi-head structure, and residual paths.
3. **RQ3: How do structural paths vary with depth?** In frozen target samples from two fixed pretrained checkpoints, we analyze spatial broadening, routing/value, and identity/branch trends and compare trained and random parameter states together with target and sampling controls.
4. **RQ4: Can local path maps rank components and spatial regions in the prescribed finite-replacement tasks?** With model, sample, layer, head, reference, replacement operator, and region budget fixed, we compare clean-only local path maps with reference-aware displacement estimates for component ranking and regional selection.

We make three contributions. First, **mathematical identity and object definition**: we establish the conditions under which original NNS learning-image propagation equals same-target Gradient × Input and distinguish Endpoint from Path IG, model constants, and reference-specific finite replacement; the 5,000 formula-regression cases and 88 operator-direction checks in Table 3 validate this object definition. Second, **a structured scanning and audit framework**: we define a registry for ViT token-channel units, channel aggregates, attention heads, residual edges, and class targets, introduce valid differential cuts on actual computational edges, and account additively for Q/K/V, multi-head, and identity/branch local differentials; direct evidence consists of the 457/385 registered targets in Table 2, 76,800 positive closure checks, 28,800 omission/duplication controls, and 6,144 analytic JVP checks. Third, **mechanistic observations and an interpretation boundary**: across frozen samples from two fixed ViT checkpoints, we measure depth-, attention-, and residual-dependent reorganization of local input dependence and use prescribed finite-replacement experiments to distinguish observed-input local representations from reference-specific intervention estimators; Figures 6–11 and Tables 4–8 report the image-level results.

We next delimit the related methods and our positioning, derive the general Jacobian representation and ViT structural decomposition, and organize experiments around identity and closure, depth trends, structural paths, and finite intervention.

## 2. Related Work

### 2.1 NNS and learning images

Neuron visualization examines internal responses through input patterns or observed samples. NNS organizes a neuron's computation as an input-coordinate learning image and represents convolutional channels through activation-weighted aggregation (Dou et al., 2023). This organization maps internal objects with different native spatial structures into common image coordinates. We retain “learning image” as the historical term and use “local input-dependence map” in the mathematical development to emphasize a first-order response at the observed input.

### 2.2 Local gradients, path integration, FullGrad, and hidden-unit attribution

Gradient × Input is the input-weighted gradient at the observed point (Shrikumar et al., 2017). Integrated Gradients averages gradients along a reference-to-input path and satisfies its corresponding completeness axioms (Sundararajan et al., 2017), while conductance extends path attribution to hidden units (Dhamdhere et al., 2018). FullGrad combines input-gradient and layerwise bias-gradient components into a complete representation and then aggregates them into a saliency map (Srinivas and Fleuret, 2019). Endpoint uses the true current-point Jacobian without path integration and does not add bias terms to its input map. Conductance path completeness, the FullGrad function-value representation, and local-differential closure on our valid cuts are distinct objects.

DAVE decomposes a ViT input gradient into a direct effective input-output transformation and a local-variation term, then aggregates the retained transformation over an input distribution (Wróbel et al., 2026). It is a close precedent for ViT-specific gradient decomposition and spatial attribution; it does not retain a family of input-coordinate maps indexed by internal consumer uses.

### 2.3 Transformer token contribution and relevance propagation

GlobEnc incorporates attention, residual, and normalization components into layerwise token attribution and aggregates them across encoder layers (Modarressi et al., 2022). ALTI decomposes token interactions over a whole attention block, including the residual and LayerNorm, and aggregates them by rollout; its published formulation does not include the classifier (Ferrando et al., 2022). DecompX propagates token-representation components through nonlinear feed-forward layers and the classification head and recomposes representations and logits under its bias-allocation rules (Modarressi et al., 2023).

AttnLRP derives relevance rules for softmax, bilinear attention, normalization, and latent attribution (Achtibat et al., 2024). LibraGrad prunes and rescales backward paths under a FullGrad-completeness objective for ViT attribution (Mehri et al., 2025). HiLRP reduces diverse ViT attention and resolution-changing modules to four conserving primitives and verifies conservation to machine precision, while GradSkip explicitly allocates relevance between attention and residual paths (Nishankar et al., 2026; Buratti et al., 2026). Unpack produces component interactions, named end-to-end paths with K/Q/V composition labels, and token attribution without gradients (Chen et al., 2026). These works establish direct precedents for Transformer component propagation, Q/K/V-aware analysis, skip-aware allocation, and exact conservation. Our comparison therefore concerns the retained indices and closure object: the model's unmodified local derivatives, input-coordinate terms for concrete consumer uses, and additive reconstruction of the same local differential on a valid cut.

### 2.4 Activation patching and edge attribution

Activation patching replaces actual internal variables with clean/corrupted reference states and measures finite output changes; its result depends on the corruption, patch location, and metric (Zhang and Nanda, 2023). Attribution patching approximates a component patching effect with the inner product of activation displacement and a local gradient, and Edge attribution patching applies this approximation to computational edges (Syed et al., 2024). AtP* further treats approximation errors caused by attention saturation and cancellation (Kramár et al., 2024). These methods directly evaluate or approximate reference-specified finite effects. We report clean-only path maps, reference-aware Taylor quantities, and actual replacement effects separately and do not substitute structural closure for finite-intervention utility.

Edge-based analysis and internal intervention already extend to vision. Vi-CD discovers and validates sparse residual-stream edge circuits in ViTs and OpenCLIP by activation patching (Żukowska et al., 2026), while CAAP constructs ViT patch attributions by inserting source-image activations into a neutral context (Izadi et al., 2026). Actual edges and ViT activation patching are therefore precedents rather than contribution claims of this study.

### 2.5 Visual attribution and position of this work

Grad-CAM provides class-conditioned localization through gradient-weighted feature maps (Selvaraju et al., 2017), attention rollout combines attention flow across layers (Abnar and Zuidema, 2020), and Chefer et al. (2021) design relevance propagation for Transformers. These approaches, Gradient × Input, token decomposition, LRP, and finite replacement address different explanation objects. Table 1 compares the closest method families using the same nine fields. “Not found” means that the cited source does not report the capability; it is not an absolute non-existence claim. The literature cutoff for this audit is 22 September 2026.

**Table 1. Difference matrix; exact closure always refers to the method-specific object named in that cell**

| Method family | Explained object | Internal targets | Consumer edge | Valid cut | Routing/value | Identity/branch | Exact closure | Mutation detection | Empirical use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gradient × Input | Current-point input-weighted gradient | Any differentiable scalar; no registry | Not found | Not found | Not found | Not found | Generally no function-value completeness | Not found | Input saliency |
| Integrated Gradients | Baseline-to-input path attribution | Output scalar; selectable targets in software | Not found | Not found | Not found | Not found | Output difference | Not found | Input attribution |
| Conductance | Path attribution through hidden units, optionally input-resolved | Neurons or unit sets | Not found | Whole hidden layer only | Not found | Not found | Output difference across a layer | Not found | Unit analysis and ablation |
| GlobEnc | Cross-layer token attribution | Tokens and encoder components | Not found | Not found | Not separated | Residual included, not paired local maps | Not reported for class score/local differential | Not found | Token attribution |
| ALTI | Attention-block token mixing and rollout | Token interactions; classifier excluded | Not found | Not found | Not separated | Residual and LayerNorm included | Local forward decomposition, not target local differential | Not found | Context mixing and token erasure |
| DecompX | Token components of representations and logits | Encoder, FFN, classifier | Not found at consumer-use granularity | Not found | Not separated as local differentials | Residual components, not paired local maps | Representation/logit under its bias rule | Not found | Prediction token attribution |
| LRP family | Conserved relevance to input or latent variables | Inputs, tokens, latent neurons | Operator links, not registered consumer uses | Layerwise conservation, not a true-Jacobian cut | Attention primitives; no paired routing/value local maps | Skip-aware relevance exists | Relevance conservation | No structural mutant suite found | Attribution and localization |
| EAP / AtP* / Vi-CD | Reference-specific edge effects, approximations, or sparse circuits | Heads, MLPs, Q/K/V nodes and residual edges | Yes | No complete additive cut found | Q/K fixes and Q/K/V edge types, not input maps | Residual-stream edges, not paired local maps | Patching effect or circuit fidelity, not local differential | Patching checks/diagnostics, not structural mutants | Circuit discovery and steering |
| Activation patching / CAAP | Finite internal replacement effect | Layers, components, tokens or patch activations | Node-level; edge variants above | Not applicable | Intervenable, not additively decomposed | Intervenable, not additively decomposed | Not additive in general | Not found | Localization and mechanism tests |
| DAVE | Distribution-aware ViT spatial attribution | Effective input-output mapping | Not found | Not found | Not found | Not found | No structural-cut closure found | Not found | Stable ViT pixel attribution |
| Unpack | Component interactions, named paths, token credit | Attention, MLP, tokens, K/Q/V-labelled paths | Component/path identity | No input-to-target valid-cut contract found | K/Q/V labels, not true-Jacobian routing/value maps | Paths traverse residual computation | Method-specific credit decomposition | No structural mutant suite found | LLM composition and IOI analysis |
| NNS–ViT (this work) | Same current-point local differential indexed by input and edge | Registered ViT scalar targets and structural terms | Cotangent at actual use edges; multi-input/shared-tensor uses distinguished | Each live input-target path crosses once | Q+K routing and V value; Q/K/V maps | Identity plus branch maps | Same signed local differential | Omit-head, omit-routing and duplicate-identity controls | Two-model depth/path analysis and separate replacement calibration |

Every individual ingredient in the final row has a precedent: internal attribution, Transformer component decomposition, Q/K/V analysis, residual allocation, computational edges, conservation, and ViT intervention. Across the sources reviewed through the cutoff, no single method combines input coordinates with concrete consumer-use identity, valid-cut reconstruction of the same current-point local differential, attention and residual map pairs, and structural mutation tests. The contribution claim is this bounded combination, followed by depth-wise mechanistic analysis; it is not a claim to the individual ingredients or to a new base attribution rule.

## 3. Jacobian Interpretation of NNS

### 3.1 General computational-graph representation of NNS

Let the network be a directed acyclic computational graph $G=(V,E)$, with input $x$ and scalar target $s(x)$. A neuron produces a scalar value, whereas its learning image retains the input coordinates on which that scalar depends. Figure 2 relates ordinary computation, input-coordinate images, and local-derivative composition in a four-coordinate network; expanded initialization and propagation diagrams are moved to the supplementary figure bank. We denote the final scalar map by $E_s(x)$ and call it the NNS local input map.

![Figure 2](../figures/png/fig_02_neuron_learning_image.png)

**Figure 2.** Ordinary scalar computation and NNS-ViT learning-image propagation in the same four-coordinate network. Matching colors link nodes to learning images; lower matrices preserve input positions and combine with the graph coefficients.

Let the image input $x\in\mathbb R^d$ and scalar gradient $\nabla_xs\in\mathbb R^d$ be column vectors. For $u(x)\in\mathbb R^m$, the learning-image matrix is $\mathcal L(u;x)\in\mathbb R^{m\times d}$. Initialization and propagation are

$$
\mathcal L(x;x)=\operatorname{diag}(x),\qquad
\mathcal L(v;x)=\sum_aJ_{f,u_a}(x)\mathcal L(u_a;x),\quad v=f(u_1,\ldots,u_r).
$$

**Proposition 1 (scalar-target equivalence).** If all local Jacobians are evaluated at the same current forward state and propagation introduces no normalization, truncation, or relevance redistribution, then

$$
\mathcal L(u;x)=J_{u,x}(x)\operatorname{diag}(x),\qquad
\mathcal L(s;x)=\nabla_xs(x)^\top\operatorname{diag}(x),\qquad
E_s(x)=\mathcal L(s;x)^\top=\operatorname{diag}(x)\nabla_xs(x).
$$

**Proof.** At the input, $J_{x,x}=I$. In topological order, substitute each predecessor's expression into the propagation rule and apply the multi-input chain rule. Transposing the scalar node's learning-image row gives the column-vector Endpoint. For linear, convolutional, and current-ReLU-gate operations, this recovers the original NNS coefficients. MaxPool uses the current forward winner. At ReLU zero and MaxPool ties, we use the framework's fixed derivative convention; smooth regions use ordinary derivatives.

Endpoint describes the target's local first-order dependence on coordinate scaling: changing $x_i$ to $(1+\epsilon)x_i$ produces first-order response $\epsilon E_{s,i}$. In general,

$$
\sum_iE_{s,i}=x^\top\nabla_xs(x)\ne s(x).
$$

Euler's theorem gives equality for differentiable degree-one homogeneous functions. A general ViT containing biases, fixed tokens, positional embeddings, LayerNorm, softmax, and GELU does not satisfy this condition. Endpoint reconstructs local input-Jacobian decomposition rather than all sources of the target or a complete function-value decomposition. The CLS token, positional embeddings, and patch/LN/QKV/MLP/classifier biases have no separate image-coordinate terms, but they change the current function and Jacobian.

If $x=(r-\mu)/\sigma$, then

$$
x_i\frac{\partial s}{\partial x_i}=(r_i-\mu_i)\frac{\partial s}{\partial r_i}.
$$

Endpoint therefore requires no externally selected intervention reference, while depending on the model's input coordinates and normalization origin.

## 4. Structured NNS for Internal ViT Targets

### 4.1 From input patches to internal scalars

Patch embedding maps a local image block to a token vector. Linear projection changes response dimensions while the learning image remains on the input-pixel grid; the corresponding sparse-projection diagram is moved to the supplementary figure bank.

We distinguish native token-channel units, aggregate responses, structural paths, and output targets (Figure 5). For patch activations in channel $c$, define

$$
s_{\mathrm{energy}}=\frac12\sum_{t=1}^{196}h_{t,c}^2,\qquad
\nabla_xs_{\mathrm{energy}}=\sum_th_{t,c}\nabla_xh_{t,c},\qquad
E_{\mathrm{energy}}=\sum_th_{t,c}E_{h_{t,c}}.
$$

This is the Jacobian form of original NNS activation-weighted filter aggregation; energy is an aggregate scalar rather than one neuron. Targets also include patch sums, internal head coordinates, final CLS channels, ground-truth logits, and frozen clean-top1 logits. Equal channel indices across models do not imply semantic correspondence.

### 4.2 Attention, multi-head, and residual path decomposition

Figure 3 separates content transmission from weight changes, and Figure 4 relates residual identity and branch paths.

![Figure 3](../figures/png/fig_06_attention_differential.png)

**Figure 3.** Q/K changes alter routing through attention weights; V changes propagate through the current weights. The two differential paths sum to the full attention differential, with output projection included in the downstream cotangent.

![Figure 4](../figures/png/fig_07_residual_principle.png)

**Figure 4.** The upper graph shows the native identity and branch edges; matching dashed links connect them to path learning images for the same input and class target. Complete signed maps add to the residual output, while blue positive views display spatial structure.

For each head, $Q=XW_Q+b_Q$, $K=XW_K+b_K$, $V=XW_V+b_V$, and

$$
A=\operatorname{softmax}(QK^\top/\sqrt{d_h}),\quad O_h=AV,\quad
\mathrm dO_h=(\mathrm dA)V+A\mathrm dV.
$$

Here $\mathrm dA=J_{\mathrm{softmax}}(S)\mathrm dS$ and $\mathrm dS=(\mathrm dQK^\top+Q\mathrm dK^\top)/\sqrt{d_h}$. The first term therefore separates into Q-induced and K-induced routing differentials, while the second is the V-induced value differential:

$$
E^{\mathrm{attn}}=E^Q+E^K+E^V=E^{\mathrm{routing}}+E^{\mathrm{value}}.
$$

These quantities are local differential path maps for the current input and target, not Q/K/V activations or shares of the forward output value. The head edges precede concatenation and output projection: $O=\operatorname{concat}_h(O_h)W_O+b_O$. Each head's downstream cotangent includes $W_O$ and all subsequent target derivatives, yielding $E^{\mathrm{MHA}}=\sum_hE_h$ after output projection. The bias has no direct input-differential term. Q/K/V paths share upstream inputs and can overlap in input space. Path-norm ratios describe local scales, not mutually exclusive causal percentages.

For a residual $y=u+f(u)$, let $g_y=\nabla_yF$. Its two terms are

$$
E^{\mathrm{id}}=\operatorname{diag}(x)J_{u,x}^\top g_y,\qquad
E^{\mathrm{branch}}=\operatorname{diag}(x)J_{u,x}^\top J_{f,u}^\top g_y.
$$

The identity path uses $g_y$ at the residual output. Using $\nabla_uF$, which already includes the branch, would double-count paths. Their sum reconstructs Endpoint without implying a function-value decomposition $F=F_{\mathrm{id}}+F_{\mathrm{branch}}$.



### 4.3 Combination over a complete computational cut

For cut edges $\{z_k\}$ and target $F$, define

$$
E^C_{k,i}=x_i\left\langle g_{z_k},\frac{\partial z_k}{\partial x_i}\right\rangle.
$$

**Proposition 2 (closure of a valid computational cut).** If each relevant input-to-target differential path crosses the cut exactly once, all components use the same forward state, cotangents are attached to the corresponding actual use edges, and no bypass is omitted or path duplicated, then $\sum_kE^C_{k,i}=x_i\partial F/\partial x_i$. Inputs to a multi-input operator are counted separately; repeated uses of a shared tensor are distinguished by graph edge rather than tensor name.

**Proof.** Expand the chain rule into Jacobian products over input-to-target paths. A valid cut partitions these paths without omission or duplication. Each group factors into its upstream Jacobian to $z_k$ and the downstream cotangent at that edge. Summing all groups and multiplying by $x_i$ gives the result. Cutting only $f(u)$ omits the identity bypass in $y=u+f(u)$; using the complete $\nabla_uF$ as an identity term and then adding the branch duplicates the branch path. A cut internal to one branch reconstructs only that branch; a full residual or target also requires the remaining paths.

### 4.4 Computational-graph dependence and non-uniqueness

Closure in Proposition 2 is relative to a specified computational graph, cut, and internal coordinate system. An equivalent reparameterization can preserve the end-to-end function and total Endpoint while changing coordinate-wise or path-wise terms. Let $x=(x_1,x_2)^\top$, hidden state $h=x$, and output $s=(1,1)h=x_1+x_2$. A cut over the two hidden coordinates gives

$$
E_1=(x_1,0)^\top,\qquad E_2=(0,x_2)^\top.
$$

Now use the invertible transform $R=\begin{bmatrix}1&1\\0&1\end{bmatrix}$ and rewrite the same function with $h'=Rh$ and $w'=R^{-\top}(1,1)^\top=(1,0)^\top$, so $s=w'^\top h'$. The coordinate-path maps become

$$
E'_1=(x_1,x_2)^\top,\qquad E'_2=(0,0)^\top.
$$

Both implementations satisfy $E_1+E_2=E'_1+E'_2=(x_1,x_2)^\top$, but their terms differ. Changing a valid cut likewise changes the named path groups. Our Q/K/V, head, and residual maps are therefore decompositions of specified actual edges in the frozen ViT implementation, not unique explanations determined solely by the end-to-end function, and they are not generally implementation invariant. Numerical closure establishes accounting without omission or duplication for that representation; it does not establish uniqueness across equivalent graph representations.

### 4.5 Mass, cancellation, and the Value-only approximation

On the raw normalized-input RGB tensor, define the primary spatial measure at pixel $p$ as

$$
M(p)=\sum_{c\in\{R,G,B\}}|E_{p,c}|,
$$

and retain $M^+(p)=\sum_c\max(E_{p,c},0)$ and $M^-(p)=\sum_c\max(-E_{p,c},0)$ for sign analysis. Patch mass is $m_j=\sum_{p\in P_j}M(p)$ and, when total mass is nonzero, $p_j=m_j/\sum_km_k$. Zero-mass cases propagate as NA rather than being regularized by an $\varepsilon$-defined distribution. Normalized spatial entropy is $H=-\sum_jp_j\log p_j/\log196$. For patch tokens, own-token locality is the mass in the Chebyshev-radius-one neighborhood around the target patch minus the corresponding uniform-area baseline; it is undefined for CLS. Normalized distance averages Euclidean distances from the target patch center after division by $13\sqrt2$.

For the complete RGB tensor, also define $m(E)=\sum_i|E_i|$ and $a(E)=\sum_iE_i$. Within-map and between-path cancellation are

$$
C^{\mathrm{map}}=1-\frac{|\sum_iE_i|}{\sum_i|E_i|},\qquad
C^{\mathrm{path}}=1-\frac{\|E^{(1)}+E^{(2)}\|_1}{\|E^{(1)}\|_1+\|E^{(2)}\|_1}.
$$

Routing scale is $\|E^{\mathrm{routing}}\|_1/(\|E^{\mathrm{routing}}\|_1+\|E^{\mathrm{value}}\|_1)$, and identity scale is defined analogously. Alongside these ratios, we report absolute mass for routing/value/full and identity/branch/output. The Value-only approximation retains $A\mathrm dV$ and omits $(\mathrm dA)V$; its Endpoint relative error is $\|E^{\mathrm{full}}-E^V\|_2/\|E^{\mathrm{full}}\|_2$. JVP discrepancy uses the corresponding complete directional response as denominator over frozen directions. An error near 0.95 indicates a discrepancy on the scale of the full response.

Statistics use complete signed RGB tensors; zero denominators or norms yield NA. The blue display convention is described in the experimental settings.

### 4.6 Finite replacement and spatial region selection

The finite component effect is $e_k=F(x)-F(x;z_k\leftarrow z_k^{\mathrm{ref}})$. The primary statistic compares $m(E_k)$ and $|e_k|$ with within-image Spearman correlation. The Taylor quantity is $p_k=\langle\nabla_{z_k}F,z_k(x)-z_k^{\mathrm{ref}}\rangle$. Interventions replace actual head-output or residual-edge activations in the model forward pass; routing/value and Q/K/V differentials are derivative paths and are not treated as independently replaceable forward activations. Endpoint and raw gradient use only the clean state and are clean-only methods; regional Taylor explicitly uses the reference displacement and is reference-aware. Because their information budgets differ, comparisons measure task fit rather than unconditional method rank.

The spatial task fixes three heads at block 5, divides the input into $4\times4$ regions, and replaces 1, 2, or 4 regions. Only the selected head edge is replaced by the activation produced by that perturbed input; other heads and the identity remain clean. Each curve integrates absolute isolated effect over 0–25% area using trapezoids and divides by 0.25, then averages the three heads within each image. Each random repetition takes the absolute effect before integration, and its AUC is then averaged over 20 repetitions, using the same absolute-value order as deterministic methods. Endpoint, total G×I, raw gradient, regional Taylor, and random share the region pool and area budgets.


## 5. Experimental Setup and Numerical Validation

### 5.1 Models and images

We study ImageNet classification models ViT-B/16 and DeiT-S/16. They share 400 evaluation images, with separate sets of 64 discovery and 64 calibration images. Inputs are resized to short side 248 and center-cropped to 224 using each model's normalization; dropout and drop path are disabled. Table 2 summarizes the models.

**Table 2. Model settings**

| Model / setting | ViT-B/16 | DeiT-S/16 |
| --- | --- | --- |
| Blocks / heads / hidden dimension | 12 / 12 / 768 | 12 / 6 / 384 |
| Registered scalar targets/image | 457 | 385 |
| Evaluation images | 400 | 400 (same IDs) |
| Resize / crop | 248 / 224 | 248 / 224 |
| Normalization mean | [0.5, 0.5, 0.5] | [0.485, 0.456, 0.406] |
| Normalization std | [0.5, 0.5, 0.5] | [0.229, 0.224, 0.225] |

The stratified unit study uses a separate set of 100 shared images. Each block selects CLS, 15 spatially stratified patch tokens, and 32 predetermined channels. This gives 6,144 post-block scalars per model-image and 1,228,800 records. Original post-block token-channel coverage is 0.01058%/0.02115% in ViT-B/16 and DeiT-S/16; the stratified study covers 0.338%/0.677%. These fractions define the scope of unit sampling.

### 5.2 Visualization settings

Concept and layerwise figures use a sloth-bear image with a recognizable subject; method comparisons also include a mantis and a little blue heron. Images are selected for presentation readability, while population conclusions retain the complete cohorts. To relate learning images to forward features, each layer selects the channel with highest spatial variance across patch activations and uses its energy as the scalar target. Figures report the channel index and actual 14×14 forward feature map; selected channels may differ across layers.

Main learning images use a deep-blue, cyan, yellow and red positive view. With $P=\max(\sum_cE_c,0)$, displayed values are $\sqrt{\min(P/q_{.99},1)}$, where $q_{.99}$ is computed over positive pixels. This transform improves visibility of weak detail while preserving pixel positions and is excluded from statistics. Supplementary figures pair it with signed and absolute views and separately show a patch-max and bilinear-interpolation display variant, distinguishing native fine detail from a smoothed regional view.

### 5.3 Statistics and numerical validation

Population inference aggregates within images and treats images as statistical units, with 10,000 bootstrap replicates. The 0.10 component-correlation and 0.05 regional-gain values are point-estimate thresholds; sign tests examine direction relative to zero, with the original 28-test Holm correction and CI-width limits of 0.30/0.50. Unit-depth statistics use five-test families per model. Subsequent residual spatial-TV analysis is labeled post-hoc and corrected separately.

Discrete numerical checks are reported in tables rather than sparse error plots. Table 3 summarizes numerical agreement in five original-NNS-compatible network families and structural checks. Original float32 finite-difference failures and their float64 reviews remain in the supplement.

**Table 3. Numerical agreement and structural checks**

| Compatible network | Max relative L2: float64 | Max relative L2: float32 | Comparisons |
| --- | --- | --- | --- |
| fcn | 4.371e-16 | 1.964e-07 | 1,000 |
| cnn | 4.289e-16 | 2.189e-07 | 1,000 |
| cnn_maxpool | 4.769e-16 | 2.427e-07 | 1,000 |
| residual_mlp | 2.686e-16 | 2.155e-07 | 1,000 |
| residual_cnn | 2.923e-16 | 1.692e-07 | 1,000 |

| Structural / directional check | Count | Result |
| --- | --- | --- |
| Positive closures | 76,800 | All pass |
| Omission/duplication controls | 28,800 | All rejected |
| Analytic JVP | 6,144 | All pass |
| Operator directions | 88 | All pass |
| Atlas float32 finite differences | 57,600 | 36 outside tolerance |
| Same-coordinate float64 review | 36 | All pass |

### 5.4 Four experiment groups and evidence levels

The experiments follow the four research questions. Group 1 compares original NNS layerwise propagation with automatic-differentiation Input × Gradient to test the identity in RQ1. Group 2 tests Q+K+V, routing+value, individual-head, and residual identity+branch closure for RQ2. Group 3 evaluates RQ3 using entropy, own-token locality, normalized distance, path absolute mass, relative scales, and Value-only error, with random-initialization, target, and sampling controls. Group 4 replaces actual forward variables or spatial regions and compares clean-only Endpoint with reference-aware Taylor for RQ4. The first two groups establish representational identity and structural correctness, the third describes local-dependence patterns, and the fourth measures finite effects under specified replacements; these evidence levels are not substitutes for one another.

## 6. Depth Patterns of Local Dependence in ViTs

### 6.1 RQ1–RQ2: NNS equivalence and structural-path closure

The 5,000 original-NNS-compatible comparisons in Table 3 give maximum relative L2 errors of $4.769\times10^{-16}$ in float64 and $2.427\times10^{-7}$ in float32 between layerwise learning-image propagation and same-target Input × Gradient. Structural validation covers 76,800 positive closure checks, 28,800 omission/duplication controls, 6,144 analytic JVP checks, and 88 operator directions. All positive closures and analytic directional checks pass, all negative controls are rejected, and all 36 float32 finite-difference exceedances pass same-coordinate float64 review. These results establish the numerical identity in Proposition 1 and structural accounting in Proposition 2; Section 7 evaluates finite-intervention utility separately.

### 6.2 RQ3: Learning images and forward features across layers

Figure 5 shows several internal targets for one input; all-12-layer and module-level learning-image/forward-feature pairs are moved to the supplementary figure bank. A forward feature map gives channel values on the token grid, whereas a learning image describes the corresponding scalar's local dependence on input pixels. The two views are therefore complementary. Early learning images retain local texture, and subsequent layers produce different distributions and details. Background responses and isolated hotspots also occur, showing that high resolution alone does not ensure a complete object contour.

![Figure 5](../figures/png/fig_08_internal_targets.png)

**Figure 5.** Different internal and class targets for one image. Energy targets use the channel of highest forward spatial variance and are distinct from individual token-channel and final class targets.

The sampled-unit statistics characterize this change further (Figure 6). Late-minus-early entropy increases by 0.2201/0.2250 in ViT-B/16 and DeiT-S/16, own-token locality decreases by 0.3558/0.3424, and normalized distance increases by 0.1312/0.1219. All contrasts have the same direction in 100/100 images. Early/middle/late refer to blocks 0–3/4–7/8–11; sampled units depend on broader and more distant input regions.

![Figure 6](../figures/png/fig_12_spatial_broadening.png)

**Figure 6.** Image-level depth trajectories from the full cohorts, with 95% bootstrap intervals. Spatial analyses use 100 images per model and attention/residual analyses 400; inference follows the original analysis.

CLS-minus-patch entropy narrows from 0.4752/0.4712 at block 0 to 0.0086/0.0238 at block 11. Net/absolute mass falls from 0.00872/0.00712 to 0.00144/0.00132. Spatial broadening and substantial signed cancellation can coexist, so absolute mass measures the total magnitude of local terms rather than the net response.

### 6.3 RQ3: Attention heads and routing/value differentials

Figure 7 separates attention into Q, K, routing and value, allowing changes in allocation weights to be examined separately from content transmission. The all-head atlas and cross-model class maps are moved to the supplementary figure bank.

![Figure 7](../figures/png/fig_15_routing_value_examples.png)

**Figure 7.** Q, K, routing and value paths at early and late blocks for the same sloth-bear image. Blue displays retain positive detail; signed closure and absolute masses use complete original arrays.

In the target-conditioned analysis of 400 images, routing scale increases from 0.3092/0.2041 at block 0 to 0.5154/0.4808 at block 11; Value-only Endpoint error increases from 0.4407/0.2774 to 0.8022/0.8488 (Figure 8). Median routing/value/full absolute masses in ViT-B/16 change from 2.6596/7.6041/9.8604 at block 0 to 40.7842/39.0241/54.6768 at block 11; the corresponding DeiT-S/16 values are 4.5294/18.4261/20.1427 and 35.9257/38.8767/44.5653. JVP discrepancies reach 0.9516/0.9636, with full-response norms 14.9711/1.5155 and absolute discrepancies 13.8064/1.4228. The ratios and absolute norms jointly show that fixed-weight $A\mathrm dV$ becomes a poorer approximation to the complete local response at depth; none is a percentage of forward output or causal contribution.

![Figure 8](../figures/png/fig_16_attention_depth.png)

**Figure 8.** Image-level depth trajectories from the full cohorts, with 95% bootstrap intervals. Spatial analyses use 100 images per model and attention/residual analyses 400; inference follows the original analysis.

After controlling for activation and downstream-cotangent norms, partial-rank correlations between mean-query saturation and routing scale remain −0.1442/−0.3257. Sharper attention is not a direct proxy for stronger routing dependence. CLS-query results are weaker; complete sensitivity analyses are in the supplement.

### 6.4 RQ3: Residual identity/branch differentials

Figure 9 shows attention and MLP residuals for the same input and target. Both combine identity and branch paths, but their identity paths originate at different input edges and exhibit different depth trends. Late-minus-early identity scales are −0.1242/−0.0449 for attention residuals and +0.1798/+0.1780 for MLP residuals (Figure 10).

![Figure 9](../figures/png/fig_17_residual_examples.png)

**Figure 9.** Attention and MLP residual paths in block 5 for the same class target. The identity paths refer to different residual input edges. Positive displays are not additive; the complete signed maps satisfy closure.

![Figure 10](../figures/png/fig_18_residual_depth.png)

**Figure 10.** Image-level depth trajectories from the full cohorts, with 95% bootstrap intervals. Spatial analyses use 100 images per model and attention/residual analyses 400; inference follows the original analysis.

Attention-residual identity/branch absolute masses in ViT-B/16 change from 495.2271/63.8344 at block 0 to 313.3689/398.0202 at block 11; in DeiT-S/16 they change from 459.3077/63.9204 to 372.6598/185.3893. MLP-residual values change from 130.4998/438.0538 to 827.8407/460.2701 and from 170.6143/412.5634 to 614.5721/354.6503, respectively. Subsequent full-cohort spatial analysis gives late-minus-early output–identity patch-TV changes of +0.0520/+0.0282 for attention and −0.0187/−0.0462 for MLP; all four survive a separate Holm correction. These results describe different local transmission patterns in two trained models. Token-channel depth analysis and class-path analysis use different targets and provide complementary model-level observations.

### 6.5 Visual comparison of representational objects

Supplementary figures compare same-input Endpoint, Grad-CAM, attention rollout and raw attention. Endpoint and Grad-CAM use the GT logit. Grad-CAM is evaluated at the final block's norm1 output, avoiding degenerate terminal patch gradients caused by CLS readout. Rollout uses head averaging, identity residuals, row normalization and successive matrix multiplication; raw attention uses the final block's mean CLS-to-patch weights.

Endpoint retains pixel-scale local terms, whereas Grad-CAM and attention methods provide coarser regional distributions. The patch-max interpolation variant shows that aggregation and interpolation alone can substantially change the appearance of a learning image without changing its base definition. Methods sometimes emphasize different regions, reflecting differences in their targets and propagation rules. This comparison illustrates granularity and explanatory objects; ranking methods requires quantitative evaluation matched to the intended task.

### 6.6 RQ3 control: training and initialization

Across 32 shared images per model, all 12 blocks and three random initialization states, random models also show spatial broadening. Trained entropy late-minus-early is 0.2288/0.2256, versus random means of 0.0940/0.1718. Trained-minus-random differences are 0.1348 [0.1232, 0.1466] and 0.0539 [0.0473, 0.0602], with intervals resampling model seeds and shared images. Routing-scale trained/random depth changes are 0.0512/−0.0479 and 0.1314/−0.0043; attention identity decreases in trained models and increases in random models. MLP identity increases in both, with the ViT-B increment-difference interval crossing zero. The trained and random parameter states differ by the reported spatial-broadening magnitudes and path trends; this comparison does not identify a causal effect of training.

### 6.7 RQ3 control: target robustness

Across 100 frozen images per model at blocks 0, 5 and 11, GT, clean-top1 and top1 margin preserve increasing routing/Value-only error, decreasing attention identity and increasing MLP identity. Correct/incorrect counts are 92/8 for ViT-B and 86/14 for DeiT-S, with both subsets reported separately. Margin changes magnitudes: ViT-B routing B11-minus-B0 is 0.1439 versus 0.2031 for GT. Native attention-output JVP is class independent; target-cotangent contracted directional discrepancy is reported separately.

### 6.8 Quantitative feature–Endpoint association

With 32 images per model, 12 blocks and 16 frozen random channels per block, 12,288 patch-energy targets pair squared forward-response distributions with absolute Endpoint patch mass. Correct-pair mean image-level Spearman is 0.2663/0.2811, exceeding wrong-channel, cross-layer same-channel, spatial-permutation and random-channel controls; all paired-difference intervals are positive. Cosine, Jensen–Shannon distance and top-20 overlap supply corresponding quantitative comparisons. Forward responses and local input dependence have a same-channel association while remaining complementary objects.

### 6.9 RQ3 control: sampling sensitivity

Reusing original M3 unit records for the same 32 images, channel/token-stratum/image bootstraps and leave-one-block/stratum-out analyses retain increasing entropy and distance and decreasing locality. This characterizes sampling sensitivity in the existing sample. Full settings, metric estimates, intervals and five supplementary figures are in [Supplementary experiments v1](supplement_experiments_v1.md).

## 7. Ranking in Prescribed Finite-Replacement Tasks

### 7.1 RQ4: Prescribed component replacement and regional selection

In component replacement, 11 of 12 model–reference–component-family combinations satisfy the original composite criteria. Table 4 directly reports estimates, intervals and threshold relationships. ViT-B normalized-zero routing/value has estimate 0.0996 and the corresponding DeiT-S estimate is 0.1065; both intervals cross 0.10. Satisfying a family criterion and establishing a population effect above 0.10 are therefore different conclusions. The all-family criterion is unmet.

**Table 4. Component association estimates and intervals**

| Model | Reference | Family | Median rho | 95% CI | Holm p | Composite criterion | CI lower > 0.10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | heads | 0.2909 | [0.2813, 0.2982] | 2.17e-119 | Met | yes |
| ViT-B/16 | normalized-zero | routing/value | 0.0996 | [0.0748, 0.1226] | 3.00e-14 | Unmet | no |
| ViT-B/16 | normalized-zero | residual/branches | 0.1774 | [0.1548, 0.1952] | 2.56e-46 | Met | yes |
| ViT-B/16 | blur | heads | 0.3063 | [0.2957, 0.3181] | 2.17e-119 | Met | yes |
| ViT-B/16 | blur | routing/value | 0.1565 | [0.1322, 0.1774] | 9.96e-29 | Met | yes |
| ViT-B/16 | blur | residual/branches | 0.2091 | [0.1817, 0.2374] | 1.46e-41 | Met | yes |
| DeiT-S/16 | normalized-zero | heads | 0.1910 | [0.1769, 0.2052] | 1.88e-76 | Met | yes |
| DeiT-S/16 | normalized-zero | routing/value | 0.1065 | [0.0817, 0.1291] | 2.11e-10 | Met | no |
| DeiT-S/16 | normalized-zero | residual/branches | 0.1352 | [0.1148, 0.1643] | 4.35e-19 | Met | yes |
| DeiT-S/16 | blur | heads | 0.1817 | [0.1688, 0.1951] | 3.82e-72 | Met | yes |
| DeiT-S/16 | blur | routing/value | 0.1600 | [0.1278, 0.1822] | 9.90e-31 | Met | yes |
| DeiT-S/16 | blur | residual/branches | 0.2287 | [0.2035, 0.2504] | 3.70e-54 | Met | yes |

In the supplementary within-block head comparison, Taylor achieves higher Spearman and Capture and lower Regret (Table 5). Across-layer residual identity has Endpoint-minus-Taylor Spearman differences of 0.0865–0.4209 but a zero median Capture@1 difference, indicating task- and metric-dependent incremental value.

**Table 5. Within-block head ranking: E=Endpoint, T=Taylor**

| Model | Reference | E rho | T rho | E Capture@1 | T Capture@1 | E Capture@3 | T Capture@3 | E Regret@1 | T Regret@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | 0.2815 | 0.6742 | 0.5174 | 0.8038 | 0.6149 | 0.8567 | 0.1701 | 0.0571 |
| ViT-B/16 | blur | 0.2727 | 0.6856 | 0.5075 | 0.8203 | 0.6185 | 0.8659 | 0.1163 | 0.0346 |
| DeiT-S/16 | normalized-zero | 0.1952 | 0.6095 | 0.5609 | 0.8028 | 0.7122 | 0.8894 | 0.2381 | 0.0860 |
| DeiT-S/16 | blur | 0.1476 | 0.6238 | 0.5506 | 0.8426 | 0.7105 | 0.8970 | 0.1326 | 0.0443 |

The spatial experiment fixes three heads in block 5, a shared 4×4 region pool and a maximum 25% area. Figure 11 and Tables 6–8 report paired differences, absolute AUC and normalized regret. Paired median Endpoint-minus-regional-Taylor differences are −0.023433, −0.023697, −0.029900 and −0.035630. None of the four families meets the +0.05 criterion over every baseline. The candidate oracle maximizes effects only over evaluated sets and is not a global optimum over all regions.

![Figure 11](../figures/png/fig_21_intervention_boundary.png)

**Figure 11.** Finite region-replacement curves for the example and absolute AUC over complete 400-image families. The candidate oracle is restricted to evaluated region sets; mathematical closure and spatial intervention selection measure different properties.

**Table 6. Paired spatial-intervention differences**

| Model | Reference | vs total G×I | vs gradient | vs Taylor | vs random | Composite criterion |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | -0.0009 | -0.0000 | -0.0234 | -0.0023 | Unmet |
| ViT-B/16 | blur | -0.0010 | -0.0007 | -0.0237 | -0.0031 | Unmet |
| DeiT-S/16 | normalized-zero | -0.0018 | -0.0006 | -0.0299 | 0.0009 | Unmet |
| DeiT-S/16 | blur | 0.0002 | -0.0002 | -0.0356 | -0.0010 | Unmet |

**Table 7. Absolute AUC and output scale**

| Model | Reference | Endpoint AUC | Taylor AUC | Random AUC | Candidate oracle AUC | Clean GT logit |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | 0.0309 | 0.0558 | 0.0339 | 0.0654 | 8.8989 |
| ViT-B/16 | blur | 0.0276 | 0.0518 | 0.0307 | 0.0747 | 8.8989 |
| DeiT-S/16 | normalized-zero | 0.0617 | 0.0961 | 0.0611 | 0.1474 | 8.7974 |
| DeiT-S/16 | blur | 0.0395 | 0.0770 | 0.0411 | 0.1063 | 8.7974 |

**Table 8. Candidate-oracle regret and logit-scale ratios**

| Model | Reference | Endpoint regret | Taylor regret | Random regret | Endpoint AUC/abs(logit) | Taylor AUC/abs(logit) |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B/16 | normalized-zero | 0.5303 | 0.1442 | 0.5059 | 0.0035 | 0.0062 |
| ViT-B/16 | blur | 0.6366 | 0.2557 | 0.5929 | 0.0031 | 0.0059 |
| DeiT-S/16 | normalized-zero | 0.5646 | 0.3352 | 0.5947 | 0.0072 | 0.0107 |
| DeiT-S/16 | blur | 0.6289 | 0.2705 | 0.6153 | 0.0045 | 0.0089 |

Cross-reference analysis clarifies this distinction. Endpoint uses the clean observed-input map, whereas Taylor explicitly uses replacement displacement. Across 163,200 component pairs, Taylor magnitude and absolute effect change under the original atol=1e-12, rtol=0 criterion. Their distributions and head-rank concordance are reported in supplementary tables. Reference choice changes intervention-related ordering, so local input maps and displacement estimators serve different questions.

With the corrected random estimator, paired Endpoint-minus-random medians are −0.0023, −0.0031, 0.0009 and −0.0010; both DeiT-S intervals include zero. All four Taylor-minus-random intervals are positive. The block-5 three-head task follows the original confirmatory protocol; supplementary cross-layer all-head M2 covers blocks 2/5/10 and favors Taylor in 11/12 full-input and 12/12 isolated-head families (Supplement S14).

## 8. Discussion and Conclusion

### 8.1 Discussion

This paper has one position: Gradient × Input is the local measurement primitive, and the contribution is a structure-aware Jacobian scanner for heterogeneous internal ViT targets with closable and auditable differential accounting on specified actual edges. Proposition 1 establishes the mathematical identity, Proposition 2 states closure conditions for a specified graph and cut, and the mechanistic findings come from frozen target samples in two fixed ImageNet-pretrained checkpoints.

Structural closure, local path scale, and finite-replacement ranking answer three distinct questions. Additive closure of Q/K/V, individual heads, and identity/branch paths shows that the terms reconstruct the specified target differential. Routing and identity scales compare relative map norms. Activation replacement measures a finite effect under a specified reference and replacement operator. RQ4 evaluates component ranking and regional selection under full replacement and fixed area budgets; it does not estimate a boundary for Taylor error as intervention amplitude changes.

Learning images place heterogeneous internal objects on the same input grid. Explicit targets, layers, and channels support comparisons between forward responses and local input dependence and show how attention and residual paths combine. Correct feature–Endpoint pairs outperform four negative controls, while their association remains far from equivalence. Pixel-resolution maps also remain conditioned by the $16\times16$ patch embedding and the normalization-coordinate origin.

Blue high-intensity displays make distributed texture easier to inspect while showing only the positive signed RGB sum. Supplementary figures pair this view with signed and absolute maps. Positive values indicate the sign of local coordinate-scaling terms rather than prediction support under finite deletion, and large absolute mass can coexist with substantial cancellation.

Training controls cover three initialization seeds per architecture; the empirical scope excludes same-architecture alternative trained checkpoints, independent channel seeds, a target-matched M3–M4 bridge, and cascading randomization. Incorrect subsets contain 8/14 images, and spatial intervention covers the registered regions, references, and blocks 2/5/10. ViT-B Path IG does not meet the common calibration requirement within the fixed 512-point range, placing cross-model Endpoint–Path comparison outside the result scope.

### 8.2 Conclusion

Four conclusions follow. First, with exact current-point local derivatives and no additional redistribution, a scalar NNS learning image equals Input × Gradient, making Gradient × Input the framework's base measurement primitive. Second, the framework registers heterogeneous internal targets and defines actual-edge differential cuts on a specified ViT graph; Q-induced, K-induced, and V-induced attention paths and identity and branch residual paths add to complete local input dependence on a valid cut, while their terms vary with graph, cut, and parameterization. Third, frozen target samples in the two fixed ViT-B/16 and DeiT-S/16 checkpoints show spatial broadening, increasing routing-relative scale and Value-only error in deep attention, and distinct attention-versus-MLP residual trends; trained and random parameter states exhibit the reported differences. Fourth, local path maps have broad but non-universal association with finite effects in the prescribed component-replacement task, while displacement-aware Taylor ranking is better suited to the prescribed reference-specified regional replacement. These elements form a unified framework for structured local-Jacobian scanning, verifiable accounting, and two-model mechanistic analysis.

## References and Supplementary Material

[References](../references.md); [supplement](supplement.md). Author, affiliation and final archive information remain to be completed: [REPOSITORY AND ARCHIVE REQUIRED].
