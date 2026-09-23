# Dual-Indexed Local Jacobian Scanning and Structural Auditing of Internal Targets in Vision Transformers


## Abstract

A scalar Neural Network Scanner (NNS) learning image equals same-target Gradient × Input under exact current-point derivatives without additional redistribution; we call this quantity Endpoint in normalized input coordinates. Ordinary Gradient × Input retains input coordinates but does not explicitly distinguish which downstream operator and input slot uses the same intermediate representation. We therefore introduce a local differential representation jointly indexed by input coordinate and concrete tensor use for a fixed Vision Transformer (ViT) graph. It decomposes a scalar target into attention-head, Q/K/V, routing/value, and residual identity/branch components, with separate branch-cut and global-cut reconstruction contracts. Graph certification of the actual ViT topology, numerical closure, an independent analytic implementation, and fault injection jointly validate completeness and implementation correctness.

Across two fixed checkpoints, the dual-indexed maps reveal clear depth-associated spatial organization. Input dependence for unit targets becomes spatially broader, K/V component entropy decreases with depth, and attention and MLP residuals exhibit distinct spatial reorganization and scale trajectories. Spatial broadening persists under low-total-mass stratification and trimming, while three input-coordinate measures retain most major trends. An independent image cohort and four public checkpoints further reproduce unit broadening, routing allocation, and residual trajectories. Dual indexing thereby supports spatial comparison across internal modules in common input coordinates and exposes structural differences unavailable from either a total map or a component scalar alone.

These results establish an Endpoint scanning framework for internal ViT targets. It represents local dependence in common input coordinates, retains the structural identity of attention, MLP, and residual tensor uses, and reconstructs the same target through valid cuts. Internal-target visualization thus becomes a decomposable, closure-tested, and cross-module structural analysis. Finite-replacement experiments further define its use: Endpoint describes structured local dependence at the observed state, while intervention ranking for a specified reference can incorporate the corresponding displacement within the same analytical framework.

Keywords: Vision Transformer; Neural Network Scanner; Gradient × Input; local Jacobian; structural audit

## 1. Introduction

Vision Transformers (ViTs) divide images into patch tokens and build representations through self-attention and feed-forward layers (Dosovitskiy et al., 2021). This architecture models relationships across positions while making internal computation difficult to interpret. Token-channel units represent responses at particular positions, attention heads transmit information across positions, and residual connections combine inputs with branch outputs. Understanding their relationships calls for a representation that preserves input locations and applies to different internal objects.

Existing visualizations commonly show feature maps, attention weights, or class-related regions. Each view serves a purpose, but their resolutions, targets, and propagation rules differ. A forward feature map shows a channel's response on the token grid, whereas attention weights describe allocation between tokens. Neither is the same object as an internal scalar's dependence on the original image. Mapping internal targets to common input coordinates provides a shared basis for comparing modules. Here, a “path map” is a local differential term for a specified tensor use, not a unique causal path.

Neural Network Scanner (NNS) associates a neuron with a learning image that preserves input positions (Dou et al., 2023). With exact local derivatives, its layerwise scale propagation is an application of the Jacobian chain rule; for a scalar target, the endpoint learning image is Input × Gradient. This identity fixes the local measurement primitive. We retain two free indices beyond that total map. Input coordinate $i$ records where local dependence occurs, while concrete consumer-use edge $e$ records the specific use through which the term passes. Summing out $e$ prevents a total map from distinguishing the spatial allocation of Q/K/V or identity/branch terms; summing out $i$ prevents an edge scalar from recovering component entropy, spatial TV, or signed spatial cancellation. The target registry fixes condition $s$, and a structurally valid cut binds the terms back to the same total local differential. The dual-indexed family therefore measures redistribution over specified uses and input positions and audits the declared cut contract.

![Figure 1](../figures/png/fig_01_concept_overview.png)

**Figure 1.** Left: Total Endpoint for one fixed target and input point. Middle: retain native-use identity and input coordinates, with a separate definition for differential partitions inside attention. Right: compare spatial distributions and prescribed finite replacement effects. Global closure includes the residual bypass; a branch partition reconstructs only the attention branch contribution.

The overarching question is how NNS can represent the local input dependence of an internal ViT target as computable, decomposable, and verifiable structural-path maps, and how these maps vary with depth and rank candidates in prescribed finite-replacement tasks. We address four questions:

1. **Mathematical object: what does an NNS learning image represent?** We relate layerwise scale propagation to the Jacobian chain rule and Input × Gradient, including the effects of non-homogeneous operations, fixed tokens, and the input-coordinate origin.
2. **Structural extension: how can NNS be applied to a ViT computational graph?** We define internal scalar targets, actual graph edges, valid computational cuts, and additive local differentials for attention, multi-head structure, and residual paths.
3. **Spatial distribution: how do structural components vary with depth?** In frozen target samples from two fixed pretrained checkpoints, we analyze spatial broadening, routing/value, and identity/branch trajectories and compare trained and random parameter states together with target and sampling controls.
4. **Finite-replacement utility: can local differential scores rank prescribed finite replacement effects?** We compare clean-state path scores, activation × gradient, and first-order Taylor scores that include actual activation displacement.

The method contribution has two parts. First, a target-conditioned dual-index local differential representation specifies distinct reconstruction contracts for global cuts and branch partitions. Second, a structural audit checks ViT use identity, omissions, and duplication through architecture-graph certification and an independent analytic reference. The empirical analysis separately addresses spatial distributions and finite-replacement ranking in fixed checkpoints. Total maps describe overall broadening, and paired maps describe attention and residual spatial differences; path maps do not improve finite head-replacement ranking, so structural resolution is distinct from intervention-predictive ability.

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

Edge-based analysis and internal intervention already extend to vision. Vi-CD discovers and validates sparse residual-stream edge circuits in ViTs and OpenCLIP by activation patching (Żukowska et al., 2026), while CAAP constructs ViT patch attributions by inserting source-image activations into a neutral context (Izadi et al., 2026). Actual edges and ViT activation patching are established precedents.

### 2.5 Visual attribution and position of this work

Grad-CAM provides class-conditioned localization through gradient-weighted feature maps (Selvaraju et al., 2017), attention rollout combines attention flow across layers (Abnar and Zuidema, 2020), and Chefer et al. (2021) design relevance propagation for Transformers. These approaches, Gradient × Input, token decomposition, LRP, and finite replacement address different explanation objects. Table 1 focuses on five nearest method families; the complete matrix is retained in the supplement. “Not reported” refers only to the versions reviewed. Live checks were performed on 23 September 2026 (PRC); DAVE, Vi-CD, CAAP, HiLRP, GradSkip, and Unpack are identified by their corresponding arXiv preprint versions.

**Table 1. Object-level comparison with five nearest method families**

| Method family | Primary object | Index retained | Reconstruction target | Empirical role |
| --- | --- | --- | --- | --- |
| Conductance | Reference-to-input path attribution through hidden units | Hidden unit and input | Output difference across a layer | Hidden-unit attribution |
| DecompX / LRP | Token representation or relevance components | Token/operator, method-dependent | Representation, logit, or conserved relevance | Token attribution and localization |
| EAP / AtP* / Vi-CD | Reference-specific component or edge effects | Component/edge | Patching effect or circuit fidelity | Circuit localization and intervention |
| DAVE | Distribution-aware ViT input-gradient decomposition | Input position and gradient term | Effective input–output map | ViT spatial attribution |
| Unpack | Component interactions and named K/Q/V paths | Component/path and token | Method-specific credit decomposition | Composition and token credit |
| NNS–ViT | Current-point local differential by concrete use and input position | Consumer use $e$ and input coordinate $i$ | Branch contribution or same-target total differential, according to the declared cut | Structural audit and spatial comparison |

Conductance uses output difference for path completeness; DecompX/LRP reconstruct representations or relevance; EAP/Vi-CD address reference-specific edge effects; DAVE decomposes ViT input gradients; Unpack retains named component paths. The technical difference studied here is the joint representation of current-point differentials by concrete use and input position, together with implementation-consistency checks for a declared cut contract. Gradient × Input, component decomposition, and activation patching are established methods.

## 3. Jacobian Interpretation of NNS

### 3.1 General computational-graph representation of NNS

Let the network be a directed acyclic computational graph $G=(V,E)$, with input $x$ and scalar target $s(x)$. A neuron produces a scalar value, whereas its learning image retains the input coordinates on which that scalar depends. Figure 1 first separates total Gradient × Input from our dual-indexed expansion; coordinate initialization and original NNS layerwise propagation diagrams remain in the supplementary figure bank. We denote the final scalar map by $E_s(x)$ and call it the NNS local input map.

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

## 4. Dual-Indexed Local Differentials for Internal ViT Targets

### 4.1 Target registry and the dual-indexed object

The machine contracts distinguish three registries. The scalar-target registry fixes readout $s$, graph location, and aggregation rule. The local-differential decomposition registry fixes native uses, analytic partitions, cuts, and reconstruction objects. The finite-replacement registry fixes intervenable variables, references, and effects. These object classes are not interchangeable: Q/K/V-induced, routing/value, and residual identity/branch terms are differential components rather than scalar targets; attention probabilities, value projections, head outputs, and residual branch activations are intervention variables rather than scalar readouts. For patch activations in channel $c$, an aggregate target can be defined as

$$
s_{\mathrm{energy}}=\frac12\sum_{t=1}^{196}h_{t,c}^2,\qquad
\nabla_xs_{\mathrm{energy}}=\sum_th_{t,c}\nabla_xh_{t,c},\qquad
E_{\mathrm{energy}}=\sum_th_{t,c}E_{h_{t,c}}.
$$

This is the Jacobian form of original NNS activation-weighted filter aggregation; energy is an aggregate scalar rather than one neuron. Per-image scalar targets comprise 192 post-block token-channel coordinates, 48 post-block patch sums, 48 post-block patch energies, 144/72 attention-head coordinates in ViT-B/DeiT-S, 18 module patch aggregates, four patch-embedding aggregates, one final-CLS coordinate, and two class logits, totaling 457/385. They are frozen structural anchors rather than an exhaustive or statistically representative neuron sample. Equal channel indices across models do not imply semantic correspondence.

The registry specifies target location and readout semantics; registration alone does not make every cut applicable. Auditing the module, location, and kind fields of the 457/385 registered targets gives the following rules.

| Target type or position | Admissible global cut | Admissible branch partition | Inapplicable condition |
| --- | --- | --- | --- |
| Final class logit | Identity plus a complete branch partition at a specified residual | Attention branch upstream of that residual | None |
| Internal scalar after a residual output | An upstream residual cut intercepting every input path | Corresponding upstream branch | Cut downstream of the target |
| Target before a residual or inside a branch | A cut determined by its ancestor subgraph | Only when the target is downstream of that branch | A later residual equality cannot be applied |
| Energy or patch-sum aggregate | Determined by the actual module/location readout | Determined by actual ancestry | Registry membership does not grant every cut |

The table is also certified entry by entry on a frozen architecture graph. For the two standard 12-block pre-LN ViTs, we explicitly expand patch embedding, Q/K/V, each head use, attention projection, both residual additions, the MLP, final normalization, and classifier, then traverse all 842 scalar targets, 12 blocks, and eight cut declarations, giving 80,832 target--cut pairs. A full target cut requires every declared edge to lie in the target ancestor graph and every input-to-target path to cross it exactly once. A branch partition is certified only for complete interception of the attention-branch contribution and requires that branch to be upstream of the target. This classifies 26,094 pairs as full target cuts, 15,660 as valid branch-only partitions, and 39,078 as upstream or reconstruction-inapplicable. Certification applies to the registry-bound standard unfused pre-LN architecture, not implementations with extra branches or different operator fusion.

Define a concrete use edge as

$$
e=(p_e,c_e,q_e,\nu_e),
$$

where $p_e$ is the producer, $c_e$ the consumer, $q_e$ the consumer input slot, and $\nu_e$ the invocation. Let $z_e(x)$ be the producer output as it enters that consumer use. Let $\bar z_e^{(s)}$ be the target cotangent at that input slot before it is merged with other fan-out uses of the same producer. We define

$$
\boxed{
\mathcal L_{e,i}(s;x)
=x_i\left[J_{z_e,x}(x)^{\mathsf T}\bar z_e^{(s)}\right]_i.
}
$$

Here $i$ and $e$ are free indices and $s$ is the target condition selected by the registry. The Jacobian, cotangent, and $z_e$ are evaluated at the same observed input $x$ and forward state. Fixing $e$ yields one input-position-preserving map; fixing $i$ compares local terms from one input coordinate across concrete uses; summing over a global cut returns the total. This representation retains concrete uses and input positions. Total Endpoint remains the established Gradient × Input primitive.

One producer tensor may enter several consumers, different slots of one consumer, or repeated invocations. Their upstream $J_{z_e,x}$ values may coincide, but $\bar z_e^{(s)}$ belongs to a different downstream use. Merging these cotangents by tensor name discards path identity before measurement, so edge identity includes the consumer, slot, and invocation. Figure 2 places this contract beside concrete ViT cuts.

![Figure 2](../figures/png/fig_02_neuron_learning_image.png)

**Figure 2.** Left: two consumer slots of a shared producer remain distinct. Middle: the lower blue path is the residual identity bypass; the upper path is attention. The green dashed region contains a branch partition and reconstructs attention branch contribution A_F. The red dashed global cut includes identity and the complete branch partition and reconstructs Total Endpoint L_F. Right: the two reconstruction targets.

### 4.2 Attention branch differentials and residual global cuts

Figure 3 describes differential partitions inside the attention branch. Figure 4 shows the identity and branch uses at a residual addition.

![Figure 3](../figures/png/fig_06_attention_differential.png)

**Figure 3.** The upper row shows one head's forward computation, with attention probability $A$ and value projection $V$. The lower row decomposes $\mathrm dO_h=(\mathrm dA)V+A\mathrm dV$ at the same forward point. Q-induced and K-induced terms form the routing-induced differential; the V-induced term is the value-induced differential. After contraction with the downstream cotangent through head concatenation and output projection, these terms reconstruct the attention branch contribution. The global target additionally requires the residual bypass.

![Figure 4](../figures/png/fig_07_residual_principle.png)

**Figure 4.** The identity and branch inputs of a residual addition define separate local maps for the same final class target and the cotangent at the addition output. Complete signed maps sum to Total Endpoint. Independently normalized blue positive displays show spatial structure.

For each head, $Q=XW_Q+b_Q$, $K=XW_K+b_K$, $V=XW_V+b_V$, and

$$
A=\operatorname{softmax}(QK^\top/\sqrt{d_h}),\quad O_h=AV,\quad
\mathrm dO_h=(\mathrm dA)V+A\mathrm dV.
$$

Here $\mathrm dA=J_{\mathrm{softmax}}(S)\mathrm dS$ and $\mathrm dS=(\mathrm dQK^\top+Q\mathrm dK^\top)/\sqrt{d_h}$. Let $a=f(u)$ be the attention branch output, $y=u+a$ the residual output, and $g_y^{(s)}$ the cotangent of final target $s$ at $y$. Define

$$
\mathcal B_i^{\mathrm{attn}}(s;x)
=x_i\left[J_{a,x}(x)^{\mathsf T}g_y^{(s)}\right]_i,
\qquad
\mathcal B_i^{\mathrm{attn}}
=\mathcal B_i^Q+\mathcal B_i^K+\mathcal B_i^V
=\mathcal B_i^{\mathrm{routing}}+\mathcal B_i^{\mathrm{value}}.
$$

Q/K/V projection slots are native computational uses. The Q-induced, K-induced, and V-induced terms are derivative contributions induced by those slots. The routing-induced differential analytically groups the Q-induced and K-induced terms; the value-induced differential is $A\mathrm dV$. These branch-internal differential partitions reconstruct the attention branch contribution and do not alone equal the full class-target G×I. Each head-map cotangent includes concatenation, output projection $W_O$, and subsequent derivatives to the final target. Components may overlap in input space.

For the residual identity, define

$$
\mathcal L_i^{\mathrm{identity}}=x_i[J_{u,x}^{\mathsf T}g_y^{(s)}]_i,
\qquad
E_{s,i}=x_i\frac{\partial s}{\partial x_i}
=\mathcal L_i^{\mathrm{identity}}+\mathcal B_i^{\mathrm{attn}}.
$$

The identity term uses the cotangent at the addition output. Substituting $\nabla_u s$, which already includes the branch, duplicates paths. At the second residual addition, carry plus MLP branch reconstructs the same Total Endpoint. The tables distinguish the mathematical objects and implemented reconstruction targets.

| Object class | Examples | Native forward tensor/use | Direct replacement | Reconstruction or estimand |
| --- | --- | --- | --- | --- |
| Native computational edge | Residual add identity/branch input; head-to-concat input; Q/K/V projection slot | Yes | Only under a specified runtime contract | Branch or full-target differential, depending on the edge set |
| Analytic differential partition | Q-induced, K-induced, routing-induced, value-induced terms | No; groups of derivative terms | No | Attention branch contribution |
| Intervenable forward variable | Attention probability $A$; value projection $V$; head output; residual branch activation | Yes | Yes | Prescribed finite replacement effect |
| Aggregate or ranking score | Map absolute mass; regional map score; reference-aware Taylor score | No | Not applicable | Specified statistical estimand |

| Named component set | Required terms | Reconstruction target |
| --- | --- | --- |
| heads_branch | All head maps | Attention branch contribution $\mathcal B^{\mathrm{attn}}$ ($A_F$ in code) |
| qkv_branch | Q-induced + K-induced + V-induced | Attention branch contribution $\mathcal B^{\mathrm{attn}}$ ($A_F$) |
| routing_value_branch | routing-induced + value-induced | Attention branch contribution $\mathcal B^{\mathrm{attn}}$ ($A_F$) |
| heads_with_bypass / qkv_with_bypass / routing_value_with_bypass | Identity + the corresponding complete branch partition | Total Endpoint $E_s$ ($L_F$ in code) |
| residual1 / residual2 | Identity + attention / carry + MLP | Total Endpoint $E_s$ ($L_F$) |

### 4.3 Structurally valid cuts and pointwise differential closure

For a specified tensor-use DAG, input node $x$, and scalar target $s$, let $C$ be a set of concrete use edges. The set is a **structurally valid cut** for that graph and target if every directed use-path from $x$ to $s$ in the target's ancestor subgraph crosses exactly one edge in $C$. This property depends on graph structure, target location, and use identity, not on the current input value or whether a derivative happens to vanish at that point.

At fixed input, parameters, and derivative convention, $C$ has **pointwise differential closure** for the target when

$$
\nabla_x s=\sum_{e\in C}J_{z_e,x}^{\mathsf T}\bar z_e^{(s)},
$$

Numerical closure tests implementation consistency at that point. It cannot detect an omitted structural path whose derivative vanishes at the tested input.

**Proposition 2 (a structurally valid cut is sufficient for pointwise differential closure).** The scalar target must lie downstream of the cut, and every structural input-to-target path must pass through the declared cut. If the graph is differentiable at $x$ under a fixed derivative convention, all components use the same forward state, and $C$ is structurally valid in the target's ancestor subgraph, then, for every input coordinate $i$,

$$
\boxed{
\sum_{e\in C^{\mathrm{global}}}\mathcal L_{e,i}(s;x)
=x_i\frac{\partial s(x)}{\partial x_i}.
}
$$

Inputs to a multi-input operator are counted separately; repeated uses of a shared tensor are distinguished by the tuple $e$ rather than tensor name. The proposition does not apply to a target downstream of, outside, or reachable around the claimed cut.

**Proof.** Treat the tensors at the cut uses as components of one multivariate map $z_C(x)=(z_e(x))_{e\in C}$ while retaining a use-specific cotangent at each consumer input slot. Structural validity implies that the target ancestor subgraph factors as $s=\tilde s(z_C(x))$: no input path bypasses $C$, and no path is counted twice in $C$. The multivariate chain rule gives

$$
\nabla_xs=J_{z_C,x}^{\mathsf T}\nabla_{z_C}\tilde s
=\sum_{e\in C}J_{z_e,x}^{\mathsf T}\bar z_e^{(s)}.
$$

Here $\bar z_e^{(s)}$ is assigned to that concrete use; different consumers, slots, or invocations of a shared producer are not merged first. Multiplication by $x_i$ gives the Endpoint equality. A coordinate expansion supplies an intuitive path view, but the proof does not enumerate tensor-computation paths. Cutting only $f(u)$ omits the identity bypass in $y=u+f(u)$; using $\nabla_us$, which already includes the branch, as the identity term and then adding the branch duplicates it.

A minimal counterexample is $s(x)=x+x^2$. At $x=0$, omitting the square branch leaves derivative $1=\mathrm ds/\mathrm dx$, so the pointwise numerical closure error is zero. The structural path through the square branch still exists, and the edge set that omits it is not a structurally valid cut. Pointwise equality therefore cannot certify structural completeness.

**Method ablation table: index necessity and object degeneration**

| Retained structure | Observable | Lost when removed | Degeneration |
| --- | --- | --- | --- |
| $[e,i]$ with registered $s$ and valid $C$ | Path-specific spatial maps that reconstruct one total local differential | Nothing in the declared object | Auditable dual-indexed map family |
| no $i$ | Per-edge aggregate magnitude or score | Spatial allocation and signed spatial cancellation | Edge scalar |
| no $e$ | Total input-coordinate dependence | Routing/value, head, and identity/branch identity | Total Gradient × Input |
| no registered $s$ | Maps from ad hoc scalar readouts | Comparable target semantics and reproducible atlas coverage | Unregistered target collection |
| no valid-cut contract | Named component maps | Detection of omitted or duplicated live paths | Unverified component list |

This is a structural rather than empirical ablation. Removing $i$ hides spatial broadening and within-path redistribution; removing $e$ adds different paths into one total map; removing the registry for $s$ removes fixed cross-target semantics; and removing the valid-cut contract leaves visually plausible maps without evidence against omitted or duplicated paths.

**Structural audit coverage**

| Audit layer | Reconstruction / question | Direct evidence | Scope |
| --- | --- | --- | --- |
| Formula identity | Same-target Total Endpoint | 5,000 numerical comparisons; 88 operator directions | Declared exact local-derivative rules |
| Branch closure | Attention branch contribution A_F | 28,800 branch-partition closures | Fixed tested attention graphs |
| Global closure | Total Endpoint L_F | 48,000 bypass-inclusive or residual closures | Fixed tested global cuts |
| Existing mutations | Omitted head/routing, duplicate identity | 9,600 of each error, all rejected | Three prescribed fault classes |
| Independent analytic reference | Q/K/V, routing/value, heads, identity, branch, full | 32 component comparisons and 9 central-difference directions | Three tiny attention/residual graphs; float64 |
| Use-identity mutations | Wrong slot and merged fan-out | Four mutations rejected after correct controls pass | Shared-producer toy and Q/K slot exchange |
| Numerical directions and summaries | Local correctness | 6,144 analytic JVP; 384 attention-detach consistency checks; 230,400 summary fields | Directional/summary checks, not fully independent component construction |

The analytic implementation is independent of the main scanner; a separate driver compares the actual scanner component by component. Nonzero asymmetric fixtures distinguish wrong slots and prematurely merged uses. Correctness is limited to the tested graphs and fault types. Arrays, per-component errors, and mutation records appear in [Supplement S17](supplement.md#v6-coordinate).

### 4.4 Computational-graph dependence and non-uniqueness

Closure in Proposition 2 is relative to a specified computational graph, cut, and internal coordinate system. An equivalent reparameterization can preserve the end-to-end function and total Endpoint while changing coordinate-wise or path-wise terms. Let $x=(x_1,x_2)^\top$, hidden state $h=x$, and output $s=(1,1)h=x_1+x_2$. A cut over the two hidden coordinates gives

$$
E_1=(x_1,0)^\top,\qquad E_2=(0,x_2)^\top.
$$

Now use the invertible transform $R=\begin{bmatrix}1&1\\0&1\end{bmatrix}$ and rewrite the same function with $h'=Rh$ and $w'=R^{-\top}(1,1)^\top=(1,0)^\top$, so $s=w'^\top h'$. The coordinate-path maps become

$$
E'_1=(x_1,x_2)^\top,\qquad E'_2=(0,0)^\top.
$$

Both implementations satisfy $E_1+E_2=E'_1+E'_2=(x_1,x_2)^\top$, but their terms differ. Changing a valid cut likewise changes the named path groups. Our Q/K/V, head, and residual maps are therefore decompositions of specified actual edges in the frozen ViT implementation, not unique explanations determined solely by the end-to-end function, and they are not generally implementation invariant. The tested cuts and mutation classes check omissions and duplication in that representation. Their terms need not be unique across equivalent graphs.

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

These scales are absolute-mass ratios of current-point local differentials, not causal contribution fractions or functional weights. Own-token locality uses Chebyshev radius 1, the one-hop neighborhood on the 14×14 patch grid; the post-hoc radius-0/1/2 sensitivity analysis checks only this descriptive choice.

Statistics use complete signed RGB tensors; zero denominators or norms yield NA. The blue display convention is described in the experimental settings.

### 4.6 Finite replacement and region-induced replacement of designated head activations

Define the finite component effect as $e_k=s(x)-s(x;z_k\leftarrow z_k^{\mathrm{ref}})$. Table 6 compares $m_k$ with $|e_k|$ by within-image Spearman correlation. Intervenable variables are attention probability $A$, value projection $V$, head output, and residual branch activation. Routing-induced and value-induced differentials supply the corresponding local maps; derivative terms are not replacement tensors.

We use five names consistently. Total Endpoint is the full-target map $E_s$. An edge-conditioned local map fixes one use in $\mathcal L_{e,:}$. A branch contribution map sums a complete branch partition. A regional map score aggregates a specified map within a candidate region. A reference-aware Taylor score combines the clean cotangent with the actual activation displacement. Tables 6–10 use the following formulas, where $k$ denotes a native use or declared differential group and $x_P$ uses reference pixels only within region $P$.

$$
E_s=x\odot\nabla_xs,\qquad
m_k=\sum_i|\mathcal L_{k,i}|,\qquad
R_k(P)=\sum_{i\in P}|\mathcal L_{k,i}|,\qquad
R_{\mathrm{total}}(P)=\sum_{i\in P}|E_{s,i}|,
$$

$$
R_{\nabla_x}(P)=\sum_{i\in P}|\partial_i s|,\qquad
T_k=\left|\langle g_k,z_k(x)-z_k^{\mathrm{ref}}\rangle\right|,\qquad
T_k(P)=\left|\langle g_k,z_k(x)-z_k(x_P)\rangle\right|.
$$

| Table | Method or quantity | Formula and information |
| --- | --- | --- |
| 6 | Component-map association | Spearman$(m_k,\lvert e_k\rvert)$ over the fixed component family within each image; clean-only map mass |
| 7 | Local-map / Taylor head rank | Rank $m_k$ / $T_k$; Capture@K is top-K absolute effect divided by the candidate top-K effect; Regret@1 is the best minus selected effect |
| 8 | Paired regional AUC difference | AUC ranked by $R_k$ minus AUC ranked by $R_{\mathrm{total}}$, $R_{\nabla_x}$, $T_k(P)$, or random order |
| 9 | Absolute isolated-effect AUC | Trapezoid over replaced area 0, 1/16, 2/16, 4/16; divide by 0.25; average heads within image |
| 10 | Candidate regret / logit ratio | $1-\mathrm{AUC}/\mathrm{AUC}_{\mathrm{oracle}}$ / $\mathrm{AUC}/\lvert s(x)\rvert$; zero denominator is NA |

The normalized-coordinate gradient baseline uses $\nabla_xs$ and is called the normalized-coordinate gradient baseline here. The raw-pixel gradient $\nabla_rs=\nabla_xs/\sigma$ is evaluated separately in coordinate sensitivity. Regional tasks fix three heads in block 5, a shared 4×4 candidate grid, and budgets of 1, 2, and 4 regions. Block 5 and each model's first, middle, and last head indices were selected before outcome access to cover positions within that block; they were not selected by effect and do not represent all heads. The mixed input supplies the selected head activation, which is inserted into a clean forward. Other heads and the identity stay clean, and all downstream nodes are recomputed. Random ranking takes absolute effects and integrates within each repeat before averaging 20 AUCs. Clean-only map scores and reference-aware Taylor use different information. The estimand is a two-stage replacement effect for designated head activations, not a total pixel intervention effect.

[Complete replacement pseudocode and variable definitions appear in Supplement S16](supplement.md#v6-intervention).

## 5. Experimental Setup and Numerical Validation

### 5.1 Models and images

Images come from the 50,000-image ImageNet ILSVRC2012 public validation set and are identified by official WNID labels, original filenames, byte counts, and SHA-256 values. After excluding prior exposures, fixed SHA-256 ordering selects one image from each of 528 classes, divided into 64 discovery, 64 calibration, and 400 evaluation images with no overlap. The 100-image unit study is a fixed evaluation subset. Each block includes CLS, 15 stratified patch tokens, and 32 channels, yielding 6,144 units per image. Correctness and confidence do not determine inclusion; misclassified and low-confidence images are retained.

The independent cohort contains one image from each of another 400 classes, disjoint from the old cohorts and concurrent 500-image cohort. Four public checkpoints include the original models and two additional training recipes. They are fixed parameter states, not independent training seeds or replication by an external team. Exact timm names, training sources, sizes, SHA-256 values, overlap matrices, and randomization rules appear in [Supplement S15](supplement.md#v6-data).

**Table 2. Fixed-model settings, descriptive scope**

| Model / setting | ViT-B/16 | DeiT-S/16 |
| --- | --- | --- |
| Blocks / heads / hidden dimension | 12 / 12 / 768 | 12 / 6 / 384 |
| Registered scalar targets/image | 457 | 385 |
| Evaluation images | 400 | 400 (same IDs) |
| Resize / crop | 248 / 224 | 248 / 224 |
| Normalization mean | [0.5, 0.5, 0.5] | [0.485, 0.456, 0.406] |
| Normalization std | [0.5, 0.5, 0.5] | [0.229, 0.224, 0.225] |

### 5.2 Visualization settings

Concept and layerwise figures use a sloth-bear image with a recognizable subject; method comparisons also include a mantis and a little blue heron. Images are selected for presentation readability, while population conclusions retain the complete cohorts. To relate learning images to forward features, each layer selects the channel with highest spatial variance across patch activations and uses its energy as the scalar target. Figures report the channel index and actual 14×14 forward feature map; selected channels may differ across layers.

Main learning images use a deep-blue, cyan, yellow and red positive view. With $P=\max(\sum_cE_c,0)$, displayed values are $\sqrt{\min(P/q_{.99},1)}$, where $q_{.99}$ is computed over positive pixels. This transform improves visibility of weak detail while preserving pixel positions and is excluded from statistics. Supplementary figures pair it with signed and absolute views and separately show a patch-max and bilinear-interpolation display variant, distinguishing native fine detail from a smoothed regional view.

### 5.3 Inferential populations and analysis identities

Images are the primary statistical units. Fixed within-image aggregation precedes 10,000 image bootstraps for pointwise median 95% intervals. The bootstrap does not generate p values or determine the resolution of small p values. One-image-per-class inference is conditional on the fixed class–image sampling rule and does not cover within-class variation. Zero image overlap between E5 and earlier cohorts does not imply zero class overlap. Intervals condition on checkpoints, target sets, references, and replacement contracts, without direct generalization to new neurons, heads, or architectures. Separate sensitivity analyses jointly resample target strata and images. The four-checkpoint check concerns fixed public models. Holm correction does not make the intervals simultaneous.

The original component and regional study jointly adjusts 28 tests. Unit depth has five tests per model, attention four per model, and residuals two per model and type. Four post-hoc residual-TV tests form a separate Holm family. Independent-cohort H1–H4 tests adjust four hypotheses within each checkpoint. Membership, freeze times, and identities appear in [Supplement S15](supplement.md#v6-data). These local plans were frozen before corresponding outcome access, without external preregistration.

| Analysis | Unit | Image-level value | Interval | Test | Sidedness | Zero handling | Multiplicity | Identity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original component association | image | Within-image Spearman$(m_k,|e_k|)$ | 10,000-replicate percentile interval for the median | Exact sign test | two-sided | $|v|\le10^{-12}$ is a tie; retained in total $n$ and interval, excluded from the binomial count | Joint 28-test Holm family with regional tests | Primary analysis locally frozen before outcome access |
| Original regional comparison | image | Paired AUC difference averaged over three fixed heads | same | Exact sign test | two-sided | same | Same 28-test Holm family | Primary analysis locally frozen before outcome access |
| H1–H4 | image | Fixed within-image hypothesis aggregate | same | Exact sign test | two-sided | same | Four-test Holm family within each checkpoint | Prespecified robustness analysis |
| v7 stratified diagnostics | image | Type-specific, block-specific, or residualized correlation | same | Exact sign test when reported | two-sided | same | Separate post-hoc Holm family | Post-hoc diagnostic |

The exact sign test evaluates equal probabilities of positive and negative signs among non-ties. It is a direct test of sign probability and is equivalent to a median-zero test only under the corresponding distributional conditions. Main-text p-value columns report Holm-adjusted values; machine evidence retains full-precision raw and adjusted values. The display $p<10^{-6}$ is used only when the adjusted value is below that bound.

The 0.10 correlation and 0.05 regional-gain thresholds are researcher-selected operational thresholds. The original composite criterion requires complete attempted coverage, at least 360 valid images, at least 300 nonzero differences, CI width no greater than 0.30/0.50, a positive estimate meeting the threshold, and original Holm p no greater than 0.05. Regional support requires all four baselines to pass. This differs from requiring the CI lower bound to exceed the threshold. Correlation sensitivity yields 12/11/9 qualifying point estimates at thresholds 0.05/0.10/0.15 versus 12/10/7 lower bounds. None of eight regional gains against Taylor or the corrected absolute-effect random baseline meets 0.025, 0.05, or 0.10. Sensitivity does not alter the original frozen decisions.

Table 3 reports errors and coverage of fixed numerical tests without image-population inference. Original float32 failures and same-coordinate float64 checks are retained.

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

### 5.4 Result order and evidence scope

Results proceed from identity and numerical correctness to normalized-input Endpoint spatial profiles, independent-cohort checks, coordinate sensitivity, and prescribed finite replacement. Closure concerns local differential accounting, spatial distributions describe fixed checkpoints, and replacement effects depend on the specified reference and intervened variable.

## 6. Depth Patterns of Local Dependence in ViTs

### 6.1 NNS equivalence and structural-path closure

The 5,000 original-NNS-compatible comparisons in Table 3 give maximum relative L2 errors of $4.769\times10^{-16}$ in float64 and $2.427\times10^{-7}$ in float32 between layerwise learning-image propagation and same-target Input × Gradient. Structural validation covers 76,800 positive closure checks, 28,800 omission/duplication controls, 6,144 analytic JVP checks, and 88 operator directions. All positive closures and analytic directional checks pass, all negative controls are rejected, and all 36 float32 finite-difference exceedances pass same-coordinate float64 review. These results establish the numerical identity in Proposition 1 and structural accounting in Proposition 2; Section 7 evaluates finite-intervention utility separately.

This revision’s independent analytic reference passes 32 component comparisons and nine central-difference directions, with maximum component absolute error $1.11\times10^{-16}$. All four new use mutations are rejected; Supplement S17 reports the direct evidence.

### 6.2 Spatial distributions of fixed targets

Figure 5 shows normalized-input Endpoint for different targets on one image. Forward features describe token-grid responses; local maps describe the scalar’s current-point response to input-coordinate scaling. Population summaries use fixed sampled targets independently of display selection.

![Figure 5](../figures/png/fig_08_internal_targets.png)

**Figure 5.** Left-to-right, top-to-bottom: the sloth-bear input and normalized-input Endpoint for patch-embedding energy, block-0/5/11 energy, block-5 attention-projection energy, MLP-output energy, and the ground-truth class logit in DeiT-S/16. Each energy is the half-squared-sum target for the channel with highest forward spatial variance at that site. Map panels show the positive signed RGB sum, clipped at positive P99 and square-root transformed independently. One display image is shown, without intervals or population inference.

Image-level late-minus-early medians and 95% intervals for 100 images per model are listed in ViT-B/16 / DeiT-S/16 order: entropy 0.220 [0.211, 0.227] / 0.225 [0.212, 0.228]; locality -0.356 [-0.363, -0.350] / -0.342 [-0.353, -0.333]; distance 0.131 [0.129, 0.136] / 0.122 [0.120, 0.128]. Each image takes medians over its fixed units in blocks 8–11 and 0–3 before differencing. Locality and distance exclude CLS. These are normalized-input Endpoint distribution profiles, without a causal interpretation of depth.

A post-hoc low-magnitude sensitivity uses total-L1-mass quantiles defined within each model and block without selecting thresholds from entropy outcomes. After trimming the lowest 1%, 5%, and 10% of mass, median late-minus-early entropy is 0.222, 0.229, and 0.234 in ViT-B and 0.226, 0.230, and 0.233 in DeiT-S; every setting remains positive in 100/100 images. All four mass quartiles have positive median changes in both models, with 0.090 and 0.112 in the lowest quartile. At the same time, median within-image, within-block Spearman association between log1p(total mass) and entropy is −0.729 and −0.760, showing that entropy depends on map quality. The result supports a broadening direction not driven solely by the lowest-mass maps; it does not make normalized entropy magnitude-independent. Supplement S19 reports intervals, valid counts, and eight typed-NA image--quartile cells.

![Figure 6](../figures/png/fig_12_spatial_broadening.png)

**Figure 6.** Left, middle, and right panels show normalized-input Endpoint spatial entropy, own-token locality, and normalized distance. At each block, medians over the fixed token-channel targets are computed within images and then summarized by the median over 100 images from the evaluation subset. CLS enters entropy only; locality subtracts the equal-area uniform baseline. Shading shows 95% intervals from 10,000 image bootstraps. Curves correspond to two fixed checkpoints and are descriptive block profiles; late-minus-early tests belong to the five-item family frozen before outcome access.

### 6.3 Joint-index necessity

Total Gradient × Input, edge scalars, and dual-indexed map sets retain different information. Table 4 connects these representations to the observed results. Depth-wise spatial broadening requires only the total map and is therefore a background regularity. Routing and identity shares require edge-level absolute masses and serve as supporting scale results. The spatial differences between full and Value-only attention and between residual output and identity require both path identity $e$ and input position $i$; they are the core empirical results for the dual-index representation.

**Table 4. Representation information and inferential scope**

| Representation | Retained indices | Recoverable quantity | Evidence / inference scope |
| --- | --- | --- | --- |
| Total Endpoint | $i$ | Entropy, locality, distance | 100 images/model; fixed sampled targets; image-level inference |
| Component scalar | use or differential-group label | $m_k$ and relative mass | 400 images/model; fixed checkpoint and class target |
| Dual-indexed map set | label and $i$ | Q/K/V entropy; output–identity TV | 400 images/model; QKV prospective robustness; residual TV post-hoc |

The alternative-explanation audit uses the same information boundary as the claim. A total map retains $i$ but sums out $e$, so it cannot recover separate routing/value or identity/branch spatial distributions. An edge scalar retains $e$ but sums out $i$, so it cannot compute normalized patch distributions, spatial TV, or signed spatial cancellation. Total absolute mass and activation norm can explain overall scale but cannot reconstruct these position-conditional differences. Both Full–Value and output–identity TV compare normalized spatial distributions, so neither is a restatement of total mass. Spatial reorganization therefore requires $[e,i]$; routing share, identity share, and overall broadening do not independently carry the joint-index necessity claim.

### 6.4 Spatial components of the attention branch

Figure 7 separates Q-induced, K-induced, routing-induced, and value-induced differentials. Complete signed terms reconstruct attention branch contribution; positive maps are displays.

![Figure 7](../figures/png/fig_15_routing_value_examples.png)

**Figure 7.** Rows show DeiT-S/16 blocks 0 and 11 for the same sloth-bear input and ground-truth logit. Columns show the input, Q-induced, K-induced, routing-induced, and value-induced maps; routing-induced is the sum of Q and K maps. These normalized-input positive displays are independently clipped at positive P99 and square-root transformed, so color magnitudes are not comparable across panels. This is a single descriptive example without uncertainty bands.

For 400 images per model, routing-relative local differential scale rises with block index, with late-minus-early median differences 0.052 [0.047, 0.059] / 0.135 [0.128, 0.143]. The corresponding increase in Value-only L2 error relative to the full attention branch map is 0.144 [0.133, 0.152] / 0.290 [0.279, 0.301]. These scale measures require component scalars; spatial comparisons additionally require input positions.

The 400-image Q/K/V contrast is B11−B0 and is distinct from four-block phase medians. Normalized-entropy median differences with this revision’s descriptive image-bootstrap intervals are vit_b16: Q 0.027 [0.022, 0.031], K -0.059 [-0.064, -0.053], V -0.053 [-0.059, -0.047]; deit_s16: Q -0.004 [-0.008, 0.001], K -0.067 [-0.074, -0.059], V -0.067 [-0.075, -0.061]. K/V decrease in both models; Q remains model-dependent and the DeiT-S interval crosses zero.

![Figure 8](../figures/png/fig_16_attention_depth.png)

**Figure 8.** Left panel: routing-induced relative absolute mass. Middle: relative L2 error of the Value-only map against the full attention branch map. Right: JVP discrepancy over frozen directions. Each block takes the within-image median across all heads and then the median across 400 images per checkpoint. Shading shows 95% intervals from 10,000 image bootstraps. Spatial maps use normalized-input coordinates; JVP uses eight frozen directions. Curves describe fixed checkpoints; depth contrasts use each model’s family frozen before outcome access.

### 6.5 Residual identity and branch

Figure 9 shows the two residual additions for one class target. Identity uses different input sites at the two additions and the cotangent at the corresponding addition output.

![Figure 9](../figures/png/fig_17_residual_examples.png)

**Figure 9.** Rows show the attention and MLP residuals in DeiT-S/16 block 5 for one sloth-bear image and ground-truth logit. Columns show the input, identity/carry local map, branch contribution, and full-target Total Endpoint. Identity refers to different residual inputs in the two rows; complete signed maps satisfy identity plus branch equals Total Endpoint. Positive panels use independent P99 clipping and square-root scaling and are not additive. This single-image display has no statistical intervals.

Late-minus-early identity-relative-scale medians are attention -0.124 [-0.134, -0.116] / -0.045 [-0.049, -0.039]; MLP 0.180 [0.167, 0.189] / 0.178 [0.166, 0.188], with 400 images per model.

![Figure 10](../figures/png/fig_18_residual_depth.png)

**Figure 10.** Left and middle panels show identity relative absolute mass for attention and MLP residuals. Right: path cancellation between signed MLP identity and branch maps. Each image contributes one value per residual and block; curves show medians across 400 images per checkpoint. Shading shows 95% intervals from 10,000 image bootstraps. All maps use normalized-input Endpoint. Curves describe fixed checkpoints; depth contrasts use model-specific families frozen before outcome access. The later output–identity spatial-TV analysis is separately post hoc.

The separately identified post-hoc spatial-TV analysis compares normalized output and identity patch mass. Late-minus-early medians and 95% intervals are vit_b16 attention 0.052 [0.049, 0.055]; vit_b16 mlp -0.019 [-0.022, -0.015]; deit_s16 attention 0.028 [0.026, 0.030]; deit_s16 mlp -0.046 [-0.052, -0.041]. Four tests pass their separate Holm family. A total map lacks identity assignment and an edge scalar lacks position, so this comparison needs both indices.

### 6.6 Independent-cohort and four-checkpoint robustness

The independent cohort tests four prospectively frozen hypotheses on 400 images with zero overlap with the original cohorts and the concurrent 500-image cohort. H1 is the block-11 minus block-0 entropy difference across 32 post-block neurons; H2 is the change in routing mass relative to Q/K-routing plus V mass; H3 subtracts the attention-residual identity trajectory from the MLP-residual identity trajectory; H4 is the path-map head-ranking correlation increment over activation × gradient, averaged across four representative blocks. Table 5 reports image-level medians and 95% bootstrap intervals.

**Table 5. Independent-cohort four-checkpoint check; median [95% CI]**

| Checkpoint | H1 | H2 | H3 | H4 | Inference |
| --- | --- | --- | --- | --- | --- |
| vit_b16 | 0.413 [0.405, 0.420] | 0.239 [0.221, 0.259] | 0.851 [0.839, 0.865] | -0.114 [-0.128, -0.091] | 400 images; fixed checkpoint; prospective robustness |
| deit_s16 | 0.378 [0.370, 0.383] | 0.322 [0.310, 0.329] | 0.522 [0.495, 0.543] | -0.186 [-0.229, -0.157] | 400 images; fixed checkpoint; prospective robustness |
| vit_b16_in1k | 0.336 [0.330, 0.340] | 0.241 [0.218, 0.255] | 0.632 [0.620, 0.648] | -0.186 [-0.205, -0.168] | 400 images; fixed checkpoint; prospective robustness |
| vit_s16_augreg | 0.390 [0.384, 0.399] | 0.276 [0.256, 0.296] | 0.867 [0.840, 0.891] | -0.136 [-0.171, -0.114] | 400 images; fixed checkpoint; prospective robustness |

H4 uses the ground-truth logit and normalized-zero reference. For image $j$, block $b$, and head $h$, let $z_{jbh}$ denote the head-channel slice at the input to `attn.proj`, and define

$$
M_{jbh}=\sum_i|\mathcal L_{jbh,i}|,\qquad
G^{\mathrm{AG}}_{jbh}=|\langle z_{jbh},\nabla_{z_{jbh}}s_j\rangle|,
$$

$$
Y_{jbh}=|s_j(x)-s_j(x;z_{jbh}\leftarrow z_{jbh}^{\mathrm{normalized\text{-}zero}})|,
\qquad
H4_j=\frac14\sum_{b\in\{0,3,7,11\}}[\rho_h(M_{jbh},Y_{jbh})-\rho_h(G^{\mathrm{AG}}_{jbh},Y_{jbh})].
$$

$G^{\mathrm{AG}}$ takes the inner product over the complete head tensor before the absolute value; it is neither a sum of elementwise absolute values nor the attention probability $A$. Spearman uses average ranks. Fewer than four valid heads, nonfinite values, or a constant vector yield NA. The two correlations are differenced within heads at each block, followed by an arithmetic mean over blocks 0, 3, 7, and 11. The final median and bootstrap interval use 400 images; the two-sided exact sign test enters the four-test H1–H4 Holm family within each checkpoint.

H1–H3 are positive at all four checkpoints, with within-checkpoint Holm-adjusted $p<10^{-6}$. H4 has the opposite direction at every checkpoint: within-image head-ranking correlation is 0.114–0.186 lower for path maps than for activation × gradient, with Holm-adjusted $p<10^{-6}$. The spatial-broadening, routing, and residual-trajectory directions remain aligned on the independent image cohort and additional public checkpoints; stable path-map head-ranking gain is not supported. The added weights are not independent training seeds.

Dual indexing increases structural resolution and auditability. H4 does not outperform activation × gradient, so additional structure does not automatically improve finite-intervention prediction.

The original H4 path score takes absolute values before summing over input coordinates, whereas its activation × gradient comparator sums over the complete head tensor before taking an absolute value. An aggregation-matched sensitivity instead uses $M^{\mathrm{sum}}_{jbh}=|\sum_i\mathcal L_{jbh,i}|$ while preserving the target, reference, blocks, effect, and image-level aggregation. Median increments remain negative at all four checkpoints: vit_b16 −0.124 [−0.147, −0.102], deit_s16 −0.114 [−0.157, −0.079], vit_b16_in1k −0.226 [−0.253, −0.205], and vit_s16_augreg −0.171 [−0.200, −0.143]. Aggregation order therefore does not explain the original negative H4 direction. This post-hoc analysis does not replace the registered estimator.

### 6.7 Coordinate parameterization and controls

This revision compares normalized-input Endpoint, raw-pixel gradient, and RGB-zero-origin weighted gradient at the same input point. Coverage is 100 fixed-unit images and 400 structural images per model, paired by image. All 48 phase-summary groups have complete image coverage and consistent main directions. For B11−B0, however, DeiT-S Q medians are −0.004, 0.003, and 0.002; all three intervals cross zero and point-estimate signs change. All 96 median differences, intervals, and directions appear in [Supplement S17](supplement.md#v6-coordinate). Main conclusions remain restricted to normalized-input Endpoint, with object-specific direction retention reported separately as robustness evidence.

**Table 5a. Eight core coordinate results; late-minus-early median difference (ViT-B / DeiT-S).** The columns are normalized-input Endpoint, raw-pixel gradient, and zero-origin raw-pixel weighted gradient; they are not three forms of Endpoint.

| Object / metric | Normalized-input Endpoint | Raw-pixel gradient | Zero-origin raw-pixel weighted gradient |
| --- | --- | --- | --- |
| Unit entropy | 0.220 / 0.225 | 0.240 / 0.245 | 0.217 / 0.221 |
| Unit own-token locality | -0.356 / -0.342 | -0.367 / -0.359 | -0.356 / -0.356 |
| Unit normalized distance | 0.131 / 0.122 | 0.138 / 0.129 | 0.134 / 0.127 |
| Q entropy | 0.037 / 0.018 | 0.039 / 0.018 | 0.034 / 0.015 |
| K entropy | -0.021 / -0.028 | -0.019 / -0.029 | -0.019 / -0.027 |
| V entropy | -0.010 / -0.013 | -0.009 / -0.016 | -0.010 / -0.015 |
| Attention output–identity TV | 0.052 / 0.028 | 0.054 / 0.028 | 0.053 / 0.027 |
| MLP output–identity TV | -0.019 / -0.046 | -0.017 / -0.044 | -0.018 / -0.039 |

Table 5a reports the phase contrast. The coordinate-sensitive DeiT-S Q exception uses the distinct B11−B0 estimand, so the two contrasts are not interchangeable.

Trained versus random initialization, GT versus alternative targets, forward-feature matching, target-stratum/image resampling, and visual method comparisons are collected in the [supplementary experiments](supplement_experiments_v1.md). Random parameters also exhibit broadening; the parameter-state comparison does not identify a causal effect of the training process. Cross-layer and all-head regional replacements remain supplementary, separate from the original locally prespecified block-5 analysis.

## 7. Ranking in Prescribed Finite-Replacement Tasks

### 7.1 Prescribed component replacement and region-induced designated-head activation replacement

The original 12 pooled component families remain available as frozen provenance, with continuous correlations, intervals, and Holm-adjusted p values in the supplement. Eleven of twelve met the original composite criterion, while the all-family claim remained unsupported. Table 6 instead prioritizes post-hoc A/V and attention/MLP stratified continuous estimates and type/depth mixing diagnostics, avoiding a single summary for distinct operations.

**Table 6. Component association by type and depth diagnostic; image-level median [95% CI]**

| Model | Reference | A-only | V-only | Attention branch | MLP branch | Type+block-adjusted A/V | Type+block-adjusted residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b16 | normalized_zero | 0.049 [0.007, 0.091] | 0.224 [0.192, 0.259] | -0.028 [-0.056, -0.003] | 0.287 [0.266, 0.318] | 0.030 [-0.003, 0.071] | -0.128 [-0.164, -0.084] |
| vit_b16 | blur_rgb | 0.073 [0.028, 0.112] | 0.311 [0.266, 0.350] | 0.000 [-0.021, 0.042] | 0.378 [0.336, 0.402] | -0.018 [-0.063, 0.016] | 0.042 [-0.007, 0.083] |
| deit_s16 | normalized_zero | 0.091 [0.049, 0.119] | 0.035 [-0.021, 0.087] | -0.157 [-0.189, -0.119] | 0.378 [0.346, 0.406] | 0.089 [0.056, 0.120] | -0.091 [-0.120, -0.059] |
| deit_s16 | blur_rgb | 0.101 [0.056, 0.133] | 0.147 [0.112, 0.182] | -0.024 [-0.070, 0.014] | 0.371 [0.332, 0.395] | 0.095 [0.055, 0.128] | 0.142 [0.096, 0.176] |

A-only, V-only, attention-branch, and MLP-branch correlations are computed across 12 blocks within each image. The last two columns separately rank the 24 predictor and effect observations and then adjust both for operation and block; all results are post-hoc diagnostics. A/V directions and magnitudes depend on model and reference. MLP branches drive the pooled residual association, whereas attention-branch intervals include zero in three settings or are negative. After joint type and block adjustment, correlations are near zero or negative for both ViT-B references and change direction across references for DeiT-S. The historical pooled correlation therefore describes a mixed family and does not establish uniform within-type ranking for A, V, attention, or MLP operations. Supplement S18 reports every block, one-factor adjustment, Fisher-z head aggregation, A/V concordance, scale diagnostics, and pooled provenance.

In the supplementary within-block head comparison, Taylor achieves higher Spearman and Capture and lower Regret (Table 7). Across-layer residual identity has Endpoint-minus-Taylor Spearman differences of 0.086–0.421 but a zero median Capture@1 difference, indicating task- and metric-dependent incremental value.

**Table 7. Within-block head ranking; median of image means [95% CI]**

| Model | Reference | Score | rho | Capture@1 | Capture@3 | Regret@1 | Inference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b16 | normalized_zero | local_map | 0.281 [0.272, 0.297] | 0.517 [0.504, 0.535] | 0.615 [0.609, 0.625] | 0.170 [0.161, 0.182] | 400 images; fixed heads; exploratory |
| vit_b16 | normalized_zero | taylor | 0.674 [0.666, 0.684] | 0.804 [0.796, 0.812] | 0.857 [0.851, 0.862] | 0.057 [0.052, 0.062] | 400 images; fixed heads; exploratory |
| vit_b16 | blur_rgb | local_map | 0.273 [0.259, 0.286] | 0.507 [0.495, 0.520] | 0.618 [0.610, 0.626] | 0.116 [0.112, 0.123] | 400 images; fixed heads; exploratory |
| vit_b16 | blur_rgb | taylor | 0.686 [0.676, 0.694] | 0.820 [0.808, 0.835] | 0.866 [0.860, 0.871] | 0.035 [0.032, 0.037] | 400 images; fixed heads; exploratory |
| deit_s16 | normalized_zero | local_map | 0.195 [0.176, 0.214] | 0.561 [0.546, 0.574] | 0.712 [0.705, 0.721] | 0.238 [0.208, 0.266] | 400 images; fixed heads; exploratory |
| deit_s16 | normalized_zero | taylor | 0.610 [0.600, 0.629] | 0.803 [0.796, 0.812] | 0.889 [0.882, 0.895] | 0.086 [0.079, 0.092] | 400 images; fixed heads; exploratory |
| deit_s16 | blur_rgb | local_map | 0.148 [0.133, 0.162] | 0.551 [0.539, 0.566] | 0.710 [0.702, 0.716] | 0.133 [0.117, 0.146] | 400 images; fixed heads; exploratory |
| deit_s16 | blur_rgb | taylor | 0.624 [0.610, 0.638] | 0.843 [0.835, 0.852] | 0.897 [0.893, 0.902] | 0.044 [0.039, 0.048] | 400 images; fixed heads; exploratory |

The region-induced designated-head activation replacement experiment fixes three heads in block 5, a shared 4×4 region pool and a maximum 25% area. Figure 11 and Tables 8–10 report paired differences, absolute AUC and normalized regret. Paired median Endpoint-minus-regional-Taylor differences are −0.023, −0.024, −0.030 and −0.036. None of the four families meets the +0.05 criterion over every baseline. The candidate-set envelope maximizes effects only over evaluated sets and is not a global optimum over all regions. The clean-only local map shows no stable or practically useful ranking gain in this task. Taylor performs better but uses actual displacement, has a different information budget, and retains a gap to the candidate-set envelope.

![Figure 11](../figures/png/fig_21_intervention_boundary.png)

**Figure 11.** Left: the DeiT-S/16 sloth-bear input. Middle: mean absolute isolated effects across three fixed block-5 heads under normalized-zero reference, at replaced areas 0, 1/16, 2/16, and 4/16. Right: image-level median absolute-effect AUC for four model-by-reference families, each n=400, after averaging the three heads within images. Bars have no uncertainty bands; Tables 8–10 report 95% intervals. This is a descriptive display of fixed tasks. The candidate envelope takes the best evaluated effect separately at each area and is not a single globally optimal policy.

**Table 8. Local-map regional AUC minus baseline; median difference [95% CI]**

| Model | Reference | vs Total Endpoint regional score | vs normalized-coordinate gradient | vs reference-aware Taylor | vs random | Original composite | Inference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b16 | normalized_zero | -0.001 [-0.002, -0.0002] | $-3.1\times10^{-5}$ [-0.001, 0.0003] | -0.023 [-0.026, -0.022] | -0.002 [-0.003, -0.001] | Unmet | 400 paired images; fixed 3 heads; descriptive intervals |
| vit_b16 | blur_rgb | -0.001 [-0.002, -0.0002] | -0.001 [-0.002, $3.1\times10^{-5}$] | -0.024 [-0.027, -0.022] | -0.003 [-0.005, -0.002] | Unmet | 400 paired images; fixed 3 heads; descriptive intervals |
| deit_s16 | normalized_zero | -0.002 [-0.003, 0.000] | -0.001 [-0.002, 0.001] | -0.030 [-0.034, -0.026] | 0.001 [-0.001, 0.003] | Unmet | 400 paired images; fixed 3 heads; descriptive intervals |
| deit_s16 | blur_rgb | 0.0002 [-0.0004, 0.001] | -0.0002 [-0.001, 0.0005] | -0.036 [-0.039, -0.031] | -0.001 [-0.002, 0.001] | Unmet | 400 paired images; fixed 3 heads; descriptive intervals |

**Table 9. Absolute-effect AUC and logit; median [95% CI]**

| Model | Reference | Local-map regional AUC | Taylor AUC | Random AUC | Candidate-envelope AUC | Clean GT logit | Inference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b16 | normalized_zero | 0.031 [0.029, 0.032] | 0.056 [0.053, 0.058] | 0.034 [0.031, 0.036] | 0.065 [0.063, 0.070] | 8.899 [8.842, 8.942] | 400 images; fixed 3 heads; descriptive |
| vit_b16 | blur_rgb | 0.028 [0.026, 0.029] | 0.052 [0.049, 0.055] | 0.031 [0.028, 0.033] | 0.075 [0.068, 0.079] | 8.899 [8.841, 8.943] | 400 images; fixed 3 heads; descriptive |
| deit_s16 | normalized_zero | 0.062 [0.056, 0.068] | 0.096 [0.087, 0.103] | 0.061 [0.057, 0.065] | 0.147 [0.137, 0.160] | 8.797 [8.691, 8.856] | 400 images; fixed 3 heads; descriptive |
| deit_s16 | blur_rgb | 0.039 [0.035, 0.043] | 0.077 [0.068, 0.082] | 0.041 [0.038, 0.044] | 0.106 [0.098, 0.114] | 8.797 [8.691, 8.856] | 400 images; fixed 3 heads; descriptive |

**Table 10. Candidate-envelope regret and logit ratios; median [95% CI]**

| Model | Reference | Local-map regional regret | Taylor regret | Random regret | Local-map AUC/abs(logit) | Taylor AUC/abs(logit) | Inference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b16 | normalized_zero | 0.530 [0.508, 0.555] | 0.144 [0.132, 0.155] | 0.506 [0.494, 0.524] | 0.004 [0.003, 0.004] | 0.006 [0.006, 0.007] | 400 images; fixed 3 heads; descriptive |
| vit_b16 | blur_rgb | 0.637 [0.607, 0.655] | 0.256 [0.244, 0.280] | 0.593 [0.585, 0.600] | 0.003 [0.003, 0.003] | 0.006 [0.006, 0.006] | 400 images; fixed 3 heads; descriptive |
| deit_s16 | normalized_zero | 0.565 [0.540, 0.586] | 0.335 [0.317, 0.357] | 0.595 [0.583, 0.601] | 0.007 [0.006, 0.008] | 0.011 [0.010, 0.012] | 400 images; fixed 3 heads; descriptive |
| deit_s16 | blur_rgb | 0.629 [0.600, 0.640] | 0.271 [0.249, 0.295] | 0.615 [0.607, 0.624] | 0.005 [0.004, 0.005] | 0.009 [0.008, 0.010] | 400 images; fixed 3 heads; descriptive |

With the corrected random estimator, paired Endpoint-minus-random medians are −0.002, −0.003, 0.001 and −0.001; both DeiT-S intervals include zero. All four Taylor-minus-random intervals are positive. The block-5 three-head task follows the original locally prespecified protocol; the supplementary cross-layer all-head analysis covers blocks 2/5/10 and favors Taylor in 11/12 full-input and 12/12 isolated-head families (Supplement S14).

### 7.2 Behavioral calibration and estimand boundary

Cross-reference analysis clarifies this distinction. Endpoint uses the clean observed-input map, whereas Taylor explicitly uses replacement displacement. Across 163,200 component pairs, Taylor magnitude and absolute effect change under the original atol=1e-12, rtol=0 criterion. Their distributions and head-rank concordance are reported in supplementary tables. Reference choice changes intervention-related ordering, so local input maps and displacement estimators serve different questions.

## 8. Discussion and Conclusion

### 8.1 Discussion

Dual-index maps permit direct comparison of spatial distributions for specified uses or differential groups. Total maps describe overall broadening, component scalars describe scale, and Q/K/V entropy and residual output–identity TV require both path labels and input positions. Global cuts include bypass; branch partitions reconstruct only attention contribution. Distinct reconstruction targets keep accounting and interpretation consistent.

Architecture-graph certification applies this contract to actual target--cut pairs instead of inferring graph structure from pointwise numerical closure. The three registries separately constrain scalar readouts, local-differential decompositions, and finite-replacement variables, avoiding one ambiguous category of registered objects.

Empirical conclusions concern normalized-input Endpoint for fixed checkpoints and sampled targets. Main directions of unit broadening, K/V entropy, and residual TV persist across the three tested input representations, while DeiT-S Q B11−B0 shows coordinate and interval sensitivity. Equivalent hidden-coordinate transformations can also change components while preserving total Endpoint. These maps therefore describe the local differential structure of a specified implementation.

H4 is lower than activation × gradient at all four checkpoints. Dual-index maps increase structural resolution and auditability without automatically improving finite-intervention prediction. Taylor directly uses replacement displacement and better ranks the prescribed regional task. Historical pooled component correlations are positive, but the v7 stratification shows that A/V and attention/MLP are not homogeneous families; type-and-block-adjusted correlations depend on model and reference. The original all-family claim remains unsupported, and the behavioral evidence is broad but non-universal.

Image intervals condition on fixed targets and checkpoints rather than neuron or architecture populations. The independent image cohort uses the same team and code; public-checkpoint differences do not identify a causal effect of the training process. Random parameters also exhibit spatial broadening. Positive displays aid spatial reading; numerical inference uses complete signed/absolute arrays. The historical ViT-B Path-IG calibration failure is retained, and no cross-model Endpoint–Path inference is made.

### 8.2 Method limitations

A current-point first-order Jacobian describes local response and does not capture nonlinear interactions under large displacements. Components depend on the specified computational graph, internal coordinates, and cut and are not generally implementation invariant. Small gradients, saturation, and normalization can change spatial-shape metrics; normalized TV is not functional importance. The empirical scope is ImageNet, patch-16 ViTs, fixed scalar targets, and prescribed replacement contracts. Vector targets require a prespecified scalar functional or coordinate-wise analysis; sequence models and other architectures were not evaluated. Differences in training, scale, and data across checkpoints are not controlled, so the causes of model heterogeneity remain unidentified.

### 8.3 Conclusion

Under exact current-point derivative propagation, a scalar NNS learning image equals Gradient × Input. The technical contribution is a target-conditioned dual-index representation, distinct branch/global closure contracts, and structural and numerical audits of actual ViT cuts and concrete-use errors. Normalized-input Endpoint on fixed sampled targets exhibits block-index-associated spatial profiles. The broadening direction persists under total-mass stratification and trimming, although entropy values depend on map quality. Q components are more sensitive to model and input coordinates, and path maps under both original and matched aggregation do not outperform activation × gradient for ranking finite head replacements. Prescribed replacement associations are broad but non-universal and do not support a unified pooled conclusion. Greater structural resolution and auditability do not automatically yield finite-intervention prediction gains.

## References and Supplementary Material

References follow below, and the supplement accompanies the manuscript.
