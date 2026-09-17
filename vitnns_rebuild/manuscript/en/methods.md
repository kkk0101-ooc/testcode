# 3. Methods

## 3.1 Problem definition and learning image

Let $x\in\mathbb{R}^{3\times224\times224}$ be a normalized image and $x_0$ the all-zero reference in normalized space, corresponding to RGB gray 0.5. For a scalar target $s(x)$ in the model, we define its learning image as

$$
L_s[i]=(x_i-x_{0i})\int_0^1
\frac{\partial s(x_0+\alpha(x-x_0))}{\partial x_i}\,d\alpha.
$$

Up to numerical integration error,

$$
\sum_i L_s[i]\approx s(x)-s(x_0).
$$

The raw learning image retains three RGB channels and their signs. A two-dimensional map is formed by signed summation over RGB. Positive-part views, negative-part views, signed-log transforms, P99 clipping, and color maps are display operations only; they do not enter completeness, residual additivity, or statistical calculations. Figures 15–17 describe coordinate initialization and image composition, while Figure 2 shows a learning image for a real attention response.

## 3.2 ViT forward structure

The frozen model receives a $224\times224$ RGB image, forms 196 image tokens from $16\times16$ patches, and adds a CLS token and position embeddings. Transformer block $b$ is written as

$$
a_b=\operatorname{Attention}(\operatorname{LN}_1(h_b)),\qquad
r_b^{(1)}=h_b+a_b,
$$

$$
m_b=\operatorname{MLP}(\operatorname{LN}_2(r_b^{(1)})),\qquad
h_{b+1}=r_b^{(2)}=r_b^{(1)}+m_b.
$$

After 12 blocks,

$$
u=\operatorname{FinalNorm}(h_{12}),\qquad
\ell=W_{\mathrm{cls}}u_{\mathrm{CLS}}+b_{\mathrm{cls}}.
$$

The block-0 MLP has dimensions $768\rightarrow3072\rightarrow768$ with GELU. The classifier is $\operatorname{Linear}(768,1000)$ applied to the final-normalized CLS token; it is distinct from the MLP branch inside a block. Figures 18 and 24–27 connect the normal forward pass to learning images in analytic examples. Figures 28–35 use real model outputs.

## 3.3 Three target classes

### 3.3.1 Module-own responses

For a patch-embedding, attention, MLP, or residual output $q(x)\in\mathbb{R}^{196\times768}$, the display target for channel $c$ is

$$
s_{q,c}(x)=\sum_{t=1}^{196}q(x)_{t,c}.
$$

CLS participates in forward attention but is excluded from this spatial-token sum. $L_{s_{q,c}}$ identifies the input locations associated with the module response; it contains no downstream class derivative. Figures 29–33 use fixed channels 0, 255, 511, and 767, selected by index before viewing the maps.

### 3.3.2 Fixed-clean-attention conditional Value-path target

The primary intervention experiment uses the CLS head-direction for block $b$ and head $h$. We capture attention $A_{\mathrm{clean}}$ on the clean forward pass and define

$$
g(z)=\sum_{j=0}^{196}A_{\mathrm{clean}}[\mathrm{CLS},j]V(z)_j,
$$

$$
d=g(x)-g(x_0),\qquad u=\frac{d}{\lVert d\rVert_2},\qquad
f(z)=u^\top g(z).
$$

$A_{\mathrm{clean}}$, $d$, and $u$ are detached and frozen before integration. The target retains the input-dependent Value path but cuts variation through Q, K, and softmax attention along the integration path. We therefore call it fixed-clean-attention conditional Value-path attribution. It is neither complete Q/K/V attribution nor a complete class explanation.

### 3.3.3 Complete class outputs and complete-cut components

A complete class learning image sets $s(x)=\ell_y(x)$, with gradients through all blocks, final normalization, CLS selection, and the classifier. A complete cut selects non-overlapping internal components $z_k$ that cover the computation and combines their input-direction derivatives with downstream class derivatives to form component maps $K_k$. Within one cut,

$$
\sum_k K_k[i]\approx L_{\ell_y}[i].
$$

Different cuts are alternative accounts of the same total attribution and are not summed across cuts. Figures 19–21 show head and branch components and their reconstruction. Figure 34 directly explains a complete class logit.

## 3.4 Numerical correspondence for attention and residual paths

The derivative of a complete attention response contains both a Value term and a routing term:

$$
d(AV)=A\,dV+(dA)V.
$$

The fixed-clean-attention conditional target removes the second term, whereas real module responses and full class IG retain both. Figure 22 illustrates the distinction with a two-token example.

For a common input path and reference, a residual addition is additive at every input coordinate:

$$
L_{h+a}=L_h+L_a,\qquad
L_{r^{(1)}+m}=L_{r^{(1)}}+L_m.
$$

We test these identities on raw signed RGB arrays before applying display transforms. Figure 23 gives an analytic downstream-target example, and Figures 31–33 show the two residual additions of a real block 0.

## 3.5 Target-matched patch-token baseline

The primary comparator is the target-matched patch-token baseline with uniform within-patch allocation. For image patch token $j$,

$$
c_j=A_{\mathrm{clean}}[\mathrm{CLS},j]\,
u^\top\left(V(x)_j-V(x_0)_j\right).
$$

The 196 values form a $14\times14$ token-space map; the CLS scalar is retained separately. For evaluation, each $16\times16$ patch is divided into sixteen $4\times4$ cells and the baseline assigns

$$
c_{j,k}^{\mathrm{uniform}}=\frac{c_j}{16},\qquad k=1,\ldots,16.
$$

NNS and the baseline share the model, image, internal target, reference, intervention fractions, and tie seeds. Their coordinate definitions differ, and equality of per-patch totals is not assumed. The comparison asks whether input attribution or a uniform spatial expansion of token contribution provides the more useful intervention ranking for the same internal target. This baseline is distinct from the Stage S matched occlusion control, which matches the number, location, and shape of occluded patches.

## 3.6 Intervention and fidelity estimand

For target $t$, let $\Delta=t(x)-t(x_0)$, $\sigma=\operatorname{sign}(\Delta)$, and

$$
S(z)=\sigma\frac{t(z)-t(x_0)}{|\Delta|}.
$$

The ranking score is the positive part of $\sigma M$. Insertion restores RGB spatial units from the reference in ranking order, whereas deletion replaces units in the clean input with the reference. Both curves use fractions $0,0.05,\ldots,1$, with the actual fraction of replaced units as the horizontal coordinate. Responses are not clipped for AUC calculation.

For image $i$, method $m$, direction $d$, and tie repeat $q$, we first average five paired repeats to obtain $\bar A(i,m,d)$. The image-level contrast is

$$
\Delta_{\mathrm{fid}}(i)=\frac{1}{2}\left[
\bar A(i,\mathrm{NNS},\mathrm{ins})-\bar A(i,\mathrm{base},\mathrm{ins})
+\bar A(i,\mathrm{base},\mathrm{del})-\bar A(i,\mathrm{NNS},\mathrm{del})
\right].
$$

A positive value favors the NNS ranking under the paired insertion/deletion definition. The statistical unit is the image; repeats, curve points, cells, and methods do not increase the sample size.

## 3.7 Display rules

All numerical claims use raw arrays. Positive blue maps use the positive part and per-panel P99 scaling with a maximum fallback. Signed views use symmetric limits or a signed-log transform. NNS and baseline panels share a scale when magnitudes are compared at the same coordinate. Independently scaled panels support comparisons of hotspot location and spatial organization, not color magnitude across panels. The four classifier targets in Figure 35 use one shared maximum-absolute signed-log scale.
