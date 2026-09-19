# Endpoint fine-grained blue-base paper figures

This is a CPU-only display package derived from sealed E2, E3, and E5 raw arrays.
It adds no model execution, attribution recomputation, statistical test, or scientific decision.

The main panels follow the R10 reading style: RGB channels are summed with sign,
only `max(L, 0)` is shown, each panel uses P99 scaling with a max fallback, and deep
blue denotes zero or low positive value. Signed companions are retained beside the main
figures. Numerical claims continue to use the sealed raw arrays.

The Endpoint targets here are scalar internal or class targets with direct `3×224×224`
input-coordinate maps. Unlike the path-paper patch-embedding figure, they do not have a
corresponding `14×14` output-token response row for the same scalar target.

## Figures

### Figure 34: Endpoint target atlas: real fine-grained learning images

Six registered scalar targets for the fixed first DeiT-S/16 evaluation image. The blue-base main panels show only positive values with per-panel P99 scaling; they are spatial reading views, not cross-target magnitude comparisons.

[Main PNG](figures_png/figure_34_registered_target_learning_images.png) · [Signed companion](signed_companions_png/figure_34_registered_target_learning_images_signed.png)

### Figure 35: Block 5: real structural Endpoint learning images

Identity, head, routing, value, attention, MLP, and complete Endpoint maps for the same image, ground-truth class target, and block 5. Every panel comes directly from raw component arrays that passed closure validation.

[Main PNG](figures_png/figure_35_block5_structural_learning_images.png) · [Signed companion](signed_companions_png/figure_35_block5_structural_learning_images_signed.png)

### Figure 36: Block 5: real maps used for spatial ranking

Total G×I, raw gradient, and three registered head-component maps used by the fixed RQ4 task. These panels show the fine-grained spatial form of the ranking inputs; finite-intervention curves determine the behavioral result.

[Main PNG](figures_png/figure_36_spatial_ranking_learning_images.png) · [Signed companion](signed_companions_png/figure_36_spatial_ranking_learning_images_signed.png)

### Figure 37: Endpoint-NNSViT across depth

Following the cross-layer organization of NNS Figure 13, this figure tracks a fixed DeiT-S/16 input from patch embedding through Blocks 0–11 to final CLS and the class logit. The bottom row is 14×14 absolute mass pooled from the same raw Endpoint maps over native 16×16 input patches; it is not a forward feature map.

[Main PNG](figures_png/figure_37_channel0_depth_progression.png) · [Signed companion](signed_companions_png/figure_37_channel0_depth_progression_signed.png)

### Figure 38: Endpoint-NNSViT across heads and depth

All six attention-head edge contributions to the same ground-truth class logit for a fixed DeiT-S/16 input at Blocks 0, 1, 5, and 11. Rows compare depth and columns compare head index; display scaling is independent per panel.

[Main PNG](figures_png/figure_38_attention_head_depth_atlas.png) · [Signed companion](signed_companions_png/figure_38_attention_head_depth_atlas_signed.png)

### Figure 39: ViT-B/16 Endpoint across heads and depth

All twelve attention-head edge contributions to the same ground-truth class logit for a fixed ViT-B/16 input at Blocks 0, 1, 5, and 11. Complete 12-block by 12-head pages are retained under all_heads_vit_b16/.

[Main PNG](figures_png/figure_39_vit_b16_attention_head_depth_atlas.png) · [Signed companion](signed_companions_png/figure_39_vit_b16_attention_head_depth_atlas_signed.png)

### Figure 40: Endpoint-NNSViT through the real module chain

For a fixed ViT-B/16 input and channel 0, this figure shows patch embedding; Norm1, attention projection, Norm2, MLP fc1, GELU, and fc2 at Blocks 0, 5, and 11; final CLS; and the class logit. It follows the module-chain reading style of NNSViT-IG Figures 28–34 while using Endpoint scalar targets.

[Main PNG](figures_png/figure_40_vit_b16_module_chain.png) · [Signed companion](signed_companions_png/figure_40_vit_b16_module_chain_signed.png)
