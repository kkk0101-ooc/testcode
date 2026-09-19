# Captions

## Figure 34

Six registered scalar targets for the fixed first DeiT-S/16 evaluation image. The blue-base main panels show only positive values with per-panel P99 scaling; they are spatial reading views, not cross-target magnitude comparisons.

## Figure 35

Identity, head, routing, value, attention, MLP, and complete Endpoint maps for the same image, ground-truth class target, and block 5. Every panel comes directly from raw component arrays that passed closure validation.

## Figure 36

Total G×I, raw gradient, and three registered head-component maps used by the fixed RQ4 task. These panels show the fine-grained spatial form of the ranking inputs; finite-intervention curves determine the behavioral result.

## Figure 37

Following the cross-layer organization of NNS Figure 13, this figure tracks a fixed DeiT-S/16 input from patch embedding through Blocks 0–11 to final CLS and the class logit. The bottom row is 14×14 absolute mass pooled from the same raw Endpoint maps over native 16×16 input patches; it is not a forward feature map.

## Figure 38

All six attention-head edge contributions to the same ground-truth class logit for a fixed DeiT-S/16 input at Blocks 0, 1, 5, and 11. Rows compare depth and columns compare head index; display scaling is independent per panel.

## Figure 39

All twelve attention-head edge contributions to the same ground-truth class logit for a fixed ViT-B/16 input at Blocks 0, 1, 5, and 11. Complete 12-block by 12-head pages are retained under all_heads_vit_b16/.

## Figure 40

For a fixed ViT-B/16 input and channel 0, this figure shows patch embedding; Norm1, attention projection, Norm2, MLP fc1, GELU, and fc2 at Blocks 0, 5, and 11; final CLS; and the class logit. It follows the module-chain reading style of NNSViT-IG Figures 28–34 while using Endpoint scalar targets.
