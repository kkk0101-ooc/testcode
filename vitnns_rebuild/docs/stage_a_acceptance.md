# Stage A acceptance contract

Stage A is complete only when the canonical state-dict model, official labels,
native extractor, and provenance all pass together.

## Numerical gates

1. The state dict has SHA-256
   `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`
   and loads with `strict=True`.
2. The model has 12 blocks, 12 heads, head dimension 64, 197 tokens, and
   86,567,656 parameters.
3. Standard fused timm logits and the unfused observable path differ by less
   than `1e-5`. This measures backend arithmetic only.
4. Adding extraction hooks changes neither unfused logits nor any block input
   or output (bitwise equality).
5. For every block, recomputing
   `softmax((Q * head_dim**-0.5) @ K.T)` matches the actual attention captured
   from timm with maximum absolute error below `1e-6`.
6. Q, K, and V are split in that order into `[B, heads, tokens, head_dim]`.

## Data gates

- The canonical class mapping contains exactly 1,000 unique WNIDs and indices
  `0..999`; anchors are `0=n01440764`, `999=n15075141`.
- Synset-folder labels are independent of directory enumeration order.
- Flat validation images are unlabeled unless the official, hash-verified
  ILSVRC2012 devkit provides the two-step
  `image number -> ILSVRC ID -> WNID -> model index` mapping.
- Every manifest row records file hash, dimensions, label source, WNID, and
  numeric target (or explicit null).

## Provenance gates

Each formal result records the Git commit and requires a clean tree; model and
image hashes; exact structured preprocessing; dependency lock hash; runtime
versions; device/dtype/seed; raw logits artifact; top-1/top-5; and all numeric
parity errors.

