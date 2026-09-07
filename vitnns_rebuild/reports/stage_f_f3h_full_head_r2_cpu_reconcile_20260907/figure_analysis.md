# F3H full-head review analysis

Review scope: all 20 exact report-side PNG copies. All panels use the frozen order-44 median-effect sample. The Value-path panels are `fixed-clean-attention head-direction Value-path attribution progression`, not raw attention, full Q/K/V attribution, or class-logit attribution. Within each block/head coordinate NNS and matched share one monotonic log scale; colors must not be compared in absolute magnitude across coordinates.

## Image #1: NNS positive 12×12 atlas

- Can show: positive support patterns for every block/head without head selection; shallow grid/edge structure and later diffuse/local structure are visible.
- Cannot show: class evidence, head importance, or comparable magnitude across coordinates.
- Likely misread: dense yellow structure could be mistaken for stronger class relevance.
- QC: PASS; B0–B11 and H0–H11 are ordered correctly; no title, label, or panel clipping; text remains readable at the intended full-atlas review size; no FAILED/NA cells exist. Bilinear display resize is visible but introduces no obvious isolated ringing artifact.

## Image #2: NNS negative-magnitude 12×12 atlas

- Can show: negative support is widespread and structurally head-dependent, demonstrating why an absolute-value view hid sign semantics.
- Cannot show: opposition to the predicted class; the sign is relative to the fixed head-direction scalar.
- Likely misread: negative contribution may be mistaken for evidence against class 713.
- QC: PASS; complete 12×12 order, labels and margins are intact; no clipping or missing cells; scale/transform matches the manifest.

## Image #3: matched positive 12×12 atlas

- Can show: the matched patch baseline is visible under the same per-coordinate pair scales, with coarse patch-token spatial support and localized later-layer cells.
- Cannot show: superiority/inferiority to NNS or cross-coordinate absolute magnitude.
- Likely misread: its 14×14-origin block structure may be mistaken for low-resolution rendering failure.
- QC: PASS; all cells are legible, not artificially brightened or dimmed; the block structure is expected from the matched method and is not a resize artifact.

## Image #4: matched negative-magnitude 12×12 atlas

- Can show: matched negative mass is genuinely small under the pair-shared scale for most coordinates, with a few visible shallow-layer structures.
- Cannot show: that matched is globally weaker or worse; only same-coordinate raw magnitudes permit pairwise inspection.
- Likely misread: dark blue panels may be read as missing or manually suppressed.
- QC: PASS; every panel is VALID and present; dark panels are data-driven, not method-specific gain; no clipping or layout error.

## Image #5: block 0 paired page

- Can show: all 12 shallow heads retain strong grid/edge organization in NNS; matched positive support is clearly visible; positive and negative signs are separated.
- Cannot show: class localization or a best head.
- Likely misread: repeated lattice structure may be mistaken for raw attention.
- QC: PASS; four rows and H0–H11 align; NNS/matched share each head's scale; text and panels are not clipped.

## Image #6: block 1 paired page

- Can show: head-to-head diversity and broad matched-positive support at this depth.
- Cannot show: absolute change from block 0 because coordinate targets differ.
- Likely misread: matched-negative dark panels may be treated as absent.
- QC: PASS; all 48 cells present; row labels, headings and spacing are correct; no obvious interpolation artifact beyond expected patch granularity.

## Image #7: block 2 paired page

- Can show: transition from strongly periodic shallow patterns toward more irregular head-specific support.
- Cannot show: a monotonic semantic hierarchy.
- Likely misread: visual complexity may be interpreted as stronger attribution.
- QC: PASS; pair-scale rule is respected; no text/color clipping or FAILED/NA ambiguity.

## Image #8: block 3 paired page

- Can show: NNS positive/negative spatial differences and matched positive structure for all heads.
- Cannot show: class-conditioned head weighting.
- Likely misread: isolated matched cells could be called object parts without class-target evidence.
- QC: PASS; correct row/column order, readable labels, aligned maps, no obvious illegal smoothing.

## Image #9: block 4 paired page

- Can show: one of the pre-existing E3 coordinates in its full 12-head context instead of as a selected single head.
- Cannot show: that head 2 is representative or dominant.
- Likely misread: the inherited head could be visually privileged; the page gives it no special styling.
- QC: PASS; all heads use identical layout; pair-shared visibility is adequate; no clipping or missing cell.

## Image #10: block 5 paired page

- Can show: mid-depth spatial support remains diverse across heads; matched positive remains visible under honest scales.
- Cannot show: an object detector or a cross-layer magnitude trend.
- Likely misread: NNS texture density may be read as stronger explanatory quality.
- QC: PASS; labels, margins and panel alignment are correct; no obvious resize artifact.

## Image #11: block 6 paired page

- Can show: the full head context around inherited coordinate (6,10), including substantial variation in matched positive localization.
- Cannot show: that the inherited coordinate is a preferred head.
- Likely misread: one bright matched cell may be mistaken for method-specific amplification.
- QC: PASS; brightness follows the common head-pair scale; four rows align and no text is clipped.

## Image #12: block 7 paired page

- Can show: head-specific matched hotspots coexist with broad NNS support.
- Cannot show: class evidence or head reliability.
- Likely misread: a visually compact matched hotspot may be equated with greater fidelity.
- QC: PASS; all cells are valid and visible; no layout or interpolation defect.

## Image #13: block 8 paired page

- Can show: later-middle heads differ substantially; some matched positives are localized while NNS remains spatially distributed.
- Cannot show: that distributed support is erroneous or localized support is better.
- Likely misread: visual similarity to Grad-CAM may be treated as target equivalence.
- QC: PASS; row order and head labels match the manifest; no clipping; same-coordinate scale equality holds.

## Image #14: block 9 paired page

- Can show: compact matched-positive cells and varied NNS signs across all heads.
- Cannot show: numerical head weights or cross-block strength.
- Likely misread: the bright H8 matched cell may invite cherry-picking.
- QC: PASS; H8 remains one ordinary cell among all 12; no selection styling, crop, or text defect.

## Image #15: block 10 paired page

- Can show: late-layer support is not a single shared spatial template across heads.
- Cannot show: a direct representation of what the layer learned.
- Likely misread: loss of recognizable object outline may be called model failure.
- QC: PASS; all panels present and aligned; matched is visible where supported by the pair scale; no illegal interpolation.

## Image #16: block 11 paired page

- Can show: all final-block heads, including inherited (11,2), without replacing or selecting heads; matched positive and both NNS signs remain inspectable.
- Cannot show: complete attention, class-logit attribution, or final-layer semantic proof.
- Likely misread: diffuse NNS patterns may be assumed to be class localization.
- QC: PASS; 48/48 cells valid, row/head labels are correct, and no clipping or FAILED/NA ambiguity appears.

## Image #17: per-layer equal-head spatial-support consensus

- Can show: display-space support shared after equal averaging of all 12 heads within each layer, for both methods and signs.
- Cannot show: raw attribution sums, class-conditioned attribution, or learned head importance.
- Likely misread: averaging may be treated as a justified network attribution.
- QC: PASS; B0–B11 and four rows are ordered correctly; titles/labels fit; all required heads were valid, so no failure marker is needed.

## Image #18: depth-band equal-head spatial-support consensus

- Can show: coarse spatial-pattern changes across the four predeclared depth bands.
- Cannot show: absolute magnitude progression or semantic stages.
- Likely misread: smooth consensus fields may be interpreted as stronger agreement than the per-head atlas supports.
- QC: PASS; B0–2, B3–5, B6–8 and B9–11 appear in order; no clipping; labels and panels align.

## Image #19: all-144-head equal-weight spatial-support consensus

- Can show: a descriptive all-head display-space average beside the frozen original image.
- Cannot show: class-conditioned NNS, full-network attribution, or a valid replacement for a path-accounted class target.
- Likely misread: the presence of the original image may invite object-localization claims.
- QC: PASS; original and four consensus panels align; titles fit; matched positive is visible and matched negative is honestly near-zero under the frozen transform.

## Image #20: sealed ViT Grad-CAM reference

- Can show: the class-713 raw-logit Grad-CAM reference emphasizes broad calculator/keypad regions and differs visibly from the head-direction maps.
- Cannot show: a same-target winner comparison or validate class-conditioned NNS weights.
- Likely misread: its intuitive object alignment may be treated as proof that it is scientifically superior to the different-target Value-path methods.
- QC: PASS; original, map and overlay are aligned; titles are not clipped; the 14×14 source produces expected coarse cell boundaries without hidden percentile clipping.

## Overall decision

All 20 review copies pass visual QC. No text or colorbar-equivalent scale information is clipped, no rows are misplaced, no panels are missing, and all 288 Value-path cells are VALID. The main scientific visual finding is head and depth diversity, not a class-conditioned weighting rule. Consequently the atlas supports the predeclared `INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS` result and only authorizes a prospective F3C0 protocol freeze.
