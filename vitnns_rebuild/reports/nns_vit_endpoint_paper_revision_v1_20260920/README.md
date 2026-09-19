# Endpoint publication revision v1

Open the [gallery](gallery.html), [Chinese manuscript](../../papers/nns_vit_endpoint/zh/full_draft.md), or [English manuscript](../../papers/nns_vit_endpoint/en/full_draft.md).

This derivative package addresses 修改意见/修改意见v1.md and reorganizes the user-named [R2 source package](../nns_vit_endpoint_all_figures_r2_20260919/README.md) into ten main figures. R2 remains byte-for-byte unchanged. Every main figure has PNG, PDF and SVG exports. The paper supplement links all 40 former main figures and 52 earlier supplemental records.

The key visual changes are paired signed/absolute views; shared robust scales for related maps; original RGB L1/L2/signed-sum annotations; consistent type, color and layout; and fewer, more informative main pages. Signed maps sum channels, while absolute maps sum channelwise magnitudes. Both originate from the same stored arrays.

New descriptive tables recompute absolute spatial AUC, evaluated-candidate oracle, normalized regret, logit-scale ratios, explicit-tolerance reference changes, and head-rank concordance from existing records. The frozen scientific decisions are unchanged. This package adds no model inference or new inferential tests.

- [Figure-by-figure explanation](figure_analysis.md)
- [Chinese captions](captions_zh.md) / [English captions](captions_en.md)
- [Analysis](analysis.json) / [analysis source hashes](analysis_sources.json)
- [Display metadata](display_receipt.json) / [figure source hashes](figure_sources.json)
- [Figure manifest](manifest.json) / [visual QC](manual_visual_qc.json)
- [Revision protocol and scope](../../papers/nns_vit_endpoint/notes/revision_v1_protocol.md)
