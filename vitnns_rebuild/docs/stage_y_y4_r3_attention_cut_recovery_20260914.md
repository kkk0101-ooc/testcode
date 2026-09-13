# Stage Y4-R3 prospective aggregate-attention cut recovery

Y4-R2 stopped at the first clean source with `BLOCKED_SCIENTIFIC_VALIDITY`. Its
prospective implementation had mapped the spatial-support and patch-mixing
depth endpoints to each block's output complete-cut. The frozen complete-cut
topology makes every block cut an alternative reconstruction of the same final
input attribution, and the final block may legally route the class-logit only
through CLS, leaving zero patch-token matrix mass. Therefore block output is
not the registered depth-varying block contribution.

Y4-R3 fixes only this mapping. The Y0 metric registry explicitly names the
primary block component as the aggregate attention branch, and the component
registry defines it as the sum of all 12 native heads. Spatial support uses
`bXX_first_attention_cell`; patch mixing uses
`bXX_first_attention_matrix/cls`. Perturbation inverse mapping uses the same
aggregate-attention matrix. This is a topology/registry correction, not an
effect-driven endpoint choice. Within-patch, head, branch, intervention,
statistical and claim-decision implementations are inherited unchanged.

The new root is `evidence/stage_y/y4_r3_mechanism_analysis_20260914/formal`.
Y4/Y4-R2 roots and all earlier evidence remain immutable. CPU-only and all
access prohibitions remain in force. No Y5 work is authorized during this run.
