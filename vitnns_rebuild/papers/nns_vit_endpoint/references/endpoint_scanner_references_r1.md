# Endpoint Scanner draft R1: provisional references

This list supports `zh/full_draft.md` and `en/full_draft.md`. It combines the previously verified manuscript bibliography
with the targeted prior-art registry frozen at Endpoint EP0. Final submission metadata and BibTeX
keys still require an editorial audit.

Achtibat, R., et al. (2024). AttnLRP: Attention-aware layer-wise
relevance propagation for Transformers. *Proceedings of the 41st International Conference on
Machine Learning*, PMLR 235, 135–168. `[FULL AUTHOR LIST TO VERIFY]`

Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. *Proceedings of the
58th Annual Meeting of the Association for Computational Linguistics*, 4190–4197.

Chefer, H., Gur, S., & Wolf, L. (2021). Transformer interpretability beyond attention
visualization. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition*, 782–791.

Dhamdhere, K., Sundararajan, M., & Yan, Q. (2018). How important is a neuron? *arXiv preprint
arXiv:1805.12233*.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words:
Transformers for image recognition at scale. *International Conference on Learning
Representations*.

Dou, H., Shen, F., Zhao, J., & Mu, X. (2023). Understanding neural network through neuron level
visualization. *Neural Networks, 168*, 484–495. https://doi.org/10.1016/j.neunet.2023.09.030

Kramár, J., Lieberum, T., Shah, R., & Nanda, N. (2024). AtP*: An efficient and scalable method
for localizing LLM behaviour to components. *arXiv preprint arXiv:2403.00745*.

Mehri, Baghshah, & Pilehvar (2025). LibraGrad: Balancing gradient flow for universally better Vision
Transformer attributions. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition*, 67–78. `[AUTHOR INITIALS TO VERIFY]`

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks.
*Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 3319–3328.

## Software/API precedents

Captum contributors. `NeuronGradient`, `InputXGradient`, and `NeuronConductance` documentation and
source snapshot. Exact URL, access record, excerpt, and content hash are frozen in
`manifests/nns_vit_endpoint/ep0_r1_20260918/literature_registry.json`. These items should be cited
as software documentation rather than peer-reviewed method papers in the final bibliography.

## Reference-audit boundary

- Dou et al. (2023) is the primary NNS source. Its FCN/CNN experiments are background, not ViT
  evidence.
- Sundararajan et al. (2017) and Dhamdhere et al. (2018) anchor path integration and hidden-unit
  attribution.
- Kramár et al. (2024) anchors the activation-displacement Taylor comparison; the present image
  intervention uses its own frozen target, sign, reference, and aggregation definitions.
- Achtibat et al. (2024), Chefer et al. (2021), and Mehri et al. (2025) are related Transformer
  attribution methods, not equivalent implementations of Endpoint Scanner.
