# References

Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S., & Samek, W.
(2024). AttnLRP: Attention-aware layer-wise relevance propagation for Transformers. *Proceedings
of the 41st International Conference on Machine Learning*, PMLR 235, 135–168.

Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. *Proceedings of the
58th Annual Meeting of the Association for Computational Linguistics*, 4190–4197.

Buratti, C., Marchetti, M., Parlapiano, F., Traini, D., Ursino, D., & Virgili, L. (2026).
Gradient-skipping relevance propagation for efficient explainability of Vision Transformers.
*arXiv preprint arXiv:2607.10365v1*.

Chefer, H., Gur, S., & Wolf, L. (2021). Transformer interpretability beyond attention
visualization. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition*, 782–791.

Chen, P.-K., Plaat, A., & van Stein, N. (2026). Every component is a lookup: Token attribution
and composition from a single decomposition. *arXiv preprint arXiv:2605.23393v2*.

Dhamdhere, K., Sundararajan, M., & Yan, Q. (2018). How important is a neuron? *arXiv preprint
arXiv:1805.12233*.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words:
Transformers for image recognition at scale. *International Conference on Learning
Representations*.

Dou, H., Shen, F., Zhao, J., & Mu, X. (2023). Understanding neural network through neuron level
visualization. *Neural Networks, 168*, 484–495. https://doi.org/10.1016/j.neunet.2023.09.030

Ferrando, J., Gállego, G. I., & Costa-jussà, M. R. (2022). Measuring the mixing of contextual
information in the Transformer. *Proceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing*, 8698–8714.

Izadi, A., Banayeeanzade, M., Mirrokni, A., Hasani, H., Bagherian, M., Mehri, F., & Soleymani
Baghshah, M. (2026). Causal attribution via activation patching. *arXiv preprint
arXiv:2603.13652v1*.

Kramár, J., Lieberum, T., Shah, R., & Nanda, N. (2024). AtP*: An efficient and scalable method
for localizing LLM behaviour to components. *arXiv preprint arXiv:2403.00745*.

Modarressi, A., Fayyaz, M., Yaghoobzadeh, Y., & Pilehvar, M. T. (2022). GlobEnc: Quantifying
global token attribution by incorporating the whole encoder layer in Transformers. *Proceedings
of the 2022 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies*, 258–271.

Modarressi, A., Fayyaz, M., Aghazadeh, E., Yaghoobzadeh, Y., & Pilehvar, M. T. (2023).
DecompX: Explaining Transformers decisions by propagating token decomposition. *Proceedings of
the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers)*, 2649–2664.

Mehri, F., Soleymani Baghshah, M., & Pilehvar, M. T. (2025). LibraGrad: Balancing gradient flow for
universally better Vision Transformer attributions. *Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition*, 67–78.

Nishankar, S., Sanjeewani, P., Perera, A., & Thuseethan, S. (2026). HiLRP: Toward one trustworthy
explanation for Vision Transformer: Conservation-valid attribution via attention primitives.
*arXiv preprint arXiv:2609.01282v1*.

Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning important features through propagating
activation differences. *Proceedings of the 34th International Conference on Machine Learning*,
PMLR 70, 3145–3153.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM:
Visual explanations from deep networks via gradient-based localization. *Proceedings of the
IEEE International Conference on Computer Vision*, 618–626.

Srinivas, S., & Fleuret, F. (2019). Full-gradient representation for neural network visualization.
*Advances in Neural Information Processing Systems, 32*, 4126–4135.

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks.
*Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 3319–3328.

Syed, A., Rager, C., & Conmy, A. (2024). Attribution patching outperforms automated circuit
discovery. *Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural
Networks for NLP*, 407–416. https://doi.org/10.18653/v1/2024.blackboxnlp-1.25

Zhang, F., & Nanda, N. (2023). Towards best practices of activation patching in language models:
Metrics and methods. *arXiv preprint arXiv:2309.16042*.

Wróbel, A., Gairola, S., Tabor, J., Schiele, B., Zieliński, B., & Rymarczyk, D. (2026). DAVE:
Distribution-aware attribution via ViT gradient decomposition. *arXiv preprint arXiv:2602.06613v1*.

Żukowska, N., Stammer, W., Schiele, B., & Fischer, J. (2026). Seeing through circuits: Faithful
mechanistic interpretability for Vision Transformers. *arXiv preprint arXiv:2604.14477v1*.

## Software references

Captum contributors. `NeuronGradient`, `InputXGradient`, and `NeuronConductance` documentation and
source snapshot. Exact URL, access record, excerpt, and content hash are frozen in
`manifests/nns_vit_endpoint/ep0_r1_20260918/literature_registry.json`. These items should be cited
as software documentation rather than peer-reviewed method papers in the final bibliography.

## Reference scope

- Dou et al. (2023) is the primary NNS source. Its FCN/CNN experiments are background, not ViT
  evidence.
- Shrikumar et al. (2017), Sundararajan et al. (2017), Dhamdhere et al. (2018), and Srinivas and
  Fleuret (2019) anchor local input weighting, path integration, hidden-unit attribution, and the
  input-plus-bias full-gradient representation.
- Modarressi et al. (2022, 2023), Achtibat et al. (2024), Chefer et al. (2021), and Mehri et al.
  (2025) anchor Transformer token contribution and relevance propagation; their completeness
  objects are not treated as equivalent to local-differential cut closure.
- Kramár et al. (2024) anchors the activation-displacement Taylor comparison; the present image
  intervention uses its own frozen target, sign, reference, and aggregation definitions.
- Zhang and Nanda (2023) and Syed et al. (2024) anchor direct activation replacement and its
  first-order component/edge attribution approximation.
- Achtibat et al. (2024), Chefer et al. (2021), and Mehri et al. (2025) are related Transformer
  attribution methods, not equivalent implementations of Endpoint Scanner.
- Ferrando et al. (2022), Chen et al. (2026), Nishankar et al. (2026), and Buratti et al. (2026)
  anchor ALTI, K/Q/V-labelled path analysis, primitive-based conservation, and skip-aware relevance.
- Wróbel et al. (2026), Żukowska et al. (2026), and Izadi et al. (2026) anchor recent ViT gradient
  decomposition, visual edge circuits, and activation-patching attribution. Their presence rules out
  novelty claims based only on applying gradients, computational edges, or interventions to ViTs.
