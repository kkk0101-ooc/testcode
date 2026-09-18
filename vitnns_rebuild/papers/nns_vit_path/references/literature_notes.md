# Literature notes

## Primary reference anchor

```yaml
primary_reference_anchor:
  title: Understanding neural network through neuron level visualization
  authors: Hui Dou; Furao Shen; Jian Zhao; Xinyu Mu
  venue: Neural Networks 168 (2023), 484-495
  doi: 10.1016/j.neunet.2023.09.030
  role: NNS source method for FCN/CNN; this manuscript adapts and evaluates the learning-image view for ViT
```

The source paper constructs neuron-level learning images, combines them to inspect network components, and demonstrates the method on FCNs, CNNs, and ResNet. Its FCN/CNN empirical findings are not treated as ViT evidence.

## Verified methodological anchors

| Key | Source | Verified role in this draft | Local source |
|---|---|---|---|
| Dosovitskiy2021 | Dosovitskiy et al., *An Image Is Worth 16x16 Words*, ICLR 2021 | ViT architecture | `相关文章/How Does Attention Work in Vision Transformers?/en/11-references.md` and other local reference lists |
| Sundararajan2017 | Sundararajan, Taly, Yan, *Axiomatic Attribution for Deep Networks*, ICML 2017 | IG definition and axioms | `相关文章/Towards Evaluating Explanations.../en/10-references.md` |
| Dhamdhere2018 | Dhamdhere, Sundararajan, Yan, *How Important Is a Neuron?*, arXiv:1805.12233, 2018 | Conductance and hidden-unit attribution | Live-verified arXiv record; local technical comparison remains incomplete |
| Abnar2020 | Abnar and Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020 | Attention rollout | local ViT related-paper reference lists |
| Chefer2021 | Chefer, Gur, Wolf, *Transformer Interpretability Beyond Attention Visualization*, CVPR 2021 | Transformer relevance comparator/background | local related-paper reference lists |
| Selvaraju2017 | Selvaraju et al., *Grad-CAM: Why did you say that?*, ICCV 2017 | Grad-CAM background | local related-paper reference lists |

## Local style and topic references

| Source | Research question | Relevance to this manuscript |
|---|---|---|
| Li et al., *How Does Attention Work in Vision Transformers? A Visual Analytics Attempt*, arXiv:2303.13731 | How do head importance, patch relations, and attention patterns vary in ViTs? | Organizing head-, patch-, and layer-level questions; attention visualization background |
| Yeh et al., *AttentionViz: A Global View of Transformer Attention*, arXiv:2305.03210 | How can joint query-key embeddings expose attention across many inputs? | Global attention visualization background and paper-structure reference |
| Komorowski et al., *Towards Evaluating Explanations of Vision Transformers for Medical Imaging*, arXiv:2304.06133 | How should ViT explanation methods be evaluated quantitatively? | Faithfulness/sensitivity framing and class-targeted versus class-agnostic distinction |
| Carrasco et al., *Vision Transformer attention alignment with human visual perception in aesthetic object evaluation*, arXiv:2507.17616 | How do ViT attention heads align with human gaze? | Example of head-wise and human-attention evaluation; not evidence for our claims |

## Reference gaps before submission

- Complete the line-by-line comparison with conductance, internal influence, path attribution, and transformer-specific attribution methods.
- Replace provisional arXiv entries with final venue/DOI metadata where applicable.
- Confirm the target journal's citation style and generate the final BibTeX database.
- Add original sources for feature visualization and attribution fragility only where the final prose retains those claims.
