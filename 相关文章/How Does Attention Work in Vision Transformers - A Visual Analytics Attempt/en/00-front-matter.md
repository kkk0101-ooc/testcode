# How Does Attention Work in Vision Transformers? A Visual Analytics Attempt

**Authors:** Yiran Li, Junpeng Wang, Xin Dai, Liang Wang, Chin-Chia Michael Yeh, Yan Zheng, Wei Zhang, and Kwan-Liu Ma

**Affiliations:** University of California, Davis; Visa Research, Palo Alto, California.

**arXiv:** <https://arxiv.org/abs/2303.13731>

## Abstract

Vision transformer (ViT) expands the success of transformer models from sequential data to images. The model decomposes an image into many smaller patches and arranges them into a sequence. Multi-head self-attentions are then applied to the sequence to learn the attention between patches. Despite many successful interpretations of transformers on sequential data, little effort has been devoted to the interpretation of ViTs, and many questions remain unanswered. For example, among the numerous attention heads, which one is more important? How strong are individual patches attending to their spatial neighbors in different heads? What attention patterns have individual heads learned? In this work, we answer these questions through a visual analytics approach. Specifically, we first identify what heads are more important in ViTs by introducing multiple pruning-based metrics. Then, we profile the spatial distribution of attention strengths between patches inside individual heads, as well as the trend of attention strengths across attention layers. Third, using an autoencoder-based learning solution, we summarize all possible attention patterns that individual heads could learn. Examining the attention strengths and patterns of the important heads, we answer why they are important. Through concrete case studies with experienced deep learning experts on multiple ViTs, we validate the effectiveness of our solution that deepens the understanding of ViTs from head importance, head attention strength, and head attention pattern.

**Keywords:** Vision transformer; Multi-head self-attention; Deep learning; Explainable artificial intelligence; Visual analytics
