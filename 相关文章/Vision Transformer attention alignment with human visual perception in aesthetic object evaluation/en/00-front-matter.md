# Vision Transformer attention alignment with human visual perception in aesthetic object evaluation

**Authors:** Miguel Carrasco, Cesar Gonzalez-Martin, Jose Aranda, and Luis Oliveros

**Affiliations:** Universidad Diego Portales; University of Cordoba; Universidad Adolfo Ibanez.

**arXiv:** <https://arxiv.org/abs/2507.17616>

## Abstract

Visual attention mechanisms play a crucial role in human perception and aesthetic evaluation. Recent advances in Vision Transformers (ViTs) have demonstrated remarkable capabilities in computer vision tasks, yet their alignment with human visual attention patterns remains underexplored, particularly in aesthetic contexts. This study investigates the correlation between human visual attention and ViT attention mechanisms when evaluating handcrafted objects. We conducted an eye-tracking experiment with 30 participants (9 female, 21 male, mean age 24.6 years) who viewed 20 artisanal objects comprising basketry bags and ginger jars. Using a Pupil Labs eye-tracker, we recorded gaze patterns and generated heat maps representing human visual attention. Simultaneously, we analyzed the same objects using a pre-trained ViT model with DINO (Self-DIstillation with NO Labels), extracting attention maps from each of the 12 attention heads. We compared human and ViT attention distributions using Kullback-Leibler divergence across varying Gaussian parameters ($`\sigma=0.1 to 3.0`$). Statistical analysis revealed optimal correlation at $`\sigma=2.4\pm0.03`$, with attention head \#12 showing the strongest alignment with human visual patterns. Significant differences were found between attention heads, with heads \#7 and \#9 demonstrating the greatest divergence from human attention ($`p\leq 0.05`$, Tukey HSD test). Results indicate that while ViTs exhibit more global attention patterns compared to human focal attention, certain attention heads can approximate human visual behavior, particularly for specific object features like buckles in basketry items. These findings suggest potential applications of ViT attention mechanisms in product design and aesthetic evaluation, while highlighting fundamental differences in attention strategies between human perception and current AI models.

**Keywords:** Vision Transformer; Human visual attention; Eye tracking; Aesthetic evaluation; Attention alignment
