# 2. Background on Transformer Models

The *transformer*, introduced in (Vaswani et al. 2017), is a neural network architecture designed to operate on sequential input. A full description of transformers is beyond the scope of this paper, but a few concepts are critical for understanding our work. First, a transformer receives as input a set of vectors (often called **embeddings**). Embeddings can represent a variety of input types. In text-based transformers, they correspond to words or pieces of words; in vision transformers, they encode patches of pixels.

The network iteratively transforms these vectors via a series of **attention layers**, each of which moves information between pairs of embeddings. The name “attention” suggests that not all embeddings will be equally related; certain pairs will interact more strongly–i.e., pay more “attention” to each other. Attention layers determine which pairs should interact, and what information should flow between them.

For example, in a transformer operating on the words of the sentence, “The brown capybara is sleeping now,” one might expect high attention (and information flow) between embeddings for “capybara” and “is,” but not between “brown” and “now.” The *self-attention* mechanism, which is our focus in this paper, allows transformers to learn and use a rich set of relationships between elements of a sequence, yielding significant performance improvements across various NLP and computer vision tasks (Devlin et al. 2018; Radford et al. 2018; Dosovitskiy et al. 2021).

There may be different reasons for embedding pairs to attend to each other. For instance, in our example sentence,“brown” and “capybara” are linked by an adjective-noun relation, while “capybara” and “is” form a subject-verb relation. To allow for several relation types, transformer attention layers consist of multiple **attention heads**, each of which can represent a different pattern of attention and information flow.

Each attention head computes its own attention pattern using a bilinear form computed from a query weight matrix $W_Q$ and key weight matrix $W_K$. Concretely, for two embedding vectors $x$ and $y$, attention is related to the scaled inner product of a **query vector**, $W_Qx$, and a **key vector**, $W_Ky$. Letting $d$ be the dimension of $W_Ky$, we have:

$$
f(x,y) = \frac{1}{\sqrt{d}}\langle W_Qx, W_Ky \rangle
$$

Given embedding vectors $\{x_1, x_2, \ldots, x_n\}$, we compute the self-attention between $x_i$ and the other vectors using the softmax function:
$$
attn(x_i, x_j) = \mathrm{softmax}_j(f(x_i, x), \ldots, f(x_i, x_n)) = e^{f(x_i, x_j)} / \sum_k  e^{f(x_i, x_k)}
$$
Critically, this formula shows that the greater the dot product between the query and key vectors, the higher the final attention value will be, a fact that we rely upon in our joint embedding visualization.

There is much more to the transformer architecture than we cover here. In particular, we have only described the attention weighting between pairs of embeddings, and not the specific information that flows between them. (As discussed later, this is an area ripe for further investigation.) One last technical point is worth mentioning, however, since it will help interpret images later in the paper. The initial embeddings given to a transformer typically incorporate a vector representation of their ordering (for a 1D sequence) or spatial configuration (for a grid, as in vision transformers). For sequences, these position vectors are defined using trigonometric functions, and are located on a helix-like curve in high-dimensional space (see (Vaswani et al. 2017)).

## Models Studied in this Paper

We study three transformer models: BERT (language), GPT-2 (language), and ViT (vision). Each has been an important object of study in the machine learning community, and the three span a range of transformer architectures and applications. **BERT**, or Bidirectional Encoder Representations from Transformers (Devlin et al. 2018), is a multi-layer transformer encoder. As a bidirectional model, BERT can attend to tokens (i.e., input elements) in either direction. **GPT-2**, or Generative Pre-trained Transformer 2 (Radford et al. 2019), is a multi-layer transformer decoder. GPT-2 is a unidirectional model, meaning it only attends to previous tokens. **ViT**, or Vision Transformer (Dosovitskiy et al. 2021), employs a self-attention-based transformer architecture by splitting images into “patches” and treating them like tokens in a sentence. Similar to BERT, ViT is a multi-layer, bidirectional transformer encoder. In this work, we look at ViT performance on 16x16 (**ViT-16**) and 32x32 (**ViT-32**) patch sizes.
