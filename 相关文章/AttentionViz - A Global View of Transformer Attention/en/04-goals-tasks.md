# 4. Goals & Tasks

The overarching aim of this work is to design a novel visualization technique that allows exploration of global attention trends in transformer models. To collect some initial feedback about this idea and learn more about user needs, we talked with 5 machine learning (ML) researchers (4 Ph.D. students and 1 professor) interested in model interpretability. During these individual interviews, we asked experts to describe their current practices and challenges when working with transformers and how attention visualizations could aid in their research objectives. We will refer to these experts as **E1-5**.

## Goals

Ultimately, our conversations with experts yielded three main goals:

**G1 - Understand how self-attention informs model behavior.** Overall, all 5 experts wanted to better understand the behavior of different attention heads and what transformer models are learning through their characteristic self-attention mechanism. Thus, they expressed the desire to be able to quickly and easily explore attention patterns. **E2** explained that “attention is still pretty closed-box and there’s lots of mysteries,” so gaining a deeper understanding of transformer attention patterns could provide insights into “why large language models fail at reasoning tasks and math,” for example.

**G2 - Compare & contrast attention heads.** **E5** mentioned that visualizing differences in attention heads could help with hypothesis generation, which is the first step in their research process: “Visualization can help formulate hypotheses to test and get an intuitive sense of what transformers are doing.” Additionally, 3 experts (**E1, E2, E5**) noted that attention head comparison would be useful for model pruning and editing purposes. That is, if two attention heads appear to behave similarly, perhaps one could be removed without significantly impacting model performance. In the words of **E1**, comparing heads might allow us to “find parts of the model that are actually useful.”

**G3 - Identify attention anomalies.** Four researchers (**E2-5**) wanted to identify irregularities and potential behavioral issues with transformers through attention pattern exploration. This information could then be used for model debugging purposes. For instance, **E4** said “visualizing attention could help you notice when the model is looking at the wrong thing, even if the result is correct.” **E3** agreed, reiterating the importance of debugging especially in the context of model training: “Training often fails and dies, but it’s hard to understand why it fails or produces unexpected behavior.”

## Tasks

Given these goals, we developed the following set of design tasks:

**T1 - Visualize attention heads at scale.** To help users quickly explore model behavior **\[G1\]** and easily compare & contrast attention patterns **\[G2\]**, our tool simultaneously visualizes self-attention heads across transformer layers.

**T2 - Explore query-key interactions.** **E1** and **E4** expressed the desire to better understand query-key pairing information toward improving their understanding of transformer self-attention. Thus, our tool further supports attention pattern comparison **\[G2\]** and anomaly detection **\[G3\]** through visualizing query-key interactions.

**T3 - Probe attention at multiple levels.** Our tool allows for local and global comparisons of attention **\[G2\]** by providing visualizations at the *sentence/image*, *head*, and *model* levels. The flexibility of switching between multiple views in a single interface also facilitates knowledge discovery **\[G1\]** and helps users identify model irregularities **\[G3\]**.

**T4 - Customize model and data inputs.** AttentionViz is easily extendable to new transformers and datasets, affording quick visual comparison **\[G2\]** and synthesis of attention patterns **\[G1\]** across different models and modalities (language & vision).
