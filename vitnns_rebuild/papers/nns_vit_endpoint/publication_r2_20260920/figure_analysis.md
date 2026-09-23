# 图件逐图说明

**Image #1 — One image, one target, many auditable local maps**

以同一懒熊输入和 ground-truth logit 开场，先展示真实 Total Endpoint，再展示 block 5 的 identity、routing-induced 和 value-induced 局部图。右侧把结构审计与有限替换评价分开，强调闭合不等于干预效用。热图使用正部显示，数值仍绑定原始 signed arrays。

[打开图片](figures/png/fig_01_concept_overview.png)

**Image #2 — Use identity and valid cuts make the decomposition auditable**

左栏说明同一 producer tensor 的不同 consumer、slot 和 invocation 仍是不同 use。右栏把 attention branch partition 和包含 identity bypass 的 residual global cut 分开；`+ -> target s` 输出边单独水平展示，避免和 global cut 线或 branch 箭头混在一起。

[打开图片](figures/png/fig_02_neuron_learning_image.png)

**Image #3 — Initialization keeps both input value and location**

输入矩阵拆成四张单坐标图，再堆叠为 $\operatorname{diag}(x)$。每张图只在一个坐标保留原输入值，因此同时保留数值和位置。

[打开图片](figures/png/fig_03_coordinate_initialization.png)

**Image #4 — Local Jacobians propagate coordinate maps to the target**

每张输入坐标图乘以当前点局部系数 $c_i=\partial s/\partial x_i$，再相加得到目标 learning image。右侧矩阵直接展示 $\mathcal L(s;x)=x\odot\nabla_xs$ 的加权结果。

[打开图片](figures/png/fig_04_local_derivative_generation.png)

**Image #5 — Patch embedding begins with strictly local support**

左栏展示真实图像上的一个 patch projection；右栏展示等价的全局稀疏输入行。橙色区域在 token、稀疏权重和 learning-image support 中一一对应。

[打开图片](figures/png/fig_05_patch_embedding.png)

**Image #6 — Attention separates routing from value transmission**

上方把 Q/K 产生 A、V 单独输入 O 的前向结构拆开，避免把 V 混入 routing。下方用真实 block-5 分项图展示 routing-induced 与 value-induced 相加重构 attention branch contribution；全局目标还需 residual identity。

[打开图片](figures/png/fig_06_attention_differential.png)

**Image #7 — Residual closure is visible in the path maps**

同一懒熊输入、ground-truth logit 和 block-5 residual 下，identity path 与 attention branch 使用相同 residual 输出 cotangent。完整 signed maps 相加得到 Total Endpoint，图中正部显示只用于空间阅读。

[打开图片](figures/png/fig_07_residual_principle.png)

**Image #8 — Different targets reveal different parts of the same input**

同一图像的不同内部与类别目标。Energy targets 使用前向空间方差最大的通道；这与单个 token-channel 目标和最终类别目标区分。

[打开图片](figures/png/fig_08_internal_targets.png)

**Image #9 — DeiT-S/16: learning images through all twelve blocks**

同一懒熊输入跨全部 12 层的学习图像及真实前向特征。每层通道按前向空间方差选择；上排为该通道 energy scalar 的输入图，下排为该通道原生 14×14 activation 的正部。各面板独立缩放。

[打开图片](figures/png/fig_09_deit_s16_layerwise.png)

**Image #10 — ViT-B/16: learning images through all twelve blocks**

同一懒熊输入跨全部 12 层的学习图像及真实前向特征。每层通道按前向空间方差选择；上排为该通道 energy scalar 的输入图，下排为该通道原生 14×14 activation 的正部。各面板独立缩放。

[打开图片](figures/png/fig_10_vit_b16_layerwise.png)

**Image #11 — Inside a block: normalization, attention and the MLP**

DeiT-S blocks 0、5、11 内部模块的学习图像和真实前向 activation。每个模块独立选择高空间方差通道并定义 energy target；LayerNorm、attention projection、MLP 的 fc1/GELU/fc2 清楚对应。

[打开图片](figures/png/fig_11_module_learning_images.png)

**Image #12 — Spatial dependence broadens with depth**

完整 cohort 的图像级层深轨迹，阴影为 95% bootstrap 区间。空间分析每模型 100 图，attention/residual 每模型 400 图；推断沿用原冻结分析。

[打开图片](figures/png/fig_12_spatial_broadening.png)

**Image #13 — Attention heads distribute dependence differently**

DeiT-S 在 blocks 0、5、11 的全部六个 heads，对同一个 ground-truth logit 的输入依赖图。原图逐行保留，无按地图观感筛选 head。

[打开图片](figures/png/fig_13_all_heads.png)

**Image #14 — Class-target learning images across two models**

三个主体可辨的展示样本及两模型的同类别 Endpoint。样本为展示选择，模型间颜色强度不作为量值比较。

[打开图片](figures/png/fig_14_cross_model_examples.png)

**Image #15 — Routing and value on the same image and class target**

同一懒熊输入在浅层和深层的 Q、K、routing 与 value 路径。蓝底保留正部细节；signed closure 和绝对质量来自完整原始 arrays。

[打开图片](figures/png/fig_15_routing_value_examples.png)

**Image #16 — Deep attention departs from Value-only propagation**

完整 cohort 的图像级层深轨迹，阴影为 95% bootstrap 区间。空间分析每模型 100 图，attention/residual 每模型 400 图；推断沿用原冻结分析。

[打开图片](figures/png/fig_16_attention_depth.png)

**Image #17 — Two residual sublayers, one downstream target**

同一 block 5 中 attention 与 MLP residual 对同一类别目标的两条路径。两行的 identity 指向不同 residual 输入边。正部图不可直接相加，完整 signed maps 满足闭合。

[打开图片](figures/png/fig_17_residual_examples.png)

**Image #18 — Attention and MLP residuals follow distinct trends**

完整 cohort 的图像级层深轨迹，阴影为 95% bootstrap 区间。空间分析每模型 100 图，attention/residual 每模型 400 图；推断沿用原冻结分析。

[打开图片](figures/png/fig_18_residual_depth.png)

**Image #19 — DeiT-S/16: learning images and other visual explanations**

同输入实际计算的 Endpoint、patch-max 显示变体、Grad-CAM、attention rollout 与 raw attention。Endpoint/Grad-CAM 使用同一 GT logit；后两种 attention 方法不依赖类别。所有图采用相同蓝底色系并独立归一化，只比较表示形态与粒度。三例为 Langur、Stethoscope、Offshore platform，对应重选协议的 orders 1/13/7。排序只读取 ViT-B Endpoint 与输入细节，未读取 Grad-CAM、rollout 或 raw attention。

[打开图片](figures/png/fig_19_deit_s16_method_comparison.png)

**Image #20 — ViT-B/16: learning images and other visual explanations**

同输入实际计算的 Endpoint、patch-max 显示变体、Grad-CAM、attention rollout 与 raw attention。Endpoint/Grad-CAM 使用同一 GT logit；后两种 attention 方法不依赖类别。所有图采用相同蓝底色系并独立归一化，只比较表示形态与粒度。

[打开图片](figures/png/fig_20_vit_b16_method_comparison.png)

**Image #21 — Spatial intervention evaluates a different property**

左侧示例的有限区域替换曲线与右侧完整 400 图 family 的绝对 AUC。候选 oracle 只在已评价区域集合中取最优；数学闭合与区域干预选择属于不同性质。

[打开图片](figures/png/fig_21_intervention_boundary.png)

**Image #22 — Reading positive structure without discarding signed information**

同一原始图的蓝底正部、signed 与 RGB absolute mass 三种视图。正部用于读取空间细节，signed 保留方向，absolute 保留通道抵消前的总幅值。

[打开图片](figures/png/fig_22_signed_and_blue.png)
