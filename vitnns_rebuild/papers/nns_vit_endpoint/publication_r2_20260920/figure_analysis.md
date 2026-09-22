# 图件逐图说明

**Image #1 — From total Gradient × Input to auditable dual-indexed maps**

已有总 Gradient × Input 先按 consumer-use edge 展开为 $[e,i]$ 图组，再由合法 attention、head 或 residual 切面重构同一 signed 总量。最后一列列出该联合对象支持的机制问题；底部直接显示去掉 $i$、$e$、cut contract 或 target registry 后的退化。

[打开图片](figures/png/fig_01_concept_overview.png)

**Image #2 — Valid differential cuts partition live paths exactly once**

左栏用 producer、consumer、input slot 与 invocation 区分同一 producer tensor 的两次实际使用；中栏对齐 Q/K/V、multi-head 与 residual 切面；右栏给出 $\mathcal L_{e,i}$、signed closure 和已封存漏项/重复失败模式。红色虚线标记切面及 residual 的完整路径要求。

[打开图片](figures/png/fig_02_neuron_learning_image.png)

**Image #3 — Initialization: split the input, then assign one image to each coordinate**

左栏将输入拆成四张稀疏坐标图，右栏以同色虚线把图分配给输入节点；公式位于步骤底部。RGB 输入对各通道采用相同初始化。

[打开图片](figures/png/fig_03_coordinate_initialization.png)

**Image #4 — Generation: collect every input path to the selected target**

灰色网络表示中间 learning images；红色虚线路径突出首末输入坐标到目标的合成系数，其余系数在目标侧列出。最终矩阵是输入坐标图的加权和。

[打开图片](figures/png/fig_04_local_derivative_generation.png)

**Image #5 — Patch embedding is a local projection and a sparse global linear map**

左栏显示真实图像上的局部 patch projection；右栏把同一运算写为全局输入坐标上的稀疏线性行。橙色 patch、非零权重块和 learning-image support 一一对应。

[打开图片](figures/png/fig_05_patch_embedding.png)

**Image #6 — Attention separates routing from value transmission**

Q/K 的变化通过 attention weights 改变 routing；V 的变化通过当前权重传播。两条微分路径相加重构完整 attention 微分，head 下游 cotangent 包含 output projection。

[打开图片](figures/png/fig_06_attention_differential.png)

**Image #7 — Residual connections preserve an identity path and add a transformed path**

上半区展示 residual 的 identity/branch 计算边，下半区保留原图并对齐两条真实路径及输出 learning image。同色虚线连接计算边与图像。

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
