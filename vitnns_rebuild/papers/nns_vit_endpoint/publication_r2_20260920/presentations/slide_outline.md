# Endpoint NNS-ViT 汇报 PPT 页级提纲

- 内容来源：`zh/manuscript.md`
- 源稿 SHA-256：`d5f8755d40b14119d78217facc5f47f2d5330bd05a5b64bda0d7173364e7c0cb`
- 论文正式图：`figures/png/fig_01` 至 `fig_22`
- 讲者备注：两套 PPT 每页均已写入

## 20页重点版

01. 看懂 ViT 内部目标到底依赖了图像哪里  （图：fig_01_concept_overview.png）
02. 先用一句话说清楚  （图：—）
03. 为什么普通可视化还不够？  （图：fig_14_cross_model_examples.png）
04. 本文实际回答四个问题  （图：—）
05. 核心公式其实很熟悉  （图：—）
06. 真正新增的是“双索引”  （图：fig_02_neuron_learning_image.png）
07. 合法切面：必须不漏，也不能重复  （图：fig_02_neuron_learning_image.png）
08. Attention 被拆成 routing 与 value  （图：fig_06_attention_differential.png）
09. Residual 被拆成 carry 与 branch  （图：fig_07_residual_principle.png）
10. 实验设计：对象固定，统计单位清楚  （图：—）
11. 先证明“算对了”  （图：—）
12. 同一张图，不同内部目标看到不同区域  （图：fig_08_internal_targets.png）
13. 发现一：依赖范围总体随深度扩展  （图：fig_12_spatial_broadening.png）
14. 一张图看完 12 层的变化  （图：fig_09_deit_s16_layerwise.png）
15. 发现二：深层 attention 更不能忽略 routing  （图：fig_16_attention_depth.png）
16. 发现三：两类 residual 的分工不同  （图：fig_18_residual_depth.png）
17. 独立队列仍复现主要趋势  （图：—）
18. 有限替换：Taylor 明显更适合回答“改了会怎样”  （图：fig_21_intervention_boundary.png）
19. 这套方法能说什么，不能说什么  （图：—）
20. 最后带走三句话  （图：fig_01_concept_overview.png）

## 40页详细版

01. 看懂 ViT 内部目标到底依赖了图像哪里  （图：fig_01_concept_overview.png）
02. 研究主线  （图：—）
03. ViT 内部解释为什么难？  （图：fig_11_module_learning_images.png）
04. 三类常见视图回答的是不同问题  （图：—）
05. 四个研究问题  （图：—）
06. 贡献在哪里？先把创新边界讲清楚  （图：—）
07. NNS 的一般计算图表示  （图：—）
08. 初始化：既保留数值，也保留位置  （图：fig_03_coordinate_initialization.png）
09. 传播：局部导数负责把坐标图送到目标  （图：fig_04_local_derivative_generation.png）
10. 命题 1：Endpoint 的数值身份  （图：—）
11. 输入坐标原点会影响 Endpoint  （图：—）
12. 双索引对象：位置 i × 使用边 e  （图：—）
13. 为什么必须记录具体 use？  （图：fig_02_neuron_learning_image.png）
14. 合法切面：每条路径恰好穿过一次  （图：—）
15. Attention 微分：routing 与 value 分开  （图：fig_06_attention_differential.png）
16. Q/K/V 分项怎样闭合？  （图：—）
17. Residual：全局切面必须把 bypass 算进去  （图：fig_07_residual_principle.png）
18. 四类对象不要混用  （图：—）
19. Target registry：我们到底扫描哪些目标？  （图：—）
20. 模型、图像与固定分析范围  （图：—）
21. 统计设计：把图像当作统计单位  （图：—）
22. 可视化只负责“看”，统计仍用原始数组  （图：fig_22_signed_and_blue.png）
23. 正确性验证覆盖了什么？  （图：—）
24. 同一输入，不同目标得到不同局部图  （图：fig_08_internal_targets.png）
25. DeiT-S/16：12 层 learning image  （图：fig_09_deit_s16_layerwise.png）
26. ViT-B/16：相同趋势，细节不同  （图：fig_10_vit_b16_layerwise.png）
27. 核心空间结果：越深，依赖通常越广  （图：fig_12_spatial_broadening.png）
28. 为什么空间结果需要双索引？  （图：—）
29. 同一层的 heads 也会分配到不同空间  （图：fig_13_all_heads.png）
30. Routing 与 value 的空间形状不同  （图：fig_15_routing_value_examples.png）
31. Attention 随深度：routing 上升，Value-only 误差变大  （图：fig_16_attention_depth.png）
32. Residual 分项：attention 与 MLP 是两种轨迹  （图：fig_17_residual_examples.png）
33. Residual 随深度：一个减，一个增  （图：fig_18_residual_depth.png）
34. 独立队列与四 checkpoint 稳健性检查  （图：—）
35. 坐标与质量敏感性：哪些结论稳，哪些会变？  （图：—）
36. 成分质量与有限替换效应：关系是“广泛但非普遍”  （图：—）
37. 同层 head 排序：加入位移后明显更好  （图：—）
38. 区域干预：Endpoint 接近随机，Taylor 更接近候选上界  （图：fig_21_intervention_boundary.png）
39. 方法局限：把外推边界说清楚  （图：—）
40. 结论：三层问题，三种答案  （图：fig_01_concept_overview.png）

