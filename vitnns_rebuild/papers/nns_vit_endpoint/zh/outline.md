# Endpoint 论文大纲（修改意见 v1 修订版）

[正文](full_draft.md) · [补充材料](supplement_revision_v1.md) · [英文正文](../en/full_draft.md)

1. 引言：NNS 内部目标与输入坐标的联系，四项贡献与 RQ1–RQ5。
2. 相关工作：NNS、G×I、IG/conductance、attention/relevance、activation patching。
3. 方法：列梯度约定；命题 1 及证明；非函数值分解与坐标原点；energy 聚合；attention output projection；命题 2 合法 cut 闭合及证明；residual cotangent；map/path cancellation；干预估计对象。
4. 实验：模型/数据/抽样；0.10/0.05 点估计门槛和相对零的符号检验；冻结与 post-hoc 分析。
5. 结果：RQ1 忠实性 → RQ2 registry → RQ3 正确性 → RQ4a 空间扩展 → RQ4b routing/value → RQ4c residual → RQ5a 成分标定 → RQ5b 空间干预 → reference 分析。
6. 讨论：Identity、Correctness、Utility；空间扩展与抵消；attention sharpness 的边界。
7. 局限：训练/初始化控制、GT/top1、同目标 bridge、同架构 checkpoint 与方法比较的缺口；抽样范围与固定干预任务。
8. 结论：忠实性、结构正确性、机制观察、行为边界。

正文 Figure 1–10 由 ../figure_manifest.json 绑定。旧 R2 Figure 1–40、全 head/module-chain、数值诊断和 post-hoc 页面全部在补充材料中索引。Figures 1、5、6、7、8、10 同时展示 signed/absolute；同组相关 maps 使用共享尺度及原始 L1/L2 标注。

Table 1 模型；Table 2 原 C4a 与 CI 下界敏感性；Table 3 补充 head 排序；Table 4 原 C4b；Table 5 绝对 AUC；Table 6 oracle regret/logit 比；Table 7 reference 排序。补充 Table S1–S5 保留完整 coverage、资源、变化量分布、排序分布和方法对象对照。
