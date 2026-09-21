# 修改意见 v5 主张—证据矩阵

所有路径相对 vitnns_rebuild。旧确认性结论与新前瞻补充实验分开列示。

| 主张或边界 | 正文位置 | 紧凑证据 | 原始或既有证据 | 结论 |
|---|---|---|---|---|
| 基础学习图像是输入乘梯度 | 3.1–3.4 | evidence/revision_v5/protocol.json | reports/nns_vit_endpoint_m3_r1_20260919/m3a_fidelity.json | 继承已有基础量；增量不在新公式 |
| 原 NNS 式（10）/（11）的末端门语义 | 表 3-1、3.1 | revision_v5_reference_checks.json | 出版 PDF 第 5 页，SHA-256 绑定 | 激活前后目标分别注册 |
| 真实边切面闭合 | 4.3–4.8 | evidence/revision_v5/package_receipt.json | E3 闭合封存根 | 原切面保持通过；E2/E5 新增 40,000 项闭合检查 |
| 完整向量边的坐标不变性 | 4.6 | 解析推导 | 正文公式 | 区分边内变换、跨边混合和切面粒度 |
| 同目标带符号结构案例 | 图 4-3、4-4 | figures/revision_v5 的数组和收据 | E3 原始分项数组 | 两页最大原始误差不超过 1.37e-7 |
| MLP 神经元空间范围随深度扩大 | 6.8、表 6-8 | e2_summary.csv | E2 200 个有效任务 | 两模型块 0→11 熵增加、稀疏性下降 |
| 神经元类别一致性 | 6.7、表 6-7 | e1_r2_summary.csv、e1_r2_protocol.json | E1-R2 1000 个有效任务 | WordNet 匹配 16 项均未获支持；旧 RGB 匹配为敏感性诊断 |
| 原 Q/K/V 与残差层深规律 | 6.2–6.6 | evidence/revision_v4/ | v3/M4 封存结果 | 事后层序与原冻结分析身份分开 |
| 结构层深假设外部复现 | 8.1、表 8-2 | e5_image_values.csv、e5_summary.csv | E5 1600 个有效任务 | H1–H3 在四检查点均保持正方向 |
| 多幅度预测量身份 | 5.2、7.6、表 7-7/7-8 | evidence/revision_v5_existing/ | v3 Step 2 封存结果 | 参照位移感知泰勒；不需要路径图 |
| 原成分联合判据 | 7.3 | 既有 E6 判定 | family_decisions.json | C4a NOT_SUPPORTED；11/12 仅为族级操作规则 |
| 无参照激活乘梯度基线 | 7.3、表 7-1/7-2 | evidence/revision_v5_existing/m1_head_*.csv | M1 800 份图像结果 | 三组配对区间优于路径图，一组跨零 |
| 区域选择与随机修正 | 7.4、表 7-3–7-5 | 原发布补充证据 | publication_r2_20260920/evidence/supplement_v1/ | C4b 保持未获支持；ViT-B 两组劣于随机 |
| 神经元干预排序 | 7.5、表 7-6 | e3_*_summary.csv | E3 200 个有效任务、789,600 个后缀样本 | 相对激活乘梯度无稳定增量 |
| 神经元行为目标选择性 | 7.5 | e3_paired_summary.csv | 真实/高分非目标/随机非目标 | 未获支持 |
| 结构索引信息增量 | 8.1、表 8-1 | e4_models.csv、e4_paired.csv、拟合 NPZ | 严格 200/100/100 划分 | 无稳定测试增益；两检查点显著更差 |
| 头排序增量外部验证 | 8.1、表 8-2 | e5_summary.csv | 四检查点×400 图 | H4 四检查点均反向 |
| 模型、预处理和分类表现 | 5.1 | protocol.json、e5_profiles.csv | 原模型注册表 | 权重身份、训练来源、标准化、模糊与正确数均披露 |
| 完成性与版面 | README、审计文件 | package_receipt.json、qc/revision_v5_validation.json | qc/page_audit.json | v5 专用验收、PDF 和源稿绑定 |

大规模逐任务数组保留在运行根；紧凑包中的分析收据记录对应路径和 SHA-256。
