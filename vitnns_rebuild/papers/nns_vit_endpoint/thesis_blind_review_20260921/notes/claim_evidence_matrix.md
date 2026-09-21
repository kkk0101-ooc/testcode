# 修改意见 v4 结论—证据矩阵

所有路径相对 vitnns_rebuild。主要交付为 thesis_blind_review_20260921 下的双语稿；v4 采用最低送审路线。

| 主张/要求 | 定义或估计量 | 正文 | 证据 | 验证与边界 |
|---|---|---|---|---|
| RQ1 结构路径图组为输出 | Phi_C(F,x)，真实边四元组 | 1.3、4.3 | 双语式 4-2 至 4-4 | 真实使用实例；相关路径恰好穿过一次 |
| 基础数学前提 | 原算子映射、局部雅可比 | 表 2-1、表 3-2、命题 3-1 | reports/nns_vit_endpoint_m3_r1_20260919/m3a_fidelity.json；m3a_operator_validation.json | 5,000 比较与 88 算子检查限于第 3 章；原软件不可获得 |
| Q/K/V、头、残差闭合 | 式 4-7、4-8、4-9 | 4.4–4.8 | reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json；reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json | 76,800 正向、28,800 负控制、6,144 解析方向；不替代有限效用 |
| 分项的图依赖 | 等价隐藏坐标重参数化 | 4.6，式 4-10 至 4-12 | 解析构造 | 总和不变、分项改变 |
| RQ2 五组目标抽样 | 原合并单元阶段差与本轮逐块中位数分开 | 表 6-1、图 6-1、表 6-4 | reports/nns_vit_endpoint_revision_v3_step2_r2_20260921/independent_sampling_summary.csv；已绑定 NPZ | 五组共享 100 图；重分析逐个单元块复算 |
| 完整 12 层轨迹 | 每图趋势、逐块区间、阶段差、端点差、相邻差 | 图 6-1 至 6-5、表 6-4/6-5 | evidence/revision_v4 统计 CSV 与 98 族图册 | 事后描述性；98 族、30,200 轨迹；各族相邻 11 项 Holm |
| 查询熵先降后升 | 全层趋势与端点不同 | 6.3 | qkv_image_values.csv；trend_summary.csv；phase_endpoint_contrasts.csv | DeiT 查询趋势为正、端点区间跨零；不宣称单调 |
| 路由与残差阶段差异 | 图内阶段中位数配对差 | 6.4–6.5 | reports/nns_vit_endpoint_m4_r1_20260919/image_values.csv；m4_a1_image_values.csv | 中期到后期存在反转；每模型单独报告 |
| 参数/目标敏感性 | 参数状态交叉重采样均值；真实类别和最高预测均值 | 6.6、表 6-6 | publication_r2_20260920/evidence/supplement_v1/training_analysis.json；target_analysis.json | 2,000 次均值区间，不与中位数错配；无训练因果结论 |
| RQ3 多幅度效用 | 式 5-1/5-2，误差与头间排序 | 表 7-1/7-2、图 7-1 | publication_r2_20260920/evidence/revision_v3_step2/multi_amplitude_summary.csv；multi_amplitude_decision.json | 同方向五幅度；1,600 图像参照误差曲线非递减 |
| 原联合失败优先 | C4a 原 28 项联合分析中的 12 族 | 摘要、7.4、8.4、8.6 | reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json | NOT_SUPPORTED 在前，11 族满足操作规则在后；0.10 不是总体阈值检验 |
| 随机基线修正 | 式 7-1 两种绝对值/平均顺序 | 7.5、表 7-3 | publication_r2_20260920/evidence/supplement_v1/baseline_image_values.csv；原 E6 统计 | 原 C4b 保留；修正表为事后估计量敏感性 |
| 2026 文献定位 | Prisma、LeGrad、Vi-CD、CAAP | 第 2 章，[38-41] | notes/revision_v4_reference_checks.json | 原站元数据和摘要核对，版本及访问日明确 |
| 统计精度与缺失 | 400 图固定精度；全部种子/有效数 | 5.3–5.4、附录 A | manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json；original_test_family.csv；amplitude_missing_counts.csv | 无功效保证；单元缺失与图像缺失分开 |
| 可选增强路线 | 独立样本趋势、同架构状态、语义/选择性与基线 | 8.5 | 未来设计文本 | 未用未执行实验支撑当前主张 |

重分析数值校验为 scripts/validate_revision_v4_data.py；全文覆盖、双语表格与数值来源校验为 scripts/validate_thesis.py；实际 PDF 与稿件绑定由 qc/render_receipt.json、qc/page_audit.json 核验。
