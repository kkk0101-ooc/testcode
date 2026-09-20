# 补充实验 v1 完成审计 — 2026-09-20

目标：针对当前 `papers/nns_vit_endpoint` 论文执行 `补充实验/补充实验v1.md`。最低完成条件是 P0 随机基线修正和跨层 M2 纳入，以及 P1 训练—随机初始化、目标稳健性、feature–Endpoint 定量关系；原文 P2/P3 增强项须明确区分已执行与未执行。

## Prompt-to-artifact checklist

| 原文要求 | 直接产物 | 覆盖与验证 | 状态 |
|---|---|---|---|
| 一、十一 A、十三 P0：每个 random repeat 先取 absolute effect AUC | `evidence/supplement_v1/baseline_analysis.json`、`baseline_image_values.csv`、S9、正文 Tables 5–7 | 从 800 个 E5 原始任务和 32,000 个 repeat-image 行重算；全部旧 estimator 图像分数误差 <1e-12；Jensen 差非负 | 完成 |
| mean absolute random AUC 与重复分布 | S9 表、`baseline_repeat_cohort_values.csv`、`baseline_repeat_image_values.csv` | 同时报告 corrected mean、median、q10/q90 和 20 个重复 | 完成 |
| Endpoint−random / Taylor−random 配对差 | 正文 Table 5、S9、Supplement Fig. 1 | 400 图 image bootstrap；修正后 Endpoint 原四个正差不再成立，Taylor 四组保持正差 | 完成 |
| 20 repeats 稳定性 | S9 稳定性表、`baseline_analysis.json` | per-image MC SE、5/10/20 重采样、split-half、leave-one-repeat-out；没有把 20 次写成普适充分 | 完成 |
| 十三 P0 / 八：blocks 2/5/10、全 heads 的 M2 | `m2_inherited.json`、S14 | 独立从 image values 重算全部 672 个 sign/Holm tests；full 11/12、isolated 12/12 偏向 Taylor | 完成；原 C4b 不回写 |
| 二、十一 1：两架构 × 3 random seeds × 32 图 | `training_analysis.json`、`training_*_block_values.csv`、Supplement Figs. 2–3 | 192 random tasks；589,824 M3 单元；全 12 blocks、16 tokens、16 channels、全部 heads；状态 digest 六个互异 | 完成 |
| M3 指标 | S10 training 表及完整分析 JSON | entropy、locality、distance、cancellation、CLS−patch entropy | 完成 |
| M4 指标 | S10 training 表及完整分析 JSON | routing scale、Value-only Endpoint error、8-direction JVP discrepancy、attention/MLP identity、TV、path cancellation | 完成 |
| seed 层级与 trained−random 比较 | `training_analysis.json` | 图像与 seed crossed bootstrap 2,000 次；保留三个 seed means；不把所有单元视为独立样本 | 完成 |
| 二 2.5 cascading randomization | S14 scope | 最低包外的增强项，没有写成已完成 | 未执行（P2/可选） |
| 三、十一 2：GT/top1/margin，100 图，B0/5/11，全部 heads | `target_analysis.json`、`target_block_values.csv`、Supplement Fig. 4 | 200 target tasks；正确/错误图为 92/8 与 86/14；三目标和全部注册指标均覆盖 | 完成 |
| routing、Value-only、JVP、saturation、residual/TV | S11 完整表和 JSON | native AV JVP 与 target-contracted JVP 分开报告；closures 和 detach 检查通过 | 完成 |
| 五、十一 3：feature–Endpoint 定量关系 | `feature_analysis.json`、`feature_unit_values.csv`、Supplement Fig. 5 | 12,288 energy targets；Spearman、cosine、JS distance、top-20 overlap | 完成 |
| wrong channel、cross-layer、spatial permutation、random channel 控制 | S12 表、`feature_analysis.json` | 61,440 pair metrics 独立复算；使用冻结 channels，无展示通道选择 | 完成 |
| 六 6.1：抽样敏感性 | `sampling_analysis.json`、S13 | channel/token/image bootstrap、leave-one-block、leave-one-stratum | 完成 |
| 六 6.2：两个独立 channel seeds | S14 scope | resampling 未伪装成独立 channel-seed 复现 | 未执行（P2） |
| 四：M3–M4 同目标桥接 | S14 scope | 当前 target/cohort 仍不匹配，不作逐单元或有限因果表述 | 未执行（P2） |
| 七：同架构替代 trained checkpoint | Discussion、S14 | 三种随机初始化不能替代不同训练 checkpoint | 未执行（P2） |
| 九：输入变换；十二 5：CNN 页 | S14 scope | 明确保留为可选项 | 未执行（P3） |
| 十：不继续扩大 leaderboard/Path-IG 调参/神经元穷举 | 协议与任务 manifest | 没有新增 Endpoint–Taylor 搜索、saliency leaderboard、百万目标或 Path-IG 调参 | 遵守 |
| 双语论文与图表 | `zh/en/manuscript.md`、`zh/en/supplement_experiments_v1.md`、5 张 PNG/PDF/SVG | 正文 5.6–5.9、Tables 5–7、Discussion/Conclusion 同步；视觉 contact sheet 人工检查 | 完成 |
| 可复现验证 | `validation_report.json`、`paper_validation_summary.json`、`native_verification.json`、`independent_statistics_verification.json`、`test_results.json` | 32,091 项最终 paper-aware 覆盖检查、16 项真实架构原生复算、63,448 项独立统计复算、15 tests | 完成 |

## 科学结论

- 随机基线修正提高四组 random AUC。Endpoint−random 配对中位差变为 −0.0023、−0.0031、0.0009、−0.0010；两个 DeiT 区间跨零。Taylor−random 四组仍为正。
- 随机初始化模型也呈空间扩展，说明架构/初始化已产生该局部 Jacobian 趋势；训练模型的 entropy 与 distance 增量更大、locality 下降更强。Routing 与 Value-only error 的训练后层深变化也明显强于随机状态。
- GT、clean-top1 和 margin 保留主要路径趋势方向，但幅度依赖目标。错误分类子集只有 8/14 图。
- 正确 feature–Endpoint 配对的图像级 Spearman 为 0.2663/0.2811，且高于四种负控制；前向响应与局部输入依赖相关但不等价。

## 验证边界与终态

原 E5/M2/M3/M4 封存根未修改。所有新增任务写入独立 supplemental root；生产 dtype、determinism、GPU identity、模型/图像 hash、QKV/residual closure、detach 和 JVP 检查均有记录。新结果是补充描述性分析，不追溯更改原 confirmatory C4a/C4b 决策。

执行中保留了两项预发布工程修正：初始 M2 选择器使用了不存在的 `patch_grid` 名称，核对注册表后改为实际冻结值 `patch`；便携清单补丁曾产生一次 Python 缩进错误，编译前即失败，未改写实验结果。另有一次 bootstrap Monte Carlo 索引对齐修正，使同尺寸配对摘要共享图像抽样索引；点估计、cohort、目标和模型运行均未改变。对应记录位于 supplemental root 的 `baseline_initial_schema_failure.json`、`analysis_bootstrap_alignment.json` 及会话命令输出。

最低补充实验包的每项 P0/P1 要求均有直接产物并通过验证，因此本目标达到完成条件。P2/P3 在原文中明列为增强/可选项，当前保持未执行状态。
