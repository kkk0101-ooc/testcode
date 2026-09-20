# 补充实验 v1 执行协议

用户于本轮明确要求执行 `补充实验v1.md`，本协议将其第十一、十三节的最低包转为实际执行；先完成全部 P0/P1。旧论文修订的“仅写作”范围不适用于本轮。原 E5/M2/M3/M4 封存根不变，新增证据写入 `reports/nns_vit_endpoint_supplement_v1_20260920/`，最终复制精简证据和图表到当前 publication 目录。所有新统计为补充描述性分析，不能回写原确认性决策。

## 冻结设计

- 模型：原 ViT-B/16、DeiT-S/16 的精确 timm 架构、预处理、输入字节；训练模型绑定原 checkpoint。随机模型使用 timm 0.9.16 原生 `pretrained=False` 初始化，CPU seeds 202609201、202609202、202609203；同一随机状态跨所有图像，保存状态张量摘要。模型 eval、float32、禁用 TF32/autocast、确定性算法，导入 Torch 前设置 CUBLAS_WORKSPACE_CONFIG=:4096:8。
- 对照/feature cohort：既有 M3 cohort_order 0–31，共享 32 张图/模型。目标稳健性：既有完整 M3 100 张图/模型。没有根据预测或新增结果选择图像。
- M3 对照：全部 12 blocks，全部原 16 tokens（1 CLS + 15 空间分层 patch），每层从原冻结 32 channels 中按 SHA-256 namespace `ENDPOINT_SUP_V1_CHANNEL` 排序取 16 个，训练/随机使用同一列表。复用并严格筛选训练模型原单元数组；随机模型实际运行。指标：entropy、3×3 own-token locality、归一化距离、map cancellation、CLS−patch entropy。
- M4 对照：32 图 × 2 架构 × 3 随机种子，全部 12 blocks、全部 heads、GT logit。指标沿用原精确定义：routing scale、Value-only Endpoint error、8 个固定方向的 native attention JVP discrepancy、attention/MLP identity scale、output–identity patch TV、path cancellation。训练结果来自同图的原 M4。
- 目标稳健性：100 图 × 2 模型，blocks 0/5/11、全部 heads，GT、clean top1、clean top1−runnerup margin。预测索引在 clean 输入固定；正确分类图 GT=top1，复用同一结果；错误分类图单独计算 top1。复核 routing、Value-only error、residual 两类型 identity/TV、saturation relation。原 M4 的 native AV JVP 与类别目标无关，按同图同 block 复用并明确其适用范围；另计算方向导数中目标 cotangent 收缩后的 Value-only discrepancy，区分两者。
- Feature：32 图/模型，全部 12 blocks，使用上述 16 个冻结 channels。energy 为 196 个 patch tokens 的半平方和（不含 CLS，与现有展示一致），q=activation²/总和，p=输入原始 RGB Endpoint absolute mass 的 patch 和/总和。保留 q/p 数组；比较 Spearman、cosine、Jensen–Shannon distance（自然对数）、top-20 overlap。负控制：同层冻结列表循环错位 channel、同 channel 下一层循环配对、固定 patch 置换、同层全维中 hash 选定另一随机 channel。明确这些相关性不是量值完整性或因果证据。

## 统计与数值验收

先在图内聚合，M3 为 patch token/channel 中位数，CLS 单独；M4 为 head 中位数后 block 中位数。early=0–3，late=8–11；目标实验 early=0、late=11。报告每种状态的层趋势、训练−随机值及 late−early 差分的差分。随机对照保留种子层级：先按每种子计算同图配对效应，同时对 3 seeds 和共享 image indices 独立有放回采样（crossed bootstrap，2,000 次），报告均值及 95% percentile CI；只有 3 seeds，CI 不代表全部初始化分布。其它效应按 image bootstrap 2,000 次，同一比较共用图像索引，原始图像表可重算。不给所有单元独立样本地位，不进行重复试验后的门槛调整。非有限值失败，零分母显式 NA。

随机基线以每个 repeat 的 absolute AUC 为单位，再平均 repeats/heads；与旧 abs(mean signed curve) 同时列出，重算 Endpoint−random、Taylor−random、regret。保留 20 repeats 的 cohort 分布、每图 MC 标准误，比较 5/10/20 repeats 的重采样误差，检查二等分与 leave-one-repeat-out；若 20 次仍不稳定，如实报告，不声称足够。原 Endpoint−Taylor 结果不变。

源数据逐文件 SHA-256 验证，原封存 manifest 和 canonical digest 复核。新 target 内核须通过 native forward parity、Q+K+V=head、两类 residual closure、attention-detach Value-only、margin=top1−runnerup 梯度及方向导数一致性；继承原误差容限，不放宽失败记录。GPU 启动记录 UUID、空闲内存、利用率、已有 PID；动态全局 SQLite 队列，单 GPU 锁，matmul/backward startup gate。失败尝试保留，不将运行成功等同于科学支持。

## 源建议逐项处置与完成要求

| 原文项目 | 本轮交付 |
|---|---|
| 一 / 十一 A / 十三 P0 | 修正 AUC，repeat 分布，两配对差，20 repeats 稳定性及论文数字 |
| 二 2.1–2.4 / 十一 1 | 两模型、3 seeds、32 图、12 blocks、M3/M4 指标、分层比较 |
| 二 2.5 cascading | 可选增强；最低包不包含，保留为未执行，不能写已完成 |
| 三 / 十一 2 | 100 图、三目标、0/5/11、全部 heads、正确/错误子集、所有核心指标 |
| 四 / 十二 1 | P2 同目标桥接，最低包之后另行考虑；本轮不声称 M3–M4 因果联系 |
| 五 / 十一 3 | energy q/p、四相似度、四负控制、冻结随机 channels、明确结论边界 |
| 六 6.1 | 使用现有 M3 单元记录补充 channel/token/image bootstrap、leave-one-block/stratum-out |
| 六 6.2 / 十二 2 | P2 独立 channel-seed 重跑未执行，不能以 resampling 替代其状态 |
| 七 / 十二 3 | P2 替代 checkpoint 未执行，保留限制 |
| 八 / 十三 P0 | 复核已封存 blocks 2/5/10 全-head M2，正文简报、补充材料完整表 |
| 九 / 十二 4 | P3 输入变换可选未执行 |
| 十 1–4 | 不扩展胜负条件、排行榜、神经元穷举、Path-IG 调参 |
| 十二 5 | P3 CNN 展示可选未执行 |
| 最终建议 | 更新中英文正文/补充材料、数据图表、证据矩阵、逐项完成审计、验证命令 |

只有 P0/P1 全部实际完成且审计有直接证据时，最低补充包才可标记完成。P2/P3 不得混写成已完成实验。
