# Endpoint M5 复用分析执行与验收协议

本轮只执行《阶段M_endpointnnsvit.md》的 M5：方法语义、reference 属性与最小比较。M5 复用已封存的 M1、M2、M4/Review-R2 和 EP2-R1 证据，不读取或解码 JPEG，不加载模型/checkpoint，不导入 Torch/timm，不使用 GPU，也不进入 M6。

## 1. 前置门与不可变来源

1. M4 主根须为 `ENDPOINT_M4_PASS_EXECUTION_AND_AUDIT`，Review-R2 须为 `ENDPOINT_M4_REVIEW_R2_PASS_POST_HOC_SENSITIVITY`。
2. M1/M2 terminal receipt、artifact manifest 与本轮所读聚合/逐图文件须重算 SHA-256；M1 的 800 个 `per_image/*.json.gz` 还须逐项匹配 `metrics_manifest.json`。
3. M4 冻结配置绑定的是结果回填前的阶段文档。当前 live 文档的预期版本漂移不得写成 M4 科学失败；M5 通过 git object 校验执行前版本 `78b1c36f2:vitnns_rebuild/docs/阶段M_endpointnnsvit.md`。
4. 任一来源状态、文件集合或哈希不符即停止，不以现有表格替代原始来源。

## 2. 最小 reference 复用分析

固定 M1 的两模型、共享 400 图、12 个 blocks、全部 attention heads 与 ground-truth logit。对每图每层分别在 normalized-zero 与 blur-rgb 下计算：

- Endpoint 排名：`m = sum(abs(E))`；
- Taylor 排名：`abs(p_ref)`；
- 实际有限效应排名：`absolute_effect`。

随后对 12 层 Spearman 取图内均值，再对 400 图报告中位数和固定 10,000 次 image-bootstrap 区间。M5 报告：

1. Endpoint 的 `m` 与 signed sum `a` 在两 reference 间是否逐项完全相同；
2. Taylor 排名在两 reference 间的一致性；
3. 实际有限效应排名在两 reference 间的一致性；
4. Endpoint 与 Taylor 在每种 reference 下对实际效应的平均行为关联；
5. Taylor 相对 Endpoint 的 reference-specific 关联增量。

这些量是对既有 M1 数据的描述性再汇总，不新增假设检验或 multiplicity family。M2 只复算 patch geometry、blocks 2/5/10、full/isolated outcome 的 Endpoint absolute 与 regional Taylor AUC 中位数，沿用原 672 项检验结果，不重签新判定。

## 3. 方法语义与成本

必须交付五列方法适用性矩阵：内部标量目标、输入空间图、外部干预 reference、局部 Jacobian 结构闭合、具体 finite replacement 预测。Endpoint logit map 必须明确写成 normalized-input 坐标中的 Input × Gradient，不能作为独立输出归因算法与 Input × Gradient 比赛。

成本分为两类：

- 信息/操作需求：clean forward/backward、reference forward、路径积分点、候选反事实 forward、缓存对象；
- 既有 package 观测：M1 与 M2 的 GPU-hours、wall time、候选/条件数量。

M1/M2 的 package 成本不能拆成公平的逐方法 runtime，因此不签发 Endpoint 的运行时优势。EP2-R1 的 Path IG 失败保持原状：DeiT-S 在 512 点通过，ViT-B 在最大 512 点仅有 40/64 与 41/64 通过两种 target 门，低于每 target 61/64；不发布跨模型 Endpoint–Path 结论，不追加调参。

## 4. 交付与终态

必要交付：

- `reports/nns_vit_endpoint_m5_r1_20260919/reference_reuse_summary.json`；
- `reports/nns_vit_endpoint_m5_r1_20260919/reference_reuse_image_values.csv`；
- `reports/nns_vit_endpoint_m5_r1_20260919/m2_reuse_summary.csv`；
- `reports/nns_vit_endpoint_m5_r1_20260919/method_applicability_matrix.json`；
- `reports/nns_vit_endpoint_m5_r1_20260919/cost_and_information.json`；
- 中英文论文定位稿、证据矩阵、防御性写作审计和 M5 结果文档；
- 独立验证、prompt-to-artifact completion audit、artifact manifest 与 terminal receipt。

只有全部要求可由真实文件与重算证据覆盖时，才签发 `ENDPOINT_M5_PASS_EXECUTION_AND_AUDIT`，并保持 `next_authorization=STOP_AFTER_M5`。
