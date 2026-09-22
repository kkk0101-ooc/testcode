# G×I 五步计划第五步 claim-to-evidence 对照表

本表绑定第五步现役中英文稿。方法新颖性统一回链 `NOV-JOINT-1`。基础 G×I、链式法则、单独的 edge、Q/K/V、residual 分解或 conservation 均不构成单项新颖性主张。统计单位中的图像数指独立图像级分析单位。

| Claim ID | 双语正文位置 | 允许主张 | 直接证据 | 统计单位与范围 | 边界或反证 |
| --- | --- | --- | --- | --- | --- |
| MI-1 | 摘要；Sections 1、3.1、8.2 | 精确当前点导数且无额外重分配时，标量 NNS 学习图像等于同目标 G×I | `m3a_fidelity.json`；5,000 项公式回归 | 五类兼容网络；float64/float32 | 数学身份澄清，不是新公式 |
| NOV-JOINT-1 | 摘要；Sections 1、2.5、4.1–4.3、8.1–8.2 | $\mathcal L_{e,i}(s;x)$ 同时保留输入坐标、具体 consumer-use edge 与目标条件，合法切面恢复同一 current-point local differential | 命题 2；Table 1；Sections 4.1–4.3；Figure 1–2 | 指定计算图、切面、参数化和注册目标 | 任一既有方法若覆盖全部联合字段，方法新颖性即失效 |
| NOV-AUD-1 | Sections 4.3、5.3、6.1 | 已封存审计可检出三类遗漏或重复记账错误，并由独立数值 oracle 复核 | 76,800 项闭合；28,800 项 mutation；6,144 项解析 JVP；有限差分 | 两个原始 checkpoint，400 图 | wrong-slot、merged fan-out、detach mutation 未作为已验证能力 |
| EF-ATTN | Sections 6.3–6.4；Table 4 | Q/K/V 空间分布具有不同层深轨迹；Full–Value 空间差异提供描述性支持 | `qkv_depth_contrasts.csv`；`bridge_block_summary.csv` | Q/K/V 为 400 图/模型；Full–Value 为 16 图/模型 | Q 方向具有模型依赖性；routing share 仅为 edge-scale 支持 |
| EF-RES | Sections 6.3、6.5；Table 4 | Attention 与 MLP residual 的 output–identity 空间差异呈相反层深轨迹 | `residual_spatial_tests.json` | 400 图/模型/residual 类型 | identity share 不能单独支持空间重组 |
| EF-E5 | 摘要；Sections 5.1、6.6、8.2；Table 5 | H1–H3 在四个公开 checkpoint 保持方向 | `e5_summary.csv`；`gxi_step4_e5_audit.json` | 400 图/checkpoint；4 checkpoints | 参数状态复现，不识别训练因果，也不是独立训练种子 |
| IB-H4 | 摘要；Sections 6.6、7.1、8.2；Table 5 | 路径图相对 activation × gradient 没有稳定 head 排序增量 | `e5_summary.csv`；第四步数组级重算 | 400 图/checkpoint；4 checkpoints | 四项均为相反方向，不能从复现结果中剔除 |
| EF-FINITE | Sections 7.1–7.2；Tables 6–10 | 路径图与有限成分效应关联广泛但非普遍；reference-matched Taylor 更适合规定区域替换排序 | `family_decisions.json`；`spatial_image_values.csv`；`analysis.json` | 400 图/模型；冻结 references、regions、budgets | all-family 结论未获支持；比较不构成无条件方法排名 |

## 一句话方法定位

既然 Endpoint 等于 Gradient × Input，本文新增的是对注册目标同时保留输入坐标和具体 consumer-use edge 的 $\mathcal L_{e,i}(s;x)$，以及使该图组在合法切面上恢复同一总局部微分的合同、结构审计和由此支持的 attention/residual 空间机制分析。

## 中英文主张同步规则

- 两版标题、摘要方法句、贡献二、相关工作结尾、讨论首段和结论第二点均使用 `NOV-JOINT-1` 的同一对象与边界。
- 任何 `first`、`novel`、`首次` 或 `首创` 句均禁止进入主稿。
- E5 的 H1–H3 与 H4 必须同处摘要、Table 5 和结论。
- 结构闭合、局部尺度和有限替换分别对应正确性、描述性结构量和规定干预效应，不得互相替代。
