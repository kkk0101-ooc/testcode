# G×I 五步计划第四步完成审计

## 目标与成功标准

目标是执行 `Endpoint_GxI创新定位与后续五步计划.md` 第四步：把现役中英文 Results 重排为身份/正确性、背景规律、双索引机制、跨 checkpoint 复现和行为标定五层；证明至少一个 attention 与一个 residual 结果需要 $[e,i]$ 联合信息；仅在 source binding、cohort 无重叠、checkpoint binding、统计合同和数字重算全部通过后同步 E5；为每个主结果绑定统计单位、效应量、区间、checkpoint、target 和封存证据；完整保留阴性与相反结果；不启动新实验。

## Prompt-to-artifact checklist

| 要求 | 直接产物 | 实际核验 | 状态 |
| --- | --- | --- | --- |
| 先报告等价复现、算子验证、闭合与 mutation | 双语 Section 6.1；Table 3 | 5,000 项公式回归、88 项算子方向、76,800 项正向闭合、28,800 项负控 | PASS |
| 总图/edge scalar/双索引图组差异表 | 双语 Table 4 | 三种 retained-index、可回答问题、冻结结果和证据角色逐项对应 | PASS |
| Attention 双索引必要性 | 双语 Section 6.4 | 每模型 400 图的 Q/K/V spatial entropy；Q 相反模型趋势保留；16 图 Full–Value TV 为描述性支持 | PASS |
| Residual 双索引必要性 | 双语 Section 6.5 | 每模型、每 residual 类型 400 图的 output–identity spatial TV；给出效应与区间 | PASS |
| 层深扩展降为背景，路径空间重组升为核心 | Sections 6.2–6.5 | 总图结果与 $[e,i]$ 结果分开陈述 | PASS |
| 替代解释审计 | Section 6.3 | 总 mass、activation norm、总图与 edge scalar 均按其可恢复信息说明 | PASS |
| E5 来源与 cohort | `evidence/gxi_step4_e5_audit.json` | protocol/cohort/targets hashes；400 E5 图与 500 E1 图及 1,626 个 prior exclusions 零重叠 | PASS |
| 四 checkpoint binding | 同上 | 四个权重的文件大小与 SHA-256 重算一致；新增两项标记为 training-recipe replication, not seed | PASS |
| E5 统计合同 | `analysis_estimands.json`；专项校验 | H1–H4、缺失值、400 图像单位、10,000 bootstrap、双侧符号检验和 checkpoint 内 Holm 一致 | PASS |
| E5 sealed JSON 与数字重算 | `e5_receipt.json`、`gxi_step4_numeric_recalculation.json` | 3,200 个源 JSON/NPZ 哈希通过；从 NPZ 重算 6,400 个 image-level 值，最大差 0；16 项汇总一致 | PASS |
| H1–H3 与 H4 同步报告 | 双语摘要、Section 6.6、Table 5、结论 | 12 项正方向和 4 项相反方向均保留 | PASS |
| 广泛但非普遍 | 双语 Section 7.1 与结论 | 11/12 family 与 all-family 未获支持同时保留 | PASS |
| Endpoint–Taylor 行为标定 | 双语 Section 7.2 | clean-only 与 displacement-aware estimand 明确分开 | PASS |
| 逐主张 evidence index | `evidence/gxi_step4_evidence_index.md` | 每项主结果含单位、范围、效应、区间、source 与角色 | PASS |
| 主文/补充分配表 | `evidence/gxi_step4_figure_allocation.md` | 现有 11 图、10 表和补充材料角色已固定 | PASS |
| 所有关键数字机器重算 | `evidence/gxi_step4_numeric_recalculation.json` | M3/M4/E3 封存摘要复核与 E5 原始数组级重算 | PASS |
| 双语现役稿和入口镜像 | `publication_r2_20260920/{zh,en}/manuscript.md`、`{zh,en}/full_draft.md` | 章节、公式、数据表和链接镜像校验 | PASS |
| PCIA–PEFA 与 CST 隔离 | 双语主稿专项扫描 | 现役正文无对应术语 | PASS |
| 无胜负导向新实验 | `gxi_step4_validation.json` | `new_experiments=0`；只读取和重算既有封存 evidence | PASS |

## 完成门审计

1. Attention 的 Full–Value patch TV 与 residual 的 output–identity patch TV 都需要 edge identity 和 input coordinate；总图或 edge scalar 不能直接恢复。
2. 每个主结果均在 `gxi_step4_evidence_index.md` 中绑定统计单位、效应量、区间、checkpoint、target 和封存路径。
3. E5 已通过 source binding、cohort 零重叠、checkpoint byte/hash、统计合同、逐任务哈希和原始数值重算，决定为 `SYNC_FOUR_CHECKPOINT_E5`。
4. 正结果、all-family 未获支持、H4 四 checkpoint 相反方向和模型依赖趋势均保留。
5. 本步未新增实验。

## 验证记录

- `validate_gxi_step4.py`：PASS，68 项检查，0 errors；逐字节核验 3,200 个 E5 源文件并重算 6,400 个图像级 H1–H4 值、bootstrap CI、符号检验和 Holm 决策，同时复核 M3/M4/E3 manifest bindings、背景层深统计、证据索引目标字段与双语连续表号。
- `validate_publication_r2.py`：PASS，1,733 项检查，0 errors；重算 162 个真实显示面板并覆盖 22 张图、10 张编号表、双语结构、链接与现役镜像。
- `validate_gxi_step3.py`：PASS，30 项检查，0 errors；第三步方法对象、切面和审计合同无回归。
- `verify_publication_r2_methods.py`：PASS，9 个 image-model 任务；Endpoint 最大绝对误差 0，Grad-CAM 最大绝对误差 $2.794\times10^{-9}$，rollout 最大绝对误差 $1.041\times10^{-17}$。
- `validate_endpoint_supplement_v1.py --paper`：PASS，32,091 项检查，0 errors。
- `tests/test_revision_v5.py`：6 passed；`thesis_blind_review_20260921/scripts/validate_revision_v5.py`：PASS，71 项检查，0 errors。
- `python3 -m py_compile` 与 `git diff --check`：PASS。

## 结论

第四步全部显式交付物和五项完成门均有直接产物及覆盖性验证。E5 同步不是基于旧稿或单一 PASS 标签，而是基于 protocol/cohort/checkpoint/source hashes 与原始数组重算；H4 阴性结果和原 all-family 未获支持结果均保留。第五步尚未启动。
