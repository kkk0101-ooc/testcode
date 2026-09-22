# G×I 五步计划第五步完成审计

## 目标与成功标准

第五步目标是形成以 `NOV-JOINT-1` 为唯一方法新颖性主轴的中英文投稿稿。两版须同步改写标题、摘要、引言、贡献、相关工作、讨论和结论；数字、引用、图表和证据边界须一致；四项对抗性问题和最强近邻反例须形成可追踪修订；六维硕士毕业论文终审须给出送审结论；所有阻断毕业和重大修改问题须修复或通过收窄主张处理。

## Prompt-to-artifact checklist

| 显式要求或完成门 | 直接产物 | 实际核验 | 状态 |
| --- | --- | --- | --- |
| 同步重写中英文标题 | 双语稿第 1 行 | 两版均直接命名 dual-indexed local Jacobian 与 structural audit | PASS |
| 同步重写摘要 | 双语摘要 | 均直接给出 $\mathcal L_{e,i}(s;x)$、$e/i/s$、valid cut、审计和主要正负结果 | PASS |
| 同步重写引言与贡献 | 双语 Section 1 | 引言正面定义两个自由索引；贡献二直接定义技术增量 | PASS |
| 同步重写相关工作 | 双语 Section 2 | Table 1 加最强近邻逐字段反例和主张失效条件 | PASS |
| 同步重写讨论与结论 | 双语 Sections 8.1–8.2 | closure、空间机制和 finite replacement 分层，H4 阴性保留 | PASS |
| 摘要与贡献二不用形容词替代对象 | 双语摘要、贡献二 | 公式、索引、切面、检验和结果均可独立读取 | PASS |
| 主张—证据—简洁边界 | 主稿与防御性写作审计 | 评分 1/10；reviewer-facing 固定构式计数为 0 | PASS |
| 样本量、覆盖率、效应、区间、任务数和模型范围 | 第四步重算、第五步校验 | 1,600 个 E5 任务、6,400 个值重算；主表和补充表双语数字一致 | PASS |
| 是否只是 G×I 加 hooks | `gxi_step5_adversarial_review.md` | 双索引定义、use identity、valid cut、Figure 1 和 Table 4 共同回答 | PASS |
| 是否只是已有组件拼装 | 模拟审稿；创新条件清单 | Conductance、DecompX/LRP、EAP/Vi-CD、Unpack 逐字段反例 | PASS |
| 闭合是否误写成因果正确 | 双语讨论、结论、模拟审稿 | correctness 与 finite intervention utility 分开 | PASS |
| 经验结果是否需要双索引 | Table 4；Sections 6.4–6.5 | Q/K/V entropy 与 output–identity TV；每模型 400 图 | PASS |
| `NOV-JOINT-1` 逐字段反例 | 双语 Section 2.5 | 六个联合字段和覆盖即失效条件齐全 | PASS |
| 正文、附录、图注和表格双语一致 | `validate_gxi_step5.py` | 章节、公式、主表逐行、图路径、补充标题/数字/图路径均一致 | PASS |
| related-work matrix 一致 | 双语 Table 1 | 主表逐行 byte-level 一致 | PASS |
| claim-to-evidence 对照表 | `evidence/gxi_step5_claim_to_evidence.md` | 每项含位置、证据、统计单位、范围和反证边界 | PASS |
| 数字与引用自动校验 | 第四步重算；第五步校验 | E5 raw 最大差 0；所有列名作者在双语稿与参考文献中出现 | PASS |
| 模拟审稿与逐项修订记录 | `qc/gxi_step5_adversarial_review.md` | 六项攻击、四个强制问题和最强近邻均有修订后裁定 | PASS |
| 创新成立与失效条件一页清单 | `evidence/gxi_step5_novelty_conditions.md` | 对象、合同、审计、必要性、文献与复现条件齐全 | PASS |
| 六维硕士毕业论文终审 | `qc/gxi_step5_thesis_final_review.md` | 六个固定维度均含问题、证据、影响和修改判断 | PASS |
| 证据与结论不匹配表 | `qc/gxi_step5_evidence_mismatch_and_risk.md` | 含位置、当前结论、证据、缺口、允许表述和状态 | PASS |
| 分级问题清单 | 同上 | 阻断毕业、重大修改、一般修改、修辞优化四级齐全 | PASS |
| 一句话回答新增内容 | claim-to-evidence；双语摘要 | 固定为目标条件化双索引对象、合法切面、结构审计和机制用途 | PASS |
| 一张图回答 G×I 加 hooks | Figure 1 | 展示总图、$[e,i]$ 展开、合法切面、机制问题和索引退化 | PASS |
| 四个强近邻的 estimand/index/closure 差异 | Table 1；模拟审稿 | Conductance、DecompX/LRP、EAP/Vi-CD、Unpack 均覆盖 | PASS |
| 中英文新颖性句均回链 `NOV-JOINT-1` | 双语主稿专项扫描 | 两版各 5 处锚点；无 first/首次/首创/novel method | PASS |
| 四 checkpoint 数字只在审计后进入主稿 | 第四步 E5 audit | source、cohort、checkpoint、统计合同和原始数组重算均 PASS | PASS |
| 发布、重算、引用、双语、Markdown/LaTeX | `validate_publication_r2.py`、`validate_gxi_step4.py`、`validate_gxi_step5.py` | 全部通过，见验证记录 | PASS |
| 终审给出结论和最小条件 | `gxi_step5_thesis_final_review.md` | 结论为 `可送审`，列出四项维持条件 | PASS |
| 阻断毕业和重大修改清零 | 风险清单、终审 | 无阻断项；三项重大问题全部修复 | PASS |
| 不追加胜负导向实验 | 第四、五步校验元数据 | `new_experiments=0` | PASS |

## 完成门复核

1. **一句话回答。** 既然 Endpoint 等于 G×I，新增内容是目标条件化双索引局部微分对象、合法切面合同、结构审计和机制用途。
2. **G×I 加 hooks 反例。** Figure 1 给出索引与切面退化；Table 4 和 Sections 6.4–6.5 给出 attention 与 residual 的双索引必要性数据。
3. **最强近邻。** Conductance、DecompX/LRP、EAP/Vi-CD 和 Unpack 已按 estimand、索引和闭合对象逐字段比较。
4. **新颖性锚点。** 中英文方法贡献句统一回链 `NOV-JOINT-1`，没有单项首次主张。
5. **四 checkpoint 门。** E5 通过 source binding、cohort 零重叠、checkpoint hash、统计合同和 6,400 个图像级值重算后才进入主稿；H4 四项反向同步保留。
6. **发布校验。** 主发布校验、第四步重算、第五步双语/引用/Markdown/LaTeX 校验全部通过。
7. **硕士终审。** 六个固定维度、证据错配表、分级风险、送审结论和最小条件齐全。
8. **重大问题。** 入口未定义双索引、强近邻失效条件不显式、closure 与 finite utility 收束不清三项均已修复；无未解决的阻断毕业或重大修改项。

## 审计结论

第五步的显式工作、交付物和八项完成门均有直接产物覆盖。终审裁定为 `可送审`，但该裁定只针对学术内容。作者、单位、通讯作者、repository/archive 和学校模板仍属于投稿或学位流程元数据。

## 验证记录

- `validate_gxi_step3.py`：PASS，30 项检查，0 errors。
- `validate_gxi_step4.py`：PASS，68 项检查，0 errors；E5 决定为 `SYNC_FOUR_CHECKPOINT_E5`。
- `validate_publication_r2.py`：PASS，1,733 项检查，0 errors；重算 162 个真实显示面板。
- `validate_gxi_step5.py`：PASS，411 项检查，0 errors；覆盖总计划第五步状态、14 项计划验收、11 张主图、10 张主表、92 页补充图、1,600 个 E5 任务和 6,400 个重算值。
- `tests/test_revision_v5.py`：6 passed。
- 四个校验器 `py_compile`：PASS。
- `git diff --check`：PASS。
