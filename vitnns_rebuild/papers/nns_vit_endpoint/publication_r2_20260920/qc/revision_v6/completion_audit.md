# Revision v6 prompt-to-artifact 完成审计

本轮目标是完成双语匿名稿、镜像、理论/图示修正、最低必要新实验、直接验证、六维终审，并在全部完成门通过后提交推送。阶段状态唯一入口仍为修改意见v6_修改实验终审一体化执行提示词.md。下表区分已有直接证据与尚待最终 Git 操作的条件，不以旧绿色收据替代当前检查。

## Prompt-to-artifact checklist

| 提示词要求 | 当前产物和直接证据 | 状态 |
| --- | --- | --- |
| 最终交付 1，中英文正文与镜像 | zh/en manuscript.md；上级 zh/en full_draft.md；逐字路径映射比较 | PASS |
| 最终交付 2，核心概念图和图注 | package figures 1/2/6/7 重画；主 Figures 1–11 图注；两个 figure manifests；visual_review.json | 已核验 |
| 最终交付 3，全局/分支/原生 use/可干预变量定义 | §4.1–4.3、四类对象表、重构目标表，A_F/L_F raw 数组复算 | 已核验 |
| 最终交付 4，坐标参数化 | 1,000 正式任务；27,600 image contrasts；96 summary rows；S17/coordinate_decisions.json | 已核验，DeiT-S Q 敏感性保留 |
| 最终交付 5，独立解析与 mutations | revision_v6_analytic.py；实际 scanner 对照 32 components/9 directions；4 mutations；原始 NPZ | 已核验 |
| 最终交付 6，数据模型总体统计身份 | §5、S15、analysis_identity.json、checkpoint/cohort bindings | 已核验 |
| 最终交付 7，有限替换伪代码 | §4.6、S16；A/V/head/residual tensors、slots、shape、reference、重算范围 | 已核验 |
| 最终交付 8，17 条处理表 | 唯一总表阶段 0 R1–R17，加本文映射 | 已核验 |
| 最终交付 9，六维终审 | thesis_final_review.md 六维每项问题/证据/影响/建议 | 已完成 |
| 最终交付 10，完成审计和机器报告 | 本表；final_validation.json；prompt_to_artifact.json | PASS |
| 边界 1–4，G×I 身份、branch/full 区分、A/V | §3.1、§4.2–4.3、§4.6；source 对照 | 已核验 |
| 边界 5，坐标原点 | §3.1/§6.7/S17；raw-gradient 旧列实际为 ∇x，已更名 | 已核验 |
| 边界 6–7，图像单位与独立性 | 主表 Inference、S15；独立参考仅 NumPy/math imports；mutation sum-preserving 攻击 | 已核验 |
| 边界 8–10，E5/H4/all-family | §6.6/§8，四 checkpoint H4 负；原 16 families/28 Holm 原样重算 | 已核验 |
| 边界 11–12，历史根不改、研究隔离 | 新结果根封存；旧 sealed evidence 未改；主文无 PCIA/PEFA/CST | 最终 diff 审计覆盖 |
| 阶段 0 必读 1–13 | stage0_source_audit.json 的文件身份，相关代码、双语正文/补充、v5 protocol/cohort/target/E5 | 已核验 |
| 阶段 0 在线文献 | 六份官方 arXiv HTML、reference_live_checks.json；CAAP v2 与 DOI 状态 | 已核验 |
| 阶段 0 R1/R2/R3/R14 门 | stage0_definition_evidence.json；同源代码/原数组/receipt 无冲突 | PASS |
| 阶段 1.1–1.3 理论与词表 | §4.2 branch contribution 与 global cut 公式、对象表、五类名称、Tables 6–10 分数公式 | PASS |
| 阶段 1.4 Figures 1/2/3/6/8/10 | 图 2 bypass；图 3 branch only；6/8/10 panel、n、聚合、误差带、分析身份 | PASS |
| 阶段 1.5 静态门 | stage1_validation.json 132 checks；新验证器另检当前全文 | PASS |
| 阶段 2 实验 A 原 100/400 图 | source/cohort freeze、1000 任务全目标/全坐标，不按 outcome 换图 | PASS |
| 阶段 2 实验 B CPU float64 | 单头、双头、pre-LN attention/MLP；affine/LN/softmax/QKV/concat/proj/residual 显式 Jacobian | PASS |
| 阶段 2 实验 C wrong-slot/fan-out | 正确 control 通过；总和保持的 Q/K 交换仍拒绝；提前合并或重复均拒绝 | PASS |
| 阶段 2 统计阈值 | 0.05/0.10/0.15 correlation、0/0.025/0.05/0.10 regional；68 行；不改原判定 | PASS |
| 阶段 2 资源/pilot | 4 pilot tasks、2808 retained raw maps、budget、GPU qualification/locks、CPU-only summaries | PASS |
| 阶段 3 必需文件 13 项 | protocol/source/environment/cohort；2 coordinate CSV；analytic/mutation；threshold；identity；manifest/validation/terminal | PASS |
| 阶段 3 closed schema、SHA、独立重算 | 原合同封存；artifact_schemas；完整 artifact_manifest；9062 checks；raw NPZ 重算 | PASS |
| 阶段 4.1 主结果总体 | Tables 4–10 图像、固定 target/checkpoint、身份/区间；Table 1 文献/Table 3 数值测试不作总体推断 | 已核验 |
| 阶段 4.2 数据细节 | S15 来源、选择、overlap、类别、误分类、种子、checkpoint、resize/crop/antialias/mean/std/blur、targets | 已核验 |
| 阶段 4.3 干预 1–10 | S16 clean/reference/mixed；tensor/shape/block/head/slot；clean siblings；重算；full/isolated；normalized zero=mu；reflect blur；pool/budget/ties/AUC/random | 已核验 |
| 阶段 4.4 RQ4 | 引言 RQ4、§4.6、§7，clean-only vs displacement-aware 信息预算 | 已核验 |
| 阶段 5.1–5.6 | 两点方法贡献；坐标范围；H4 六处；控制压缩；三位数字及保方向例外；双语一致 | 已核验 |
| 阶段 6.1–6.13 自动覆盖 | contracts + final_checks；独立实验 validator；旧资产/display 原始验证；Markdown/公式/引用/路径/镜像 | PASS |
| 强制 validate_gxi_step3.py | V6-A1 方法合同，历史记录不覆盖 | 已运行 PASS 54 |
| 强制 validate_gxi_step4.py | 原 E5/旧结果直接重算 + 当前文稿 | 已运行 PASS 967 |
| 强制 validate_publication_r2.py | 当前正文 + 22 图/92 补充图/真实 display arrays 原验证 | PASS |
| 强制 validate_gxi_step5.py | 当前六维/十答辩/负结果/双语覆盖 | PASS |
| 强制 pytest tests/test_revision_v5.py | 既有实验实现回归 | 已运行，与 v6 合计 11 PASS |
| 新 revision-v6 tests/validator | 独立解析 imports、实际 scanner、wrong-slot/fan-out、错误闭合文本 mutation | PASS |
| 所有新增修改 Python py_compile | 本轮清单由 git diff/untracked 定位后逐个编译 | PASS |
| git diff --check / --cached --check | 工作树已无空白错误；仅暂存本轮范围 | 暂存门待执行 |
| 阶段 7 六维及十答辩 | thesis_final_review.md 六维、证据错配表、四级风险、十个逐题证据 | 已完成 |
| 最终门 1–16 | 上述逐项产物；机器完成审计检查必需字段和直接哈希 | 最终科学审计 PASS，Git 门待执行 |
| 最终门 17 | 终审科学范围内阻断/重大为零，匿名稿选择已应用 | 当前满足 |
| 最终门 18 | 工作树/commit/push/local HEAD/tracking/ls-remote/LFS | 未执行，不宣称整体完成 |

## 终审结论与科学范围

六维评审对当前匿名科学稿判断为可送审。主结果限定 normalized-input Endpoint 和固定 targets/checkpoints。三坐标存在 Q 例外；H4 负结果与原 all-family 未获支持完整保留。该判断不授权跳过最后的命令、文件身份和 Git 核验。

## 最终发布审计记录

最后命令已通过。Git 操作完成后追加本地与远端身份记录。

最后一轮命令全部退出 0。Step3 54、Step4 971、publication 2678、Step5 997、v6 final 1075 项通过。Publication 独立重算 162 个实际展示 panel。revision_v5 6 tests、revision_v6 5 tests 全部通过。全部 24 个新增/修改 Python 文件编译通过，git diff --check 通过。命令、stdout、耗时及文件清单位于 command_results.json 与 py_compile_results.json。上述当前结果取代本表中此前“待最终运行”状态。仅剩 Git 暂存、提交和远端一致性门。
