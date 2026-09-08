# 阶段 F：F3 工作包与产物索引

> 生效日期：2026-09-08。
> 目的：把历史上的 F3、F3 visual、F3H、F3C 统一解释为阶段 F 的 F3 工作包，降低命名复杂度，同时保持全部封存路径、hash、completion 和 archive ref 不变。

## 1. 规范名称

| 当前规范名称 | 历史 token | 作用 | 科学权限 |
|---|---|---|---|
| `F3-Core` | F3 | 冻结规则选例、方法示意、定性 NNS/matched panels、Figure 5/6 与 supplementary modules | display-only；不产生新 decision |
| `F3-Display` | post-F3 blue-display | 对 F3-Core 的 presentation-only deep-blue 视图 | shape-only；不比较跨面板幅值 |
| `F3-HeadAtlas` | F3H | 12×12 head atlas、signed components 与单个 Grad-CAM qualitative reference | descriptive；不同 target 不作数值优越比较 |
| `F3-ClassCompare` | F3C | 五方法 class-targeted/agnostic 比较、formal fidelity analysis、atlas/signed audit 和后续 display views | 仅限其冻结 conditional class-logit 协议 |
| `F3-Final` | 原计划 F4 中的图形部分 | 组合最终 Figure 1–6、Supplement S1–S4（以及明确纳入时的 S5）、captions 和 inventories | 只组合已验证模块；不写正文、不重算科学结果 |

F3H 和 F3C 中的字母是历史工作包代号，不是新的顶层阶段。新文档优先使用规范名称，并在首次出现时附历史 token。

## 2. 权威产物映射

### F3-Core

- terminal completion：`evidence/stage_f/f3_r7_recovery_v2_20260907/formal/completion/f3.json`
- completion SHA-256：`f1423479cc391b729cea37f34c663db833b91e3dff1cdee576efa61ce87e56d6`
- report root：`reports/stage_f_f3_r7_20260907/`
- archive ref：`refs/archive/stage-f-v1-f3-completion`
- archive commit：`1589fc3895657ee6a8f01d843ee26b67603998cb`

### F3-Display

- terminal completion：`evidence/stage_f/f3_visual_r4_recovery_v2_20260907/formal/completion/f3_visual.json`
- completion SHA-256：`0231b38a6886ab8e3b46b56e1d93928490fbe4dc3a858aa0c92781826177547d`
- report root：`reports/stage_f_f3_visual_r4_20260907/`
- archive ref：`refs/archive/stage-f-v1-f3-blue-display-revision`
- archive commit：`6ee82735be3b73448370ba0e71845b0e43cc4910`

### F3-HeadAtlas（历史 F3H）

- terminal completion：`evidence/stage_f/f3h_full_head_r3_terminal_seal_20260907/formal/completion/f3h.json`
- completion SHA-256：`26c47acc7729f20b3b22ab66f9a50ca67568fc67ecb9a939bee57c19ed6538c2`
- scientific/render payload：`evidence/stage_f/f3h_full_head_r2_cpu_reconcile_20260907/formal/`
- report root：`reports/stage_f_f3h_full_head_r2_cpu_reconcile_20260907/`
- archive ref：`refs/archive/stage-f-v1-f3h-completion`
- archive commit：`56afedd2e73a760dc5127ab4ee9c38d72623dbbc`

### F3-ClassCompare（历史 F3C）

- protocol：`evidence/stage_f/f3c0_protocol_v1_20260907/formal/completion/f3c0.json`
- implementation：`evidence/stage_f/f3c1_implementation_r6_cpu_seal_20260907/formal/completion/f3c1.json`
- pilot：`evidence/stage_f/f3c2_pilot_v1_20260907/formal/completion/f3c2.json`
- formal compute：`evidence/stage_f/f3c3_formal_v1_20260908/formal/completion/f3c3.json`
- terminal analysis：`evidence/stage_f/f3c4_analysis_r3_terminal_seal_20260908/formal/completion/f3c4.json`
- terminal analysis SHA-256：`4e5f8197d3dd1346aa935c522611c9b572be9013777d0c359acb50e42f0a64c1`
- scientific/render payload：`evidence/stage_f/f3c4_analysis_r1_20260908/formal/`
- report-side copy completion：`evidence/stage_f/f3c4_report_copy_r1_terminal_seal_20260908/formal/completion/f3c4_report_copy.json`
- report-side copy SHA-256：`b3838308f0d1b35037a765223d14d5bf5bed06e86a5b8c7e220ccc0dafb736e0`
- display-only 50-image package：`reports/stage_f_f3c_visual_v1_20260908/`
- display-only completion SHA-256：`e41c788058ae80c7834ffd457b0b1fcc03e016381a486940ceb605f028ae99c8`
- archive refs：`refs/archive/stage-f-v1-f3c0-protocol-freeze`、`refs/archive/stage-f-v1-f3c1-implementation`、`refs/archive/stage-f-v1-f3c2-pilot`、`refs/archive/stage-f-v1-f3c3-formal`、`refs/archive/stage-f-v1-f3c4-analysis-seal`、`refs/archive/stage-f-v1-f3c4-report-copy`

## 3. 为什么不移动 evidence 和 reports

上述路径已写入 source binding、schema constants、completion records、whole-root inventories、tests 和 archive commits。直接移动会改变路径、Git tree、文件 hash 关系及 validator 输入，使“整理目录”变成对封存证据的破坏性重写。

因此采用以下规则：

1. 旧路径永久保持只读；
2. 新文档和阶段 G 稿件使用本索引的规范名称；
3. 机器读取使用 `manifests/stage_f_f3_work_packages_v1.json`；
4. 如需方便浏览，只建立索引或 manifest，不复制或软链接 formal evidence；
5. 历史 receipts 中的 `F4_ONLY` 不修改；当前调度解释见 [阶段 F 规划](阶段F规划.md)和 [阶段 G 规划](阶段G规划.md)。

## 4. F3-Final 缺口

现有工作包已经提供统计模块、消融模块、定性案例、head atlas、class comparison、review copies 和 display-only views。旧 completion records 同时明确部分 final combined figures 尚未生成。因此阶段 G 启动前还需要一个不重算科学结果的 `F3_FINAL_FIGURE_PACKAGE_ONLY`：

- 从已验证模块组合最终 Figure 1–6 和 Supplement S1–S4；
- 决定 F3-ClassCompare 是否以 Supplement S5 或独立 atlas 附件纳入，选择规则不得依据“更好看”；
- 冻结 caption data、panel keys、figure inventory 和 source binding；
- 禁止模型加载、attribution/fidelity 重算、样本替换和统计修改；
- terminal PASS 后只授权 `STAGE_G_WRITING_G0_ONLY`。

本文只做分类与索引，不把任何历史工作包重新判定为 PASS，不代表 F3-Final 已执行。
