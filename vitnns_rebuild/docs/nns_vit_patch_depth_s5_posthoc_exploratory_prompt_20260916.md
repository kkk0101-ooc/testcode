# S5 CPU-only post-hoc exploratory analysis prompt

> DRAFT HANDOFF — NOT AUTHORIZED BY THE CURRENT BLOCKED S4-R11 TERMINAL.

在 `/data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild` 执行 `NNS_VIT_PATCH_DEPTH_SEMANTICS` 的 S5 CPU-only post-hoc exploratory analysis。启动时必须读取并严格验证 `evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/terminal_receipt.json`：只有当 `SCIENTIFIC_STATUS=PASS_POST_HOC_EXPLORATORY` 且 `next_authorization=S5_CPU_ONLY_POST_HOC_EXPLORATORY_ANALYSIS_THEN_S6_EXPLORATORY_REPORTING` 时才可继续；否则只输出 `S5_NOT_AUTHORIZED`，不得读取任何 endpoint 数值。本草案对应的当前 S4-R11 terminal 为 `BLOCKED_DATA_INTEGRITY`，因此当前执行必须停在 `S5_NOT_AUTHORIZED`。

若未来由 owner 前瞻性修正 selection identity/class-count 合同并重新发布合法 PASS terminal，则 S5 仍必须遵循以下边界：

1. 全程设置 `CUDA_VISIBLE_DEVICES=''`；禁止 Torch、timm、torchvision、PIL、JPEG/mask decode、模型/checkpoint 加载、CUDA/GPU 初始化和任何新增干预。
2. 唯一科学输入是新 PASS S4-R11 terminal 绑定的 240-source manifest、terminal、P1/P2/D1 arrays，以及 S3/S4 metadata 所授权的 D2 数值；不得回退绑定 S4-R10，也不得读取未授权 source。
3. 明确披露 `post-hoc exploratory`：P2 的 `0.07` 点估计门、零效应 adjusted lower-bound 门和 `0.02` precision 门是在结果访问后指定。不得声称 preregistered、outcome-blind、confirmatory、strong FWER preserved 或原 S4-R10 `STOP_SUCCESS`。
4. producer 与 independent validator 必须分别从原始数组重算 P1/P2/D1/D2、image-level 值、equal-class aggregation、20,000-replicate within-class whole-image BCa、adjusted CI/p、validity/NA/matching denominators和最终 decisions；validator 不得信任 producer 的 PASS 字段。
5. P1、P2、D1 使用 S4-R11 探索性规则；D2 也只能作为 exploratory endpoint。原 P2 `0.10` effect rule 和 `0.015` half-width rule必须并列显示为 `ORIGINAL_RULE_NOT_PASSED`，不得隐藏或改写。
6. 每个 endpoint 输出 `n_total/n_valid/n_NA/n_failed`、estimate、ordinary BCa CI、family-adjusted CI/p、阈值、探索性 decision 和原规则 comparison；typed NA 不得删除、替补或填零。
7. 生成 endpoint decisions、depth profiles、intervention controls、validity accounting 和 claim ledger。所有表、标题、caption 与 claim 必须带 `post-hoc exploratory` 标签。D1/D2 不一致时不得写因果结论。
8. closed schemas、input hashes、whole-root inventory、false-pass mutation audit、CPU access audit、independent validation 和 terminal receipt 必须齐全。禁止执行 S6；只可在 S5 PASS 后给出 `S6_POST_HOC_EXPLORATORY_REPORTING_ONLY` 建议授权。
9. 完成后 commit/push 当前分支并验证 local HEAD、tracking ref、remote ref、git status、Git LFS status/fsck。

最终仅报告探索性结果和边界，不得把任何 S5/S6 产物重新包装为未窥视 confirmatory evidence。
