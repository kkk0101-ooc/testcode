# NNS_VIT_PATCH_DEPTH_SEMANTICS S5-R1 post-hoc exploratory execution

在 `/data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild` 执行 S5 CPU-only post-hoc exploratory analysis。本任务只能在下列启动门全部通过后执行：

1. 读取 `evidence/nns_vit_patch_depth/s4_r12_selection_anchor_recovery_20260916/terminal_receipt.json`，重新计算文件 SHA-256 和 canonical digest。
2. 验证 terminal 的 `status` 与 `SCIENTIFIC_STATUS` 均为 `PASS_POST_HOC_EXPLORATORY`。
3. 验证 `FORMAL_CONFIRMATORY_GATE_OPEN=false`、`EXPLORATORY_FIRST_LAYER_GATE_OPEN=true`。
4. 验证 `next_authorization=S5_CPU_ONLY_POST_HOC_EXPLORATORY_ANALYSIS_THEN_S6_EXPLORATORY_REPORTING`。
5. 验证当前 branch 的 local HEAD、tracking ref 与 live remote 均包含 R12 publication commit。

任一项失败时只输出 `S5_NOT_AUTHORIZED`，不得读取 endpoint arrays、R10/R11 outcome values 或 D2 prediction/error values。

## 性质与披露

本轮及所有下游结论必须始终标记为 `post-hoc exploratory`。P2 的点估计门 `>=0.07`、零效应 family-size-three L1 adjusted lower-bound 门 `>0` 和 ordinary BCa half-width 门 `<=0.02` 是在 outcome access 后指定。不得声称 preregistered、outcome-blind、confirmatory、strong-FWER-preserved，或原 S4-R10 `STOP_SUCCESS`。

原 P2 规则必须并列报告为 `ORIGINAL_RULE_NOT_PASSED`：

- point estimate `>0.10`：未通过；
- adjusted lower bound `>0.10`：未通过；
- ordinary BCa half-width `<=0.015`：未通过。

## 执行隔离

- 在任何依赖导入前设置 `CUDA_VISIBLE_DEVICES=''`。
- 禁止 Torch、timm、torchvision、PIL、JPEG/mask decode、模型/checkpoint 加载、CUDA/GPU 初始化和新增 intervention。
- 使用命令级 read whitelist，仅允许读取 R12 terminal 绑定的 240-source manifest、对应 terminal、P2 plan、原始 `.npy` arrays，以及 R12 明确授权的 S2/S3 metadata 和 D2 数值。
- 不读取任何未绑定 source；不回退绑定 R10/R11 terminal 作为 S5 授权。
- 不修改 S0--S4 evidence/artifact roots。

## 独立重算

producer 与 independent validator 分别从原始 arrays 重算：

- P1、P2、D1 的 image-level values、equal-class aggregation、valid/NA denominators；
- P2 matched-control denominator；
- 20,000-replicate within-class whole-image BCa；
- family-adjusted CI、heterogeneous-null adjusted p-values及 post-hoc decisions；
- 经 R12 授权的 D2 endpoint，只能作 exploratory analysis；
- R12 的 122 selected distinct class ranks、P1/D1 120 valid classes、P2 119 valid classes；
- full records digest 与三字段 membership projection digest。

validator 不得读取或信任 producer 的 PASS 字段来代替重算。typed NA 不得删除、替补或填零；INVALID endpoint 不得从 family 消失。

## 输出

至少生成：

- endpoint decisions：endpoint、n_total/n_valid/n_NA/n_failed、estimate、ordinary BCa CI、adjusted CI/p、post-hoc threshold、exploratory state；
- original-vs-posthoc P2 comparison；
- depth profiles 与 D1/D2 关系；
- intervention controls 与 matching diagnostics；
- validity accounting；
- claim ledger，逐项列出允许措辞与 evidence path；
- closed schemas、input inventory、CPU access audit、independent validation、false-pass mutation audit、whole-root inventory 和 terminal receipt。

所有标题、表格、图注和 claim ledger 必须保留 `post-hoc exploratory` 标签。不得把 probe 显著写成 NNS 显著，不得把深度相关写成因果形成机制。

## 终态与下一步

S5 完成后只允许签发探索性状态。若 S5 integrity 和预定探索性分析均完成，可建议：

```text
S6_POST_HOC_EXPLORATORY_REPORTING_ONLY
```

本轮不得执行 S6。完成后 commit/push 当前分支，并验证 local HEAD、tracking ref、live remote、git status、Git LFS status 和 `git lfs fsck`。
