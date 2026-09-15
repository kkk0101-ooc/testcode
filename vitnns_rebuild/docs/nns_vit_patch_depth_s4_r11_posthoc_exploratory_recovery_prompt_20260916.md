# S4-R11 post-hoc exploratory recovery execution prompt

在 `/data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild` 执行本任务。唯一目标是：安全停止仍在运行的 S4-R10 L1 生产进程（若已退出则记录 no-op），保留 S4-R10 全部历史证据不变，固定最早完成的 240 张 L1 source 作为一次明确的 post-hoc exploratory 分析，在新的 S4-R11 evidence root 中独立复算并验证 P1/P2/D1，按本文的新探索性规则签发真实的 `PASS_POST_HOC_EXPLORATORY`，然后只生成可供探索性 S5/S6 使用的交接授权；本任务不执行 S5 或 S6。

## 1. 性质与允许声明

本轮是已查看结果后的探索性分析，不是 preregistered、confirmatory 或 outcome-blind 分析。必须明确记录：

- S4-R10 已发生 P1/P2/D1 aggregate outcome access，不能再声称“全程未窥视”，也不能将本轮状态写成原协议的 `STOP_SUCCESS`。
- P2 的阈值、证据规则和精度规则是在查看约 237--242 张结果后由 owner 指定的 post-hoc 调整。
- 允许状态为 `PASS_POST_HOC_EXPLORATORY`、`FAIL_POST_HOC_EXPLORATORY` 或 `BLOCKED_DATA_INTEGRITY`。
- PASS 只表示本文定义的探索性门槛通过；不得称为原 S4-R10 的 confirmatory PASS，不得声称强 FWER 或预注册证据得到保持。
- 后续 S5/S6 可以继续，但必须标记为 exploratory，并绑定新的 S4-R11 terminal；不得把 S5/S6 结果重新包装成未窥视的 confirmatory 结果。

## 2. 必读输入与不可变历史

开始前完整读取并核验：

- `docs/nns_vit_patch_depth_s4_group_sequential_recovery_20260915.md`
- `evidence/nns_vit_patch_depth/s4_r10_group_sequential_20260915/implementation_seal.json`
- `evidence/nns_vit_patch_depth/s4_r10_group_sequential_20260915/group_sequential_recovery_amendment.json`
- `evidence/nns_vit_patch_depth/s4_r10_group_sequential_20260915/statistical_analysis_plan.json`
- `evidence/nns_vit_patch_depth/s4_r10_group_sequential_20260915/critical_boundary_table.json`
- `evidence/nns_vit_patch_depth/s4_r10_group_sequential_20260915/look_eligibility_report.json`
- `manifests/nns_vit_patch_depth/s4_r10_group_sequential_schedule.json`
- `src/vitnns/nns_patch_depth_s4.py`
- `src/vitnns/nns_patch_depth_s4_group_queue.py`
- `src/vitnns/nns_patch_depth_s4_group_sequential.py`
- `scripts/adjudicate_nns_vit_patch_depth_s4_group_sequential.py`
- `docs/nns_vit_patch_depth_execution_plan_20260914.md` 的 S5/S6 部分。

把以下目录视为不可变历史，禁止覆盖、删除、重命名或回写：

- `evidence/nns_vit_patch_depth/s4_r10_group_sequential_20260915/`
- `artifacts/nns_vit_patch_depth/s4_r10_group_sequential_20260915/`
- 所有 S4-R4 及更早 evidence/artifact root。

新产物使用 fresh roots：

- `evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/`
- 如确有必要，使用 `artifacts/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/`；优先通过带 SHA-256 的 manifest 引用既有数组，不复制多 GB payload。

## 3. 安全停止 S4，绝不影响 Y4

1. 先用只读进程检查定位 S4-R10 launcher/worker。候选进程必须同时满足命令行含 `nns_vit_patch_depth_s4` 或 `s4_group_sequential`，并能绑定到本仓库及 `look-1-launch-001`。
2. 单独盘点所有 Stage Y/Y4 进程。任何命令行含 `stage_y`、`Y4`、`y4_r` 或指向 Stage Y evidence/artifact root 的进程均属于排除集合，禁止发送任何信号。
3. 只有经过 PID、`/proc/<pid>/cmdline`、`/proc/<pid>/cwd` 和父子关系交叉核验的 S4 PID 才可收到 `SIGTERM`。先停 launcher，再等待 worker 正常退出；只有仍存活且身份再次核验一致的 S4 PID 才可进一步处置。禁止 `pkill python`、`killall`、宽泛正则 kill 或任何可能命中 Y4 的命令。
4. 若没有活动 S4 PID，不执行 kill；将状态记录为 `ALREADY_EXITED_NO_SIGNAL_SENT`。当前 2026-09-16 06:39 CST 的只读检查未发现活动 S4/Y4 进程，但执行者必须以执行时现场为准。
5. 保存 `s4_r10_interruption_audit.json`，记录检查时间、候选 PID、已核验 PID、信号、退出状态、Y4 排除 PID、`Y4_signal_count=0`、S4-R10 文件写入数为 0。

停止进程不是删除数据。保留已经完成的所有 S4-R10 terminal、source completion、claim、log 和 payload。停止后不得等待或补算到 400 张。

## 4. 固定 240 张探索性快照

从原 L1（`formal_class_rank < 200`）中，只选择三类 unit 均已有 resolved terminal 的 source。每个 source 的完成时刻定义为三个 terminal 的 `payload.completed_unix_ns` 最大值，按以下不可变键升序排列：

```text
(source_completion_time_max_unix_ns, schedule_index)
```

精确取前 240 个 source。禁止按 P1/P2/D1 数值、状态优劣、类别效果或图像外观挑选；第 241 个及以后完成的 source 不进入本轮分析，但仍保留在 S4-R10。生成闭合的 `exploratory_240_source_manifest.json`，至少包含每个 source 的 schedule index、formal class rank、source ID、三份 terminal 路径、状态、SHA-256、canonical digest、完成时间和 provenance。

当前已观察到的复算锚点如下，仅用于发现选择或实现漂移，执行者必须从原始 terminal 和数组独立重算，不能直接抄数：

- source count：240；120 个 class rank；
- 固定身份列表的 `canonical_digest`：`ad0c10e5057d767ef4329f81b22c5f1569afb7e820c7f88bb5199da7e03cb9b2`；
- 第 240 个 source 的完成时间：`1789511492710807249`；其 `schedule_index=300`；
- terminal states：`VALID_NNS=235`、`NA_VOID_MASK=5`、`VALID_OCCLUSION=448`、`NA_MATCHED_CONTROL_UNAVAILABLE=22`、`NA_UPSTREAM_NNS=10`；
- unexplained implementation/data-integrity failures：0。

若身份 digest、240 数量、class 数或 terminal state 计数不一致，必须 `BLOCKED_DATA_INTEGRITY`，不得通过另挑 240 张来追求 PASS。

## 5. 新的探索性判据

保留原 P1、D1 的 estimand、方向、阈值和 family-size-three L1 adjusted-CI 计算；P2 改为下列三项同时满足：

1. P2 点估计 `>= 0.07`；
2. 以零效应为 null，P2 family-size-three L1 adjusted one-sided CI lower bound `> 0`；
3. P2 ordinary within-class whole-image BCa 95% CI half-width `<= 0.02`。

其他探索性 PASS 条件：

- 240 个 selected source 均有三 unit 完整终态，共 720 个 resolved terminal；
- 没有 `FAILED_IMPLEMENTATION`、`BLOCKED_DATA_INTEGRITY`、`FAILED_NNS` 或 `FAILED_OCCLUSION`；
- P1、P2、D1 各自 valid fraction `>= 0.85` 且 NA fraction `<= 0.15`；
- P2 matching fraction `>= 0.85`；
- P1、D1 ordinary BCa 95% CI half-width仍要求 `<= 0.015`；P2 使用新的 `<= 0.02`；
- P1 的 adjusted lower bound `> 0.02`；
- D1 的 adjusted interval 完全位于 `[-0.02, 0.02]` 之外；
- 用异质 null `P1=0.02`、`P2=0`、`D1=+-0.02` 和原 family-size-three L1 boundary 计算的三项 closed-testing elementary decisions 均为 rejected。

原 P2 `0.10` 门槛和原 P2 half-width `0.015` 必须同时作为 `ORIGINAL_RULE_NOT_PASSED` 显示，不得删去。原设计 eligibility 只能记作历史参考，不能声称 240 张是预封的 eligible look。

## 6. 预期结果与 fail-closed 边界

当前独立只读复算得到以下预期值：

| Endpoint | estimate | ordinary BCa 95% CI | half-width | family-adjusted CI | post-hoc decision |
|---|---:|---:|---:|---:|---|
| P1 | 0.0626003110 | [0.0573913097, 0.0678940724] | 0.0052937614 | lower 0.0514420599 | PASS |
| P2 | 0.0749396239 | [0.0570841929, 0.0927266631] | 0.0178554310 | lower 0.0347448291 | PASS under new exploratory rules |
| D1 | 0.0363141999 | [0.0316151874, 0.0409722310] | 0.0046990126 | [0.0258156526, 0.0468127473] | PASS |

异质 null 下的预期 adjusted p-values 为 P1 `1.8768816516782683e-19`、P2 `1.613666525930498e-05`、D1 `4.136584313500504e-05`。这些数值只是验算锚点。独立实现若在合理浮点误差外不一致，必须停止并报告 `BLOCKED_DATA_INTEGRITY`，不得改算法、阈值或样本选择来恢复 PASS。

只有第 4--5 节全部条件实际满足时，才签发：

```text
S4_WORKFLOW_STATUS=PASS
SCIENTIFIC_STATUS=PASS_POST_HOC_EXPLORATORY
ORIGINAL_S4_R10_STOP_SUCCESS=NOT_CLAIMED
FORMAL_CONFIRMATORY_GATE_OPEN=false
EXPLORATORY_FIRST_LAYER_GATE_OPEN=true
```

## 7. 实现与产物

实现 CPU-only producer 和独立 validator，要求：

- `CUDA_VISIBLE_DEVICES=''`；禁止 Torch、timm、torchvision、PIL、JPEG/mask decode、模型/checkpoint 加载和 CUDA/GPU 初始化；
- 只读取 240 张 manifest 所绑定的 terminal、JSON plan 和 `.npy` 数值数组；
- 对每个输入重算 SHA-256、canonical digest、shape、dtype、finite、source identity 和 dependency binding；
- 20,000 次 BCa 必须沿用原 within-class whole-image resampling 方法，但新结果必须处于 S4-R11 root；
- producer 与 validator 独立重算 endpoint values、class aggregation、BCa、adjusted CI、adjusted p 和最终 truth table；validator 不得只信任 producer 的 PASS 字段；
- 为阈值、快照成员、终态、数组内容、P2 point gate、P2 lower-bound gate、P2 precision gate及 Y4 隔离设计 false-pass mutation tests；
- 使用闭合 schema，不得用开放 `payload: {type: object}`。

至少生成：

- `posthoc_exploratory_amendment.json`
- `s4_r10_interruption_audit.json`
- `exploratory_240_source_manifest.json`
- `input_integrity_report.json`
- `endpoint_numeric_result.json`
- `original_vs_posthoc_rule_comparison.json`
- `independent_validation_report.json`
- `false_pass_mutation_audit.json`
- `phased_access_and_deviation_audit.json`
- `whole_root_inventory.json`
- `terminal_receipt.json`

`terminal_receipt.json` 必须绑定全部文件 SHA-256/canonical digest、selection digest、原规则结果、新规则结果、outcome access disclosure、S4/Y4 process-isolation audit，以及下一授权。

## 8. S5/S6 交接

若且仅若 S4-R11 为 `PASS_POST_HOC_EXPLORATORY`，terminal 的下一授权写为：

```text
S5_CPU_ONLY_POST_HOC_EXPLORATORY_ANALYSIS_THEN_S6_EXPLORATORY_REPORTING
```

这表示 S5/S6 工作流可以继续，不表示原 confirmatory gate 被恢复。S5 必须继续 CPU-only、不得读 JPEG、不得加载模型或使用 GPU；P1/P2/D1 和后续 D2 都按 exploratory scope 重算并逐项报告。S6 可生成图表和结论，但标题、表格、claim ledger 与 captions 必须保留 `post-hoc exploratory` 标签，同时并列报告原规则下 P2 未通过。

本任务只写出完整可复制的 S5 执行提示词，不启动 S5；禁止启动 S6。

## 9. 验收、Git 与最终回复

完成前运行：

- 新增 S4-R11 focused tests；
- producer/validator 数值一致性测试；
- schema validation；
- false-pass mutation suite；
- S4-R10 不可变性检查；
- S4/Y4 进程隔离检查；
- `git diff --check` 和相关 non-GPU regression tests。

只提交本任务相关代码、schema、文档和小型 evidence JSON；不要提交或改写大数组。按仓库当前分支完成 commit 和 push，并验证 local HEAD、tracking ref、remote ref、branch status 与 Git LFS status。

最终回复必须给出：停止 S4 的实际 PID/结果（或 no-op）、Y4 未受影响证据、240 张 selection digest、三端点数值和每项 gate、原规则与新规则的并列表、terminal 路径与 digest、测试结果、commit/push 状态，以及下一条完整 S5 提示词。
