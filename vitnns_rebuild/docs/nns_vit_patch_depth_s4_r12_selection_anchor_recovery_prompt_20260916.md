# S4-R12 selection-anchor recovery execution prompt

在 `/data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild` 执行 `NNS_VIT_PATCH_DEPTH_SEMANTICS` 的 S4-R12 控制面恢复。唯一任务是纠正 S4-R11 提示词中两个已经查明的元数据锚点定义错误，在不重跑 S4、不改变 240 张成员、不改变任何 endpoint 数值或 post-hoc 门槛、不修改 S4-R10/S4-R11 历史 root 的前提下，独立验证并封存新的 S4-R12 `PASS_POST_HOC_EXPLORATORY` terminal；若验证通过，只生成并授权新的 S5 CPU-only post-hoc exploratory 提示词。本任务不得执行 S5 或 S6。

## 1. 当前权威事实与恢复性质

启动 commit 必须是或包含：

```text
9cec5ddca0cdd8730571ac3e14ddf7fd08c06331
```

先验证 local HEAD、tracking ref 和 live remote branch 均包含该提交。S4-R11 的 `BLOCKED_DATA_INTEGRITY` 是有效且不可变的历史终态，不得覆盖或改写。R12 是 fresh recovery root。

R11 的阻断来自提示词中的两个控制面锚点定义错误，而不是样本、terminal、数组或科学数值损坏：

1. 提示词把 `selected distinct class ranks` 错写成 120；正确口径是：240 个 selected source 覆盖 122 个 distinct formal class ranks，排除 typed NA 后 P1/D1 有效 class 数为 120、P2 有效 class 数为 119。
2. 提示词把三字段 membership projection 的 digest `ad0c10...` 错当作完整 manifest records digest；实际两者是不同、同时有效的摘要。

这个勘误发生在已经读取 outcome 之后，因此 R12 仍是明确的 `post-hoc exploratory`，绝不能声称 preregistered、outcome-blind、confirmatory 或 strong-FWER-preserved。它只纠正控制面描述，不恢复原 S4-R10 的 `STOP_SUCCESS`。

## 2. 不可变输入绑定

完整读取并校验以下 R11 文件：

- `evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/terminal_receipt.json`
  - SHA-256 `eaabb5cc0912ea7b6bc632300e1214bcedbc16febe7c2fc1b11e2d71e6efa6a4`
  - canonical digest `651a8ce172f34b6894f098fe2ab2e3910d11d56353bffd19d312c0140f92ccac`
  - status 必须保持 `BLOCKED_DATA_INTEGRITY`。
- `evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/exploratory_240_source_manifest.json`
  - SHA-256 `81e3414cd04681bfc626b4cca836caa3cc2052b916aba18454b43eca08fcd1f5`
  - canonical digest `94b44455674f7fd68a85bfefc97ca1677d09beefdb7fa056f1998b1f728974f0`。
- `evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/endpoint_numeric_result.json`
  - SHA-256 `1357713798be111b6f971552b3d9fff32d64986d4f723ad0696b6c8174df49f4`
  - canonical digest `323d0758404226efdcb8cffab621c185818b30bc6bee898fdfeb31e8f0c68334`。
- `evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/independent_validation_report.json`
  - SHA-256 `85e52a1de3b091351068b7adcbf4e0acc71b421ddd248f68d14f887bb614e72f`
  - canonical digest `baf90e27eef2ed8c3495a3fc34cee5d565fa84f35e467bb7620aa237ad60821b`。
- `evidence/nns_vit_patch_depth/s4_r11_posthoc_exploratory_20260916/s4_r10_immutable_root_inventory.json`
  - SHA-256 `c5d0183f09455f6d544e4fd20b218e9fae06d4aa9466e9d907aa9645f93a1119`
  - canonical digest `96cbc818bac5b4e4d8df62b90991071211407b8b7a57e3d594483392bd8d3b3d`
  - file count `8684`，records digest `eb1557b9747d4c0ac31dc3930c3fbe3c9374a564ae8c6d7947a90a00be726d41`。

同时读取 R11 implementation、producer、validator、schemas、tests 和 S5 草案，理解失败原因；但禁止修改它们。S4-R10、S4-R11 的 evidence/artifact/code history全部保留。

新 root 必须为：

```text
evidence/nns_vit_patch_depth/s4_r12_selection_anchor_recovery_20260916/
```

如需新代码和 schema，使用 `s4_r12` 独立命名；不得覆盖 R11 文件。无需创建新的大数组目录。

## 3. 修正后的 240-source 身份合同

R12 不重新选择样本。唯一允许的 selected records 是 R11 manifest `payload.records` 中既有、既定顺序的 240 条记录。必须验证其仍符合原选择规则：在 S4-R10 L1 中对三份 resolved terminal 的 `payload.completed_unix_ns` 取最大值，再按 `(source_completion_time_max_unix_ns, schedule_index)` 升序取前 240。

以下锚点必须全部成立：

- selected source count：240；
- selected terminal count：720；
- selected distinct formal class rank count：122；
- P1/D1 valid class count：120；
- P2 valid class count：119；
- cutoff completion time：`1789511492710807249`；
- cutoff schedule index：`300`；
- terminal states：`VALID_NNS=235`、`NA_VOID_MASK=5`、`VALID_OCCLUSION=448`、`NA_MATCHED_CONTROL_UNAVAILABLE=22`、`NA_UPSTREAM_NNS=10`；
- unexplained implementation/data-integrity failure count：0。

必须计算并区分以下两个 digest，不得再共用一个含糊的 `selection_digest` 字段：

### 3.1 完整 records digest

对象是 R11 manifest 中顺序不变的整个 `payload.records` list，包括 source fields 和每份 terminal 的 path、bytes、SHA-256、canonical digest、status、完成时间及 provenance。序列化规则：

```python
json.dumps(
    records,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

对上述 bytes 做 SHA-256，预期：

```text
full_manifest_records_digest = 3bb68cb6bb627f4a84f1a6d7bd54e26d430541b4ae4cd565a6fe3eb2d46863e9
```

### 3.2 三字段 membership projection digest

从同一 records 顺序生成：

```python
membership_projection = [
    {
        "schedule_index": int(record["schedule_index"]),
        "formal_class_rank": int(record["formal_class_rank"]),
        "source_image_id": record["source_id"],
    }
    for record in records
]
```

对该 list 使用与 3.1 完全相同的 canonical JSON 序列化和 SHA-256，预期：

```text
membership_projection_digest = ad0c10e5057d767ef4329f81b22c5f1569afb7e820c7f88bb5199da7e03cb9b2
```

注意 projection 的字段名必须是 `source_image_id`，其值来自 manifest record 的 `source_id`。不得添加 wrapper，不得改字段名，不得排序 records，不得省略或增加字段。

如果上述任一 digest、成员顺序、cutoff、状态计数或 class 口径不匹配，则 R12 才是真正的 `BLOCKED_DATA_INTEGRITY`。禁止为了匹配锚点另挑样本。

## 4. 科学数值与 post-hoc PASS 合同保持不变

R12 不修改 R11 的 estimator、聚合、bootstrap、边界或 owner 指定的探索性阈值。必须从 R11 manifest 绑定的 terminal、P2 plan 和原始 `.npy` 数组独立复算，不得仅复制 R11 的 PASS 字段。

P2 三项规则仍为：

1. 点估计 `>= 0.07`；
2. 以零效应为 null，family-size-three L1 adjusted one-sided CI lower bound `> 0`；
3. ordinary within-class whole-image BCa 95% CI half-width `<= 0.02`。

其他规则保持：

- P1 adjusted lower bound `> 0.02`，ordinary BCa half-width `<= 0.015`；
- D1 adjusted interval完全位于 `[-0.02, 0.02]` 之外，ordinary BCa half-width `<= 0.015`；
- 三端点 valid fraction `>= 0.85`、NA fraction `<= 0.15`；
- P2 matching fraction `>= 0.85`；
- 720 个 terminal 完整，无 unexplained failure；
- 异质 null `P1=0.02`、`P2=0`、`D1=+-0.02` 下，沿用原 family-size-three L1 boundary 的 closed-testing elementary decisions 全部 rejected。

预期复算值必须在 R11 已用的浮点容差内一致：

| Endpoint | estimate | ordinary BCa 95% CI | half-width | family-adjusted CI | adjusted p |
|---|---:|---:|---:|---:|---:|
| P1 | 0.0626003110 | [0.0573913097, 0.0678940724] | 0.0052937614 | lower 0.0514420599 | 1.8768816516782683e-19 |
| P2 | 0.0749396239 | [0.0570841929, 0.0927266631] | 0.0178554310 | lower 0.0347448291 | 1.613666525930498e-05 |
| D1 | 0.0363141999 | [0.0316151874, 0.0409722310] | 0.0046990126 | [0.0258156526, 0.0468127473] | 4.136584313500504e-05 |

原 P2 规则仍必须并列报告：点估计 `>0.10`、adjusted lower `>0.10`、half-width `<=0.015` 均未通过，状态为 `ORIGINAL_RULE_NOT_PASSED`。不得隐去或改写。

## 5. R12 实现、验证和状态

全程 `CUDA_VISIBLE_DEVICES=''`。禁止 Torch、timm、torchvision、PIL、JPEG/mask decode、checkpoint/model load、CUDA/GPU 初始化和任何新增 intervention。禁止发送进程信号；S4-R10 已在 R11 中安全停止，R12 只验证 R11 interruption audit 的身份、hash 和 `Y4_signal_count=0`。

实现 fresh R12 producer、independent validator、closed schemas 和 tests。至少生成：

- `selection_anchor_correction_amendment.json`；
- `r11_blocked_terminal_binding.json`；
- `corrected_selection_identity_report.json`；
- `input_integrity_report.json`；
- `endpoint_numeric_recomputation.json`；
- `original_vs_posthoc_rule_comparison.json`；
- `cpu_access_and_process_isolation_audit.json`；
- `independent_validation_report.json`；
- `false_pass_mutation_audit.json`；
- `whole_root_inventory.json`；
- `terminal_receipt.json`。

`selection_anchor_correction_amendment.json` 必须明确记录：

- R11 的 BLOCKED terminal 没有被改写；
- `selected_class_rank_count: 120 -> 122` 是元数据口径纠正；
- 120 是 P1/D1 valid class count，119 是 P2 valid class count；
- `ad0c10...` 从未是 full records digest，而是三字段 membership projection digest；
- full records digest 为 `3bb68...`；
- correction 不改变任何 selected source、terminal、array、endpoint estimate、CI、p-value、threshold 或 decision rule；
- correction 是 outcome-accessed post-hoc recovery，不恢复 confirmatory 性质。

independent validator 必须自行重算两个 digest、三个 class counts、240-source 选择、720 terminal identity/hash、所有 endpoint 数值和全部 gate。validator 不得信任 producer 的 status。false-pass suite 至少攻击：records 顺序、成员增删、terminal SHA、`source_id` 到 `source_image_id` 投影、122/120/119 三种 class 口径、两个 digest 对调、cutoff、P2 三门、R11 terminal binding、Y4 signal count。每项攻击必须被拒绝。

仅当身份、完整性和全部 post-hoc 数值门均通过时，签发：

```text
S4_WORKFLOW_STATUS=PASS
SCIENTIFIC_STATUS=PASS_POST_HOC_EXPLORATORY
R11_TERMINAL_STATUS_PRESERVED=BLOCKED_DATA_INTEGRITY
R11_BLOCK_CLASSIFICATION=CONTROL_PLANE_ANCHOR_DEFINITION_ERROR
ORIGINAL_S4_R10_STOP_SUCCESS=NOT_CLAIMED
FORMAL_CONFIRMATORY_GATE_OPEN=false
EXPLORATORY_FIRST_LAYER_GATE_OPEN=true
next_authorization=S5_CPU_ONLY_POST_HOC_EXPLORATORY_ANALYSIS_THEN_S6_EXPLORATORY_REPORTING
```

若任何真实成员、hash、terminal、数组、依赖或数值不一致，签 `BLOCKED_DATA_INTEGRITY`；若身份完整但 post-hoc 数值门不通过，签 `FAIL_POST_HOC_EXPLORATORY`。不得把真正的数据错误降格为控制面勘误。

## 6. S5 提示词交接

不要修改 R11 留下的旧 S5 草案，它是 BLOCKED 历史的一部分。另建：

```text
docs/nns_vit_patch_depth_s5_r1_posthoc_exploratory_execution_prompt_20260916.md
```

新提示词必须只信任 R12 terminal，并在启动时验证 R12 terminal 的 SHA-256、canonical digest、`SCIENTIFIC_STATUS=PASS_POST_HOC_EXPLORATORY` 和精确 `next_authorization`。其余 CPU-only、原始数组独立复算、D2 exploratory、typed NA、original-vs-posthoc comparison、closed schema、禁止 S6 等边界继承 R11 的 S5 草案。R12 未 PASS 时，该提示词必须 fail closed 为 `S5_NOT_AUTHORIZED`。本轮只生成提示词，不执行 S5。

## 7. 测试、提交与最终报告

建议先提交并 push R12 implementation/schema/tests/amendment，再在远端可达的实现 commit 上生成 fresh R12 evidence，最后提交并 push evidence 与新的 S5 提示词。不得提交大数组副本。

至少运行并报告：

- R12 focused tests；
- producer 与独立 validator 一致性；
- 所有 R12 JSON 的 closed-schema validation；
- false-pass mutation suite；
- R11/R10 immutable binding 验证；
- CPU-only/import/access audit；
- 相关 non-GPU regression；
- `git diff --check`；
- `git status --branch --porcelain=v2`；
- `git lfs status` 与 `git lfs fsck`；
- local HEAD、tracking ref、`git ls-remote` live remote SHA 一致。

忽略且不得改动仓库外层已有的 `.root_cleanup_quarantine/`。

最终回复必须报告：R11 BLOCKED 历史保留情况、两项锚点错误的更正、两个 digest 的精确定义和值、122/120/119 三种 class 口径、三端点数值与每项 gate、原 P2 规则未通过、R12 terminal 路径/SHA/canonical digest、Y4 未受信号影响、测试、commit/push 状态，以及新的完整 S5 提示词路径。
