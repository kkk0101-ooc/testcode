# Stage F 跨层/头归因展示补充协议

> 建立日期：2026-09-07。
> 时点：Stage E E4 和 Stage F F2 已完成，F3 尚未开始。
> 性质：post-E4、display-only 的 F3/F4 补充合同；无科学决策权限。
> 上位文档：[阶段 F 规划](阶段F规划.md)。
> 既有选例规则：[阶段 F 展示选择协议 v1](stage_f_display_protocol_v1.md)。

## 1. 目的与术语

本补充协议增加参考文章 Fig. 11 / Fig. 13 风格的跨层/头图像矩阵，用于展示同一样本在冻结坐标 `(0,9)`、`(4,2)`、`(6,10)`、`(11,2)` 上的空间归因模式。

F3 启动配置必须记录本文件的 file SHA-256 和所在 Git commit；F3 producer、validator 和 completion 必须绑定同一 identity。对本文件的后续修改只能形成新版本，不能静默改变已启动的 F3。

这些图必须称为：

> fixed-clean-attention head-direction Value-path attribution progression

它们不是 raw self-attention matrix、完整 attention attribution、完整 Q/K/V attribution、class-logit attribution，也不能表述为“网络逐层学到了什么”的直接证明。不同坐标对应不同的冻结 head-direction target；跨列只允许描述空间分布变化，不允许把颜色强弱直接解释为层间绝对 attribution magnitude 或因果演化。

## 2. 冻结输入

### 2.1 Primary 坐标

`(0,9)` 只从 D9 验证、D10 使用的 Stage D sealed L1 evidence 读取：

```text
evidence/stage_d/d8_recovery_v2_20260904/formal_r2/
```

预期为 500 图 × 2 方法的 1,000 个 `L1/metric_ready_map.npy`。不得重新计算 primary map。

### 2.2 Secondary 坐标

`(4,2)`、`(6,10)`、`(11,2)` 必须从 E3 最终验证的 typed-terminal union 解析：

```text
evidence/stage_e/e3_recovery_v2_20260906/formal/
evidence/stage_e/e3_r1_recovery_v2_20260906/formal/
evidence/stage_e/e3_r2_recovery_v2_20260906/formal/
evidence/stage_e/e3_r4_recovery_v2_20260907/formal/
```

四个 root 当前分别保留 516、12、636、1,834 个有效 `map.npy`，合计 2,998 个有效 secondary maps。预期总数为 500 图 × 3 坐标 × 2 方法 = 3,000；两个缺失 map 都是 `(11,2)` 的 NNS 单元：selection order `117` 为 `ig_conservation_failed_after_e3_r1_schedule`，selection order `352` 为 `ig_conservation_failed_typed_terminal`。这两个 typed failures 必须从 terminal records 解析并显示，不能因缺少 `map.npy` 而从矩阵中消失；同样本同坐标的 matched 单元若验证有效则仍正常显示。

F3 必须使用 E3-R4 已验证的 union tables 解析唯一来源：

- `evidence/stage_e/e3_r4_recovery_v2_20260907/formal/paper_data/layer_head_method_rows.json`；
- `evidence/stage_e/e3_r4_recovery_v2_20260907/formal/paper_data/new_compute_terminal_rows.json`；
- `evidence/stage_e/e3_r4_recovery_v2_20260907/formal/paper_data/layer_head_paired_delta.json`；
- E4 sealed copy `evidence/stage_e/e4_recovery_v2_20260907/formal/paper_data/layer_head_descriptive.json`。

每个 resolved map 必须核对 sample ID、selection order、block/head、method、validity status、attempt ID、origin root、file SHA-256、`signed_map_sha256` 和 pair binding。若多个 root 对同一逻辑单元提供冲突的 terminal payload，F3 必须 `BLOCKED`，不得按文件时间或视觉效果选择。

### 2.3 方法和原图

每个坐标只允许以下两种已冻结方法：

- NNS：`conditional_fixed_clean_attention_head_direction_value_path_ig`；
- matched：`matched_fixed_clean_attention_patch_token_contribution`。

原图只从 `assets/data/ILSVRC2012_val/` 读取，并必须与 registry/E3 record 中的 image SHA-256 一致。禁止加载模型、执行 forward/backward、重新生成 map 或 fidelity curve；GPU 预算为 0。

## 3. 样本选择冻结

跨层图不得依据 E3 map 外观、层间变化大小或人工“好看程度”选择样本。

### 3.1 主文 Figure 6

Figure 6 的唯一主样本固定为既有展示协议产生的 `median_effect` slot。该 slot 只按 D10 primary `delta_fid` 与完整 500 图 median 的距离确定，沿用既有 tie-break，不查看 E3 maps。

如果该样本某个 secondary coordinate 为 `FAILED` 或 `NA`，对应 cell 必须显示状态、selection order 和 failure code；不得换成第二接近 median 的样本。若 `median_effect` 规则本身无法解析，Figure 6 为 `BLOCKED`。

### 3.2 Supplement S4

S4 按以下固定顺序生成跨层矩阵：

1. D1 的 5 个 fixed audit IDs，保持 registry 顺序；
2. `best_effect`；
3. `median_effect`；
4. `baseline_strong_nns_stronger`；
5. `weak_or_failure`；
6. `misclassified_or_difficult`。
7. typed failure sample at selection order `117`；
8. typed failure sample at selection order `352`。

同一样本命中多个规则或 failure slot 时，底层 manifest 必须保留所有 slot-to-image 行；渲染文件允许按首次出现顺序去重，但不得静默替换重复项。任何失败/NA cell 原位保留；两个 failure rows 中 `(11,2)` NNS 显示明确的 failure panel，同一坐标的 matched 及其他有效坐标仍照常显示。

## 4. Figure 6 与 S4 版式

Figure 6 使用 `full_width_qualitative` profile，固定结构为：

```text
                  Primary             Secondary frozen coordinates
Original          (0,9)               (4,2)       (6,10)       (11,2)
NNS               map                 map          map          map
Matched           map                 map          map          map
```

允许增加一行 `NNS - matched` difference maps，但如果增加，四个坐标必须全部生成，不能只展示有利坐标。原图只需出现一次，不得针对不同坐标使用不同裁剪。

S4 使用同一列顺序和方法顺序，为第 3.2 节的完整固定集合逐样本生成；允许分页或拆成多个 supplementary figure，但排序和内容不能改变。若 failure sample 已被较早 slot 命中，渲染可去重，但 failure slot 和失败 panel 仍必须在 manifest 与图注中显式标识。

## 5. 色限与可比性

1. 同一样本、同一坐标的 NNS 与 matched 使用两张 raw signed maps 的联合绝对最大值形成共享对称色限；禁止分别自动缩放。
2. difference map 使用自己的对称色限，并明确标为 `NNS - matched`。
3. 不同坐标对应不同 target，默认不共享色限；每列必须提供 colorbar 或机器可读色限。
4. 因此跨列颜色只比较空间 pattern，不比较绝对 magnitude。图注必须明确这一点。
5. 如另行生成全坐标共享 raw 色限的 audit view，必须标为 supplementary，并仍不得宣称 target 间可直接比较。
6. 不允许 percentile clipping 隐藏极值；若为显示必须启用 clipping，必须同时交付 unclipped audit view 并记录阈值和被裁元素数。
7. resize、smoothing、alpha blending 和 colormap 只属于 display-only；原始 map hash 和变换后 array hash 均需记录。

## 6. 强制图注

Figure 6 和 S4 必须包含等价声明：

> Panels show fixed-clean-attention head-direction Value-path attribution, not raw self-attention. The four layer/head coordinates were frozen before Stage E3 outcome access. The main example was selected by the pre-existing D10-based `median_effect` display rule, not by inspecting cross-layer maps. NNS and matched maps share a scale within each coordinate; colors across coordinates describe spatial pattern only and are not an absolute-magnitude comparison. Failed or unavailable cells are retained explicitly.

中文内部报告必须包含等价声明：

> 图中展示的是 fixed-clean-attention head-direction Value-path attribution，不是原始 self-attention。四个层/头坐标在 E3 outcome access 前已冻结；主样本来自既有 D10 `median_effect` 展示规则，而非查看跨层 map 后挑选。每个坐标内 NNS 与 matched 共享色限；不同坐标间只比较空间模式，不比较绝对强度。失败或不可用单元原位保留。

## 7. 输出和验证

F3 新增的机器可读输出至少包括：

- `cross_layer_source_union.json`：四个 source roots、inventory、逻辑单元、origin 和冲突检查；
- `cross_layer_selection_manifest.json`：Figure 6/S4 的固定 slots、sample IDs、重复映射和选择依据；
- `cross_layer_panel_manifest.json`：每个 sample × coordinate × method 的 source/result/map/status/hash；
- `cross_layer_scale_manifest.json`：每列色限、difference 色限、clipping 和 transform；
- Figure 6 与 S4 的 PNG/PDF/SVG、render reports 和 file hashes。

validator 必须独立验证：

1. D8 primary 1,000 maps 的身份和 D10 binding；
2. E3 union 3,000 typed terminals、2,998 valid maps 和 2 个原位 failure；
3. 只有四个冻结坐标和两种冻结方法；
4. Figure 6 精确使用 `median_effect`；
5. S4 完整覆盖 5 个 audit IDs、5 个规则 slots 和 orders `117`、`352` 两个 failure slots；
6. 同坐标 NNS/matched 共享色限，跨坐标无幅度主张；
7. JPEG/map 只读，模型/GPU/新科学计算为零；
8. PNG/PDF/SVG、caption 和机器报告绑定同一 panel manifest。

## 8. False-pass 条件

以下任一情况必须使 F3 `BLOCKED`：

- 只读取 E3-R4 而遗漏 E3/E3-R1/E3-R2 的有效历史 union；
- 缺失或覆盖 orders `117`、`352` 的 typed failure；
- 根据跨层 map 外观更换 Figure 6/S4 样本；
- 把这些图称为 raw attention map 或完整 attention/QKV/class-logit attribution；
- 使用不同色限夸大同坐标方法差异；
- 直接比较不同坐标的颜色幅度并形成层间强弱结论；
- 对缺失 map 补零、插值、复制相邻坐标或替换样本；
- 为生成图片重新加载模型或运行 attribution；
- 图像面板与 manifest、caption 或 source hash 不一致。

本补充协议只扩展 F3/F4 的展示交付物，不修改 D10、E4、F0、F1、F2 的任何结果、completion 或 archive ref。
