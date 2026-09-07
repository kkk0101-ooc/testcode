# Stage F F1 图片分析

状态：`PASS`

本文档分析 F1 已渲染并提交的三个正式统计图模块，图像位于 `evidence/stage_f/f1_r3_recovery_v2_20260907/formal/renders/`。本文档只解释已生成图片和对应 machine-readable plot data，不产生新的 scientific decision，也不引入 D10 封存数据之外的新统计量。

## 数据与渲染范围

- 数值来源：`evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/paper_data/`。
- 元数据来源：同一 D10 `analysis` root 下的 `figure_recipes.json`、`analysis_report.json`、`status.json`、`payload_manifest.json` 和 `provenance.json`。
- `formal_r2/paper/` 只用于 byte/hash parity 检查，不作为第二套数值输入。
- 未读取 Stage E paper data、reserve JPEG、legacy 图片、参考论文数值、模型 checkpoint、Torch/timm、CUDA、GPU device handle、attribution map 或 fidelity rerun。
- Renderer：复用 F0 冻结的 `stage-f-renderer-v1`，包括 STIXGeneral 字体、600 DPI PNG、PDF/SVG 导出以及 single/double-column layout profiles。

## 已渲染模块

F1 生成 3 个 scientific modules、6 个 layout render instances、18 个导出图像文件。未生成最终组合 Figure 2；组合仍留给 F4。

| 模块 | 单栏文件 | 双栏文件 | Plot-data digest |
| --- | --- | --- | --- |
| F1-A cohort median interval | `f1_a_single_column.png`, `f1_a_single_column.pdf`, `f1_a_single_column.svg` | `f1_a_double_column.png`, `f1_a_double_column.pdf`, `f1_a_double_column.svg` | `3ec50287dda12e2cad0158c64880c230c9958564a54589a6c426e0047eeb2c2f` |
| F1-B Stage D delta_fid ECDF | `f1_b_single_column.png`, `f1_b_single_column.pdf`, `f1_b_single_column.svg` | `f1_b_double_column.png`, `f1_b_double_column.pdf`, `f1_b_double_column.svg` | `d15f51ee2d4eaf8cbf762be8d316c3ee094888d5a8db97b81dadeb0cf6f486f5` |
| F1-C median fidelity curves | `f1_c_single_column.png`, `f1_c_single_column.pdf`, `f1_c_single_column.svg` | `f1_c_double_column.png`, `f1_c_double_column.pdf`, `f1_c_double_column.svg` | `16d1d8493074d2425d8ae0540f65e899f75ca770f3a6ae6afb65e23bf8e83337` |

6 个 render report 均绑定同一个 F1 source input digest：`de6f5923804fafbf4250e22a5422ac98161b1a58f616545dc9c6663be4ae77f3`。

## F1-A: Cohort Median + Interval

F1-A 按要求顺序绘制 D10 封存的 `cohort_comparison` 行：先 `c10`，再 `stage_d`。

- C10 作为独立历史 15 图 holdout 展示：median `delta_fid=1.7534972022995547`，interval `[1.3317595184568727, 3.3813212483296478]`，decision `NO_GO_NOVELTY`，spatial status `FAIL_SPATIAL`，workflow status `BLOCKED`，Holm status `HOLM_BLOCKED_UNDERSPECIFIED`。
- Stage D 作为独立 500 图 cohort 展示：median `delta_fid=2.3081996221714247`，paired image-bootstrap 95% CI `[2.1110806126093067, 2.445235411755993]`，decision `GO_FIDELITY_SCOPE`，fidelity status `PASS_FIDELITY`。
- 图中和 caption data 均保持两个 cohort 分开，不池化为 `n=515`。

强制图注语句已原样保留：

```text
C10 的 15 图 holdout 与 Stage D 的独立 500 图 cohort 分开报告且不池化；C10 的历史 NO_GO_NOVELTY、FAIL_SPATIAL 和 BLOCKED 状态保持不变。
```

## F1-B: Stage D Delta FID ECDF

F1-B 绘制完整 Stage D `per_sample_delta` 分布。

- 使用行数：`500/500`。
- ECDF 分母：`500`。
- 最终累计概率：`1.0`。
- 保留 misclassified rows：`82`。
- 未过滤、删除、截尾、winsorize、平滑或按结果挑选任何行。
- 唯一变换是按 `delta_fid`、`selection_order`、`sample_id` 做确定性排序，然后显示为 `rank / 500`。
- 图中 `delta_fid` 范围为 `[0.4975565526556265, 9.197223597695693]`。

该模块只是对 D10 封存逐图 delta 的描述性显示，不创建新 endpoint。

## F1-C: Median Fidelity Curves

F1-C 绘制 D10 封存的 `curve_summary` 值。

- Panel/facet：`insertion` 与 `deletion` 分开。
- Methods：`nns` 与 `matched`。
- 共 4 条曲线，每条曲线 21 个 fraction 点，范围 `0.0` 到 `1.0`。
- 方法样式使用 F0 冻结身份：NNS `#0072B2`、实线、圆点；matched `#D55E00`、虚线、方点。
- Band 使用 D10 封存的 `pointwise_ci_lower` 和 `pointwise_ci_upper`。
- Band 解释为 pointwise、非 simultaneous，且不进入 primary decision。
- 未重算或改变 bootstrap、CI、fraction grid、tie-repeat reduction 或曲线统计量。

## Validation Summary

F1 terminal validator 通过，结果如下：

- `scientific_modules_rendered=3`
- `layout_render_instances=6`
- `exported_figure_files=18`
- `final_combined_figure_2_rendered=false`
- `model_torch_timm_gpu_cuda_jpeg_access=false`
- `bootstrap_ci_new_endpoint_recomputed=false`
- `c10_stage_d_pooled=false`
- `d10_f0_e4_decisions_modified=false`
- `next_authorization=F2_ONLY`

False-pass audit 拒绝全部 18 类要求的攻击。正式测试命令在 `CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 下通过，结果为 `41 passed, 0 failed`。
