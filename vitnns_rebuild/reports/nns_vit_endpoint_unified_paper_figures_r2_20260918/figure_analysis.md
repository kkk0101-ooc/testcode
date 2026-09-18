# Endpoint Scanner 论文图逐图分析

[Gallery](gallery.html) · [Figure index](figure_index.md) · [Analysis](figure_analysis.md)

## Image #1: Endpoint Scanner study overview

从注册目标、结构分解、有限 activation replacement 到空间干预与论文判定的完整研究链。Identity、Correctness 与 Utility 分开报告。

![Figure 1](figures_png/figure_01_endpoint_scanner_study_overview.png)

## Image #2: Endpoint / CNN / Path method identity

NNS-CNN、ViT Endpoint 与 Path 方法的 seed、Jacobian 和方法身份。

![Figure 2](figures_png/figure_02_endpoint_cnn_path_method_identity.png)

## Image #3: Same target: Endpoint versus Path

同一解析目标的终点图与路径积分图；非线性 attention 下二者不必相等。

![Figure 3](figures_png/figure_03_same_target_endpoint_versus_path.png)

## Image #4: Original CNN endpoint gates

原始 CNN 门控例子说明 Endpoint 使用当前 ReLU/MaxPool 状态。

![Figure 4](figures_png/figure_04_original_cnn_endpoint_gates.png)

## Image #5: Attention Q/K/V differential paths

Attention 的 Q、K、V 微分路径及其闭合关系。

![Figure 5](figures_png/figure_05_attention_q_k_v_differential_paths.png)

## Image #6: Residual edge accounting

Residual 输入边记账，避免遗漏或重复旁路。

![Figure 6](figures_png/figure_06_residual_edge_accounting.png)

## Image #7: Unit, sum, and energy targets

单元、patch sum 与 energy 是三个不同标量目标。

![Figure 7](figures_png/figure_07_unit_sum_and_energy_targets.png)

## Image #8: Path, seed, and input coordinates

Endpoint、baseline-relative endpoint 与 IG 的 seed 和坐标原点。

![Figure 8](figures_png/figure_08_path_seed_and_input_coordinates.png)

## Image #9: LayerNorm and GELU Jacobians

LayerNorm 与 GELU 的真实 Jacobian 不能用冻结统计量或二值门代替。

![Figure 9](figures_png/figure_09_layernorm_and_gelu_jacobians.png)

## Image #10: Structured edge cuts

结构化切面把张量成分映回同一输入坐标并闭合到注册对象。

![Figure 10](figures_png/figure_10_structured_edge_cuts.png)

## Image #11: Registered target atlas design

ViT-B/16 每图 457 个目标，DeiT-S/16 每图 385 个目标，覆盖 12 个 block 与类别输出。

![Figure 11](figures_png/figure_11_registered_target_atlas_design.png)

## Image #12: Frozen first-eight endpoint gallery

冻结前 8 张 evaluation 图在两个模型上的 ground-truth 与 clean-top1 Endpoint 图。 读图时保留原始数值与显示缩放的区别；颜色图只用于空间形态。

![Figure 12](figures_png/figure_12_frozen_first_eight_endpoint_gallery.png)

## Image #13: One-image multi-target atlas

固定第一张 DeiT 图像的六类注册目标，不按显示效果挑选。 读图时保留原始数值与显示缩放的区别；颜色图只用于空间形态。

![Figure 13](figures_png/figure_13_one_image_multi_target_atlas.png)

## Image #14: Scan scale, latency, and resources

E2 共完成 336,800 个 finite target evaluation，并报告完整 atlas 的实际耗时。

![Figure 14](figures_png/figure_14_scan_scale_latency_and_resources.png)

## Image #15: Structural closure contracts

六种切面分别闭合 attention 分支或 residual total；负例必须失败。

![Figure 15](figures_png/figure_15_structural_closure_contracts.png)

## Image #16: Formal closure and negative controls

76,800 项正向闭合通过，28,800 项遗漏/重复负例被拒绝。

![Figure 16](figures_png/figure_16_formal_closure_and_negative_controls.png)

## Image #17: Real structural component maps

固定第一张图、block 5、同一类别目标的真实结构成分图。 读图时保留原始数值与显示缩放的区别；颜色图只用于空间形态。

![Figure 17](figures_png/figure_17_real_structural_component_maps.png)

## Image #18: Condition-scaled analytic JVP checks

6,144 个 condition-scaled analytic JVP 检查全部在冻结边界内。

![Figure 18](figures_png/figure_18_condition_scaled_analytic_jvp_checks.png)

## Image #19: Matched finite activation replacement

将同一个注册 component 替换为 reference activation，以有限 logit effect 检验 Endpoint mass。

![Figure 19](figures_png/figure_19_matched_finite_activation_replacement.png)

## Image #20: C4a family estimates and intervals

RQ3 的 12 个 C4a family：11 个满足全部条件，一个临界例外。

![Figure 20](figures_png/figure_20_c4a_family_estimates_and_intervals.png)

## Image #21: RQ3 threshold zoom and boundary exception

唯一例外 0.099565 距 0.10 仅 0.000435，方向为正且 Holm 显著；RQ3 整体支持。

![Figure 21](figures_png/figure_21_rq3_threshold_zoom_and_boundary_exception.png)

## Image #22: Endpoint and registered secondary predictors

Endpoint mass 与预注册次级 predictor 同时报告；p_ref 等指标是描述性比较，不进入 C4a 门。

![Figure 22](figures_png/figure_22_endpoint_and_registered_secondary_predictors.png)

## Image #23: Spatial intervention protocol

固定 4×4 区域、三个 head、两种 reference 和四个 mandatory baseline 的空间干预协议。

![Figure 23](figures_png/figure_23_spatial_intervention_protocol.png)

## Image #24: Real block-5 component and baseline maps

固定第一张 DeiT 图像的 total G×I、raw gradient 与三个 block-5 head component map。 读图时保留原始数值与显示缩放的区别；颜色图只用于空间形态。

![Figure 24](figures_png/figure_24_real_block_5_component_and_baseline_maps.png)

## Image #25: One-image spatial effect curves

固定单图的 isolated-head effect 曲线，展示各排序方法如何形成 AUC。

![Figure 25](figures_png/figure_25_one_image_spatial_effect_curves.png)

## Image #26: C4b mandatory-baseline contrasts

16 个 C4b baseline contrast 的中位差和 bootstrap 区间。

![Figure 26](figures_png/figure_26_c4b_mandatory_baseline_contrasts.png)

## Image #27: RQ4 family summary

四个 family 均未达到 +0.05；regional Taylor 在四个 family 中都更高。

![Figure 27](figures_png/figure_27_rq4_family_summary.png)

## Image #28: Identity–Correctness–Utility conclusion

最终证据分层：Identity 和 Correctness 成立，RQ3 整体支持，RQ4 不支持，Path 比较不发布。

![Figure 28](figures_png/figure_28_identity_correctness_utility_conclusion.png)
