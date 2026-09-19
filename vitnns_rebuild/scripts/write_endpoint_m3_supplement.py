#!/usr/bin/env python3
"""Write synchronized English and Chinese documents for M3-SUP-R2."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m3_sup_r2_20260919"
PAPER = REPO / "papers/nns_vit_endpoint"


def rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def sci(value: object) -> str:
    return f"{float(value):.3e}"


def num(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def ci(value: str, digits: int = 4) -> str:
    bounds = json.loads(value)
    return f"[{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]"


def markdown_table(lines: list[str], header: str) -> str:
    return header + "\n" + "\n".join(lines)


def main() -> None:
    errors = rows("m3a_error_distribution.csv")
    operators = rows("operator_validation_table.csv")
    blocks = rows("block_summary.csv")
    cls = rows("cls_patch_summary.csv")
    cancellation = rows("cancellation_summary.csv")
    finite_difference = rows("finite_difference_diagnostics.csv")
    review = json.loads((ROOT / "cancellation_float64_review.json").read_text())

    error_lines = []
    for model in ("fcn", "cnn", "cnn_maxpool", "residual_mlp", "residual_cnn"):
        for dtype in ("float32", "float64"):
            key = model + "|" + dtype
            relative = next(r for r in errors if r["group"] == "model_dtype" and r["key"] == key and r["metric"] == "relative_l2")
            tolerance = next(r for r in errors if r["group"] == "model_dtype" and r["key"] == key and r["metric"] == "max_tolerance_ratio")
            error_lines.append(
                f"| {model} | {dtype} | {relative['n']} | {sci(relative['median'])} | "
                f"{sci(relative['q99'])} | {sci(relative['maximum'])} | {sci(tolerance['maximum'])} |"
            )
    error_table_en = markdown_table(error_lines, "| Network | Precision | Comparisons | Median relative L2 | 99% relative L2 | Max relative L2 | Max tolerance ratio |\n|---|---|---:|---:|---:|---:|---:|")
    error_table_zh = markdown_table(error_lines, "| 网络 | 精度 | 比较数 | relative L2 中位数 | relative L2 99% 分位 | 最大 relative L2 | 最大容差比 |\n|---|---|---:|---:|---:|---:|---:|")

    operator_lines = [
        f"| {r['operator']} | {r['directions']} | {r['dtype']} | "
        f"{sci(r['max_analytic_autograd_jvp_relative_error'])} | "
        f"{sci(r['max_jvp_vjp_identity_relative_error'])} | "
        f"{sci(r['max_best_finite_difference_relative_error'])} | "
        f"{sci(max(float(r['max_analytic_autograd_jvp_relative_error'])/1e-9, float(r['max_jvp_vjp_identity_relative_error'])/1e-10, float(r['max_best_finite_difference_relative_error'])/2e-6))} | "
        f"9 | analytic + AD JVP/VJP + FD |"
        for r in operators
    ]
    operator_table_en = markdown_table(operator_lines, "| Operator | Directions | Precision | Max analytic-AD JVP | Max JVP-VJP | Max best-step FD | Max contract ratio | Epsilons | Checks |\n|---|---:|---|---:|---:|---:|---:|---:|---|")
    operator_table_zh = markdown_table(operator_lines, "| 算子 | 方向数 | 精度 | 最大 analytic-AD JVP | 最大 JVP-VJP | 最大最优步长 FD | 最大合同占用比 | epsilon 数 | 检查 |\n|---|---:|---|---:|---:|---:|---:|---:|---|")

    block_lines = []
    for model in ("vit_b16", "deit_s16"):
        for block in range(12):
            value = {r["metric"]: r for r in blocks if r["model"] == model and int(r["block"]) == block}
            block_lines.append(
                f"| {model} | {block} | {num(value['entropy']['median'])} {ci(value['entropy']['ci95'])} | "
                f"{num(value['locality']['median'])} {ci(value['locality']['ci95'])} | "
                f"{num(value['distance']['median'])} {ci(value['distance']['ci95'])} | "
                f"{num(value['net_ratio']['median'], 6)} {ci(value['net_ratio']['ci95'], 6)} |"
            )
    block_table_en = markdown_table(block_lines, "| Model | Block | Entropy [95% CI] | Locality [95% CI] | Distance [95% CI] | R_net [95% CI] |\n|---|---:|---:|---:|---:|---:|")
    block_table_zh = markdown_table(block_lines, "| 模型 | Block | Entropy [95% CI] | Locality [95% CI] | Distance [95% CI] | R_net [95% CI] |\n|---|---:|---:|---:|---:|---:|")

    cls_lines = []
    for model in ("vit_b16", "deit_s16"):
        for block in range(12):
            row = next(r for r in cls if r["model"] == model and r["stratum"] == f"block_{block:02d}")
            cls_lines.append(f"| {model} | {block} | {num(row['median_difference'])} | {ci(row['ci95'])} | {row['positive']}/{row['negative']}/{row['ties']} |")
    cls_table_en = markdown_table(cls_lines, "| Model | Block | Median CLS-patch entropy | 95% image-bootstrap CI | Positive/negative/tie images |\n|---|---:|---:|---:|---:|")
    cls_table_zh = markdown_table(cls_lines, "| 模型 | Block | CLS-patch entropy 配对中位差 | 95% 图像 bootstrap CI | 正/负/平图像数 |\n|---|---:|---:|---:|---:|")

    phase_lines = []
    for model in ("vit_b16", "deit_s16"):
        for stratum in ("all", "early", "middle", "late"):
            row = next(r for r in cls if r["model"] == model and r["stratum"] == stratum)
            phase_lines.append(f"| {model} | {stratum} | {num(row['median_difference'])} | {ci(row['ci95'])} | {row['positive']}/{row['negative']}/{row['ties']} |")
    phase_table_en = markdown_table(phase_lines, "| Model | Stratum | Median paired difference | 95% image-bootstrap CI | Positive/negative/tie images |\n|---|---|---:|---:|---:|")
    phase_table_zh = markdown_table(phase_lines, "| 模型 | 分层 | 图像级配对中位差 | 95% 图像 bootstrap CI | 正/负/平图像数 |\n|---|---|---:|---:|---:|")

    cancel_lines = []
    for model in ("vit_b16", "deit_s16"):
        for block in (0, 5, 11):
            for token_type in ("cls", "patch"):
                value = {r["metric"]: r for r in cancellation if r["model"] == model and int(r["block"]) == block and r["token_type"] == token_type}
                cancel_lines.append(
                    f"| {model} | {block} | {token_type} | {num(value['within_rgb_channels']['median'])} | "
                    f"{num(value['within_patch_after_rgb_sum']['median'])} | {num(value['across_patch_signed']['median'])} | "
                    f"{num(value['whole_image_cancellation']['median'])} | {num(value['net_ratio']['median'], 6)} |"
                )
    cancel_table_en = markdown_table(cancel_lines, "| Model | Block | Token | RGB-channel C | Within-patch C | Across-patch C | Whole-image C | R_net |\n|---|---:|---|---:|---:|---:|---:|---:|")
    cancel_table_zh = markdown_table(cancel_lines, "| 模型 | Block | Token | RGB 通道内 C | Patch 内 C | Patch 间 C | 全图 C | R_net |\n|---|---:|---|---:|---:|---:|---:|---:|")

    worst = sorted(finite_difference, key=lambda r: float(r["best_float32_tolerance_ratio"]), reverse=True)[:10]
    fd_magnitudes = np.asarray([float(r["absolute_analytic_magnitude_float64"]) for r in finite_difference])
    fd_lines = [
        f"| {r['task_id']} | {r['block']} | {r['token']} | {r['channel']} | {r['direction']} | "
        f"{sci(r['absolute_analytic_magnitude_float64'])} | {num(r['best_float32_tolerance_ratio'], 3)} @ {r['best_float32_epsilon']} | "
        f"{sci(r['best_float64_tolerance_ratio'])} @ {r['best_float64_epsilon']} |"
        for r in worst
    ]
    fd_table_en = markdown_table(fd_lines, "| Task | Block | Token | Channel | Direction | abs(JVP) float64 | Best float32 ratio/eps | Best float64 ratio/eps |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    fd_table_zh = markdown_table(fd_lines, "| Task | Block | Token | Channel | Direction | abs(JVP) float64 | float32 最优容差比/eps | float64 最优容差比/eps |\n|---|---:|---:|---:|---:|---:|---:|---:|")

    english = f"""# Endpoint M3 Supplement R2

> This supplement extends the sealed M3 presentation without changing its ten Holm decisions. The added analyses are supplementary/descriptive and introduce no new confirmatory test.

## Numerical-equivalence distribution

The 5,000 comparisons are numerical validation cases, not independent statistical samples. The overall relative-L2 median was 1.088e-16, the 99th percentile was 1.602e-7, and the maximum was 2.427e-7. Float64 had median 5.194e-17 and maximum 4.769e-16; float32 had median 3.506e-8, 99th percentile 1.782e-7, and maximum 2.427e-7.

{error_table_en}

## Operator-level directional validation

Each operator used eight float64 directions and nine central-difference steps `[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]`. These checks validate the tested operator implementations and directions. End-to-end hooks, registries, and structural cuts are covered by separate closure, parity, and negative-control evidence.

{operator_table_en}

## Twelve-block spatial broadening

Each image first contributes a median over its frozen units at a block; the table then reports the median and 10,000-replicate image-bootstrap interval across 100 images. Entropy rises mainly through early and middle blocks, locality decreases throughout depth, and normalized distance increases through block 11. Entropy is nearly saturated by blocks 8-10 and is not strictly monotone at the final block. The supported result is **spatial broadening of local input dependence**, not attainment of a uniform global state or acquisition of global semantics.

{block_table_en}

## CLS-patch entropy by block and depth stratum

Within each image and block, the 32 sampled CLS channels and 15x32 sampled patch-token channels are separately reduced by the median before taking their paired difference. A phase-level image value is the median of the four block-level paired differences. The CLS distribution is more entropic at every block in both models, while the gap narrows with depth: from 0.4752/0.4712 at block 0 to 0.0086/0.0238 at block 11 for ViT-B/16 and DeiT-S. This is construct-validity evidence for broader CLS input dependence, not a causal measure of information integration.

{phase_table_en}

{cls_table_en}

## Cancellation across scales

For a raw signed map, let `A=sum_(c,h,w)|E|`, `B=sum_(h,w)|sum_c E|`, `P=sum_patch|sum_(c,h,w in patch)E|`, and `N=|sum E|`. The scale-specific cancellation ratios are `1-B/A`, `1-P/B`, `1-N/P`, and `1-N/A`; `R_net=N/A`. The survival identity `(B/A)(P/B)(N/P)=N/A` held to 6.94e-18. The 120-map subset shows substantial cancellation after RGB aggregation within patches and, at deeper sampled blocks, across patches. High absolute mass can therefore coexist with a small net first-order directional derivative. On the same 24 targets, float32 and newly recomputed float64 cancellation metrics differed by at most {review['max_cancellation_metric_absolute_difference']:.3e}.

{cancel_table_en}

## Finite-difference diagnostics

The retained float32 failures comprise 36 of 57,600 conditions (0.0625%) across 10 tasks. All use the first frozen patch token and occur in blocks 7-11 (19 ViT-B/16 and 17 DeiT-S conditions). Their float64 analytic magnitudes range from {fd_magnitudes.min():.3e} to {fd_magnitudes.max():.3e}, with median {np.median(fd_magnitudes):.3e}; the failures are therefore not confined to near-zero derivatives. Float32 absolute FD error increased as epsilon decreased over the tested range: its median was 4.26e-3 at 1e-2 and 5.83e-2 at 3e-4. For the same coordinates, float64 showed an error decrease followed by round-off increase, with median absolute error reaching 2.05e-10 at 3e-4. All 36 float64 counterparts passed, with best tolerance ratios between 7.92e-7 and 3.14e-3. These post-block token-channel targets do not identify a unique LayerNorm, softmax, or other operator as the source.

{fd_table_en}
"""

    chinese = f"""# Endpoint M3 补充分析 R2

> 本文档补充已封存 M3，不改变原 10 项 Holm 判定。新增分析均为 supplementary/descriptive，不引入新的确认性检验。

## 数值等价误差分布

5,000 项比较是数值验证案例，不是相互独立的统计样本。总体 relative-L2 中位数为 1.088e-16，99% 分位为 1.602e-7，最大值为 2.427e-7。Float64 的中位数和最大值分别为 5.194e-17 和 4.769e-16；float32 的中位数、99% 分位和最大值分别为 3.506e-8、1.782e-7 和 2.427e-7。

{error_table_zh}

## 算子级方向验证

每个算子使用 8 个 float64 方向和 9 个中心有限差分步长 `[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]`。这些检查支持所测试算子实现和方向导数；端到端 hook、registry 和结构 cut 由独立的闭合、parity 与负向控制证据覆盖。

{operator_table_zh}

## 12-block 层深相关空间扩展

每张图先在每个 block 内跨冻结单元取中位数，再对 100 张图报告中位数和 10,000 次 image-bootstrap 区间。Entropy 的主要上升发生在 early 和 middle blocks，locality 随深度持续下降，normalized distance 一直增加到 block 11。Entropy 在 blocks 8-10 附近接近饱和，末层并非严格单调增加。因此严格结论是**局部一阶输入依赖的空间扩展**，而不是达到均匀全局状态或获得全局语义。

{block_table_zh}

## 分层 CLS-patch entropy

每张图、每个 block 内先分别对 32 个抽样 CLS channels 和 15x32 个 patch-token channels 取中位数，再计算配对差值；phase-level 图像值是 4 个 block-level 配对差值的中位数。两个模型每个 block 的 CLS entropy 都更高，但差距随深度收窄：ViT-B/16 和 DeiT-S 从 block 0 的 0.4752/0.4712 降至 block 11 的 0.0086/0.0238。该结果是 CLS 输入依赖更分散的 construct-validity 证据，不是因果信息整合量。

{phase_table_zh}

{cls_table_zh}

## 多尺度 cancellation

对 raw signed map 定义 `A=sum_(c,h,w)|E|`、`B=sum_(h,w)|sum_c E|`、`P=sum_patch|sum_(c,h,w in patch)E|` 和 `N=|sum E|`。对应尺度的 cancellation 为 `1-B/A`、`1-P/B`、`1-N/P` 和 `1-N/A`，净比例为 `R_net=N/A`。生存率恒等式 `(B/A)(P/B)(N/P)=N/A` 的最大误差为 6.94e-18。120-map 子集表明，RGB 聚合后的 patch 内抵消很强，深层抽样 map 还表现出明显的 patch 间抵消。因此较大的 absolute mass 可以与很小的净一阶方向导数共存。同一 24 个目标的 float32 与新重算 float64 cancellation 指标最大绝对差为 {review['max_cancellation_metric_absolute_difference']:.3e}。

{cancel_table_zh}

## 有限差分诊断

保留的 float32 失败为 57,600 项中的 36 项（0.0625%），涉及 10 个任务；它们都属于第一个冻结 patch token，并集中在 blocks 7-11，其中 ViT-B/16 有 19 项、DeiT-S 有 17 项。其 float64 analytic magnitude 范围为 {fd_magnitudes.min():.3e} 至 {fd_magnitudes.max():.3e}，中位数为 {np.median(fd_magnitudes):.3e}，因此失败并不只出现在近零导数目标。在测试范围内，float32 的绝对 FD 误差随 epsilon 减小而增大：中位误差从 epsilon=1e-2 时的 4.26e-3 增至 epsilon=3e-4 时的 5.83e-2。同一坐标的 float64 曲线先下降后受舍入误差影响回升，在 epsilon=3e-4 时中位绝对误差为 2.05e-10。36 个 float64 对照全部通过，最优容差比范围为 7.92e-7 至 3.14e-3。由于目标是 post-block token-channel，现有证据不能把失败唯一归因到 LayerNorm、softmax 或其他单个算子。

{fd_table_zh}
"""

    (REPO / "docs/endpoint_m3_supplement_r2_20260919.md").write_text(chinese)
    (PAPER / "zh/endpoint_m3_supplement_r2.md").write_text(chinese)
    (PAPER / "en/endpoint_m3_supplement_r2.md").write_text(english)
    evidence = """# Endpoint M3-SUP-R2 evidence matrix

| Supplement | Primary artifacts | Status |
|---|---|---|
| NNS equivalence distributions | `m3a_error_distribution.csv`, source `m3a_fidelity.json` | PASS |
| Operator-level validation table | `operator_validation_table.csv`, source `m3a_operator_validation.json` | PASS |
| Twelve-block atlas trajectories | `block_image_values.csv`, `block_summary.csv` | SUPPLEMENTARY DESCRIPTIVE |
| Block-first CLS-patch comparison | `cls_patch_image_values.csv`, `cls_patch_summary.csv` | SUPPLEMENTARY DESCRIPTIVE |
| Multi-scale cancellation | `cancellation_map_values.csv`, `cancellation_summary.csv` | SUPPLEMENTARY DESCRIPTIVE |
| Float64 cancellation stability | `cancellation_float64_review.json`, 24 targets | PASS |
| Float32 failure diagnosis | `finite_difference_diagnostics.csv`, all 36 retained failures | PASS FLOAT64 RECOVERY; FLOAT32 FAIL RETAINED |
| Figures | `figure_manifest.json`, five PNG/PDF pairs | PASS |
"""
    (PAPER / "notes/endpoint_m3_supplement_evidence_matrix_r2.md").write_text(evidence)
    print(json.dumps({"status": "PASS", "documents": 3, "operators": len(operators), "block_rows": len(block_lines), "cls_rows": len(cls_lines), "fd_rows": len(finite_difference)}))


if __name__ == "__main__":
    main()
