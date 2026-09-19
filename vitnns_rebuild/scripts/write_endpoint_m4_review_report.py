#!/usr/bin/env python3
"""Write bilingual Endpoint M4 review revisions without editing the sealed draft."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_review_r2_20260919"
M4 = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
PAPER = REPO / "papers/nns_vit_endpoint"


def load(path):
    return json.loads(Path(path).read_text())


def rows(name):
    with (ROOT / name).open() as stream:
        output = list(csv.DictReader(stream))
    for row in output:
        for key in ("block", "n", "positive", "negative", "ties"):
            if key in row and row[key] != "":
                row[key] = int(row[key])
        for key in ("q10", "q25", "median", "q75", "q90", "p", "holm_p", "block_trajectory_spearman"):
            if key in row:
                row[key] = None if row[key] == "" else float(row[key])
        if "ci95" in row:
            row["ci95"] = None if row["ci95"] == "" else json.loads(row["ci95"])
    return output


def fmt(value, digits=4):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def sci(value):
    return "NA" if value is None else f"{float(value):.3e}"


def ci(value):
    return "NA" if value is None else f"[{value[0]:.4f}, {value[1]:.4f}]"


def select(table, **keys):
    return next(row for row in table if all(row.get(key) == value for key, value in keys.items()))


def revised_full_draft(language, supplement):
    source = (PAPER / language / "full_draft.md").read_text()
    start, end = "<!-- M4_MECHANISM_START -->", "<!-- M4_MECHANISM_END -->"
    replacement = start + "\n" + supplement + "\n" + end
    before, rest = source.split(start, 1); _, after = rest.split(end, 1)
    source = before + replacement + after
    if language == "en":
        abstract_anchor = "Endpoint therefore provides a numerically validated internal-target representation whose"
        abstract_insert = (
            "A target-conditioned analysis of the frozen ground-truth logit further showed that late attention "
            "had larger routing-to-total-path norm ratios and larger Value-only Jacobian discrepancies than early "
            "attention. Attention and MLP residuals had opposite identity-path depth trends, and their spatial "
            "changes also diverged. These ratios and normalized errors describe local Jacobian geometry rather than "
            "percent contributions.\n\n"
        )
        source = source.replace(abstract_anchor, abstract_insert + abstract_anchor, 1)
        source = source.replace(
            "Sixth, it shows depth-related spatial\nbroadening in a frozen token-channel sample from both models, while the behavioral studies\nseparately delimit component-level and reference-specific spatial utility.",
            "Sixth, it shows depth-related spatial broadening in a frozen token-channel sample and relates this "
            "atlas result to target-conditioned attention routing and sublayer-specific residual transmission. "
            "The behavioral studies separately delimit component-level and reference-specific spatial utility.", 1)
        discussion = """### 6.6 Target-conditioned routing and sublayer-specific residual transmission

M3 and M4 describe complementary levels of the local Jacobian. M3 measures how sampled post-block token-channel maps broaden over input coordinates. M4 fixes the ground-truth logit and partitions its downstream-cotangent-weighted input Jacobian through attention and residual paths. Late attention had larger routing path-norm ratios and larger Value-only approximation discrepancies in both models. Attention residuals showed higher late-minus-early spatial change, whereas MLP residuals showed lower late-minus-early spatial change. The pre-retained 16-image raw-map subset per model also showed increasing full-versus-Value patch-distribution distance over depth. Model-level block trajectories were directionally consistent with the M3 atlas, but the target and cohort mismatch precludes a unit-level or causal claim.

The saturation result is also submetric-specific. Mean-query saturation retained negative associations with routing scale and both approximation errors after controlling for head-output and downstream-cotangent norms. CLS-query associations were smaller; the fully adjusted ViT-B routing and Endpoint-error tests did not survive the post-hoc sensitivity Holm family. Mean-query saturation is therefore the supported summary, while CLS-query saturation provides a narrower sensitivity boundary.
"""
        source = source.replace("\n## 7. Limitations", "\n\n" + discussion + "\n## 7. Limitations", 1)
        limitation = """M4 conditions every path map on the downstream cotangent of the frozen ground-truth logit at the observed input. Routing and identity shares are relative path-norm scales, not additive percentages of the final Endpoint or causal effects. The normalized JVP discrepancy is a ratio of directional norms, not a percent omitted. The full-versus-Value spatial bridge uses 16 pre-retained class-logit maps per model; it is descriptive and is not paired to the 100-image token-channel M3 cohort.

"""
        source = source.replace("Preregistered Path-IG calibration", limitation + "Preregistered Path-IG calibration", 1)
        conclusion_header = "## 8. Conclusion"
        conclusion_anchor = "Under matched\nactivation replacement"
        conclusion_insert = (
            "For the frozen ground-truth logit, late attention was less accurately represented by a Value-only "
            "Jacobian, and attention versus MLP residuals showed opposite identity-path depth trends. These are "
            "target-conditioned local path-accounting results.\n\n"
        )
        before, conclusion = source.split(conclusion_header, 1)
        conclusion = conclusion.replace(conclusion_anchor, conclusion_insert + conclusion_anchor, 1)
        source = before + conclusion_header + conclusion
        source = source.replace("> Draft R2,", "> Draft R3 (M4 review revision),", 1)
        source = source.replace("and 33 paper figures are maintained", "33 sealed base figures and four M4-review supplemental figures are maintained", 1)
    else:
        abstract_anchor = "Endpoint 因而形成"
        abstract_insert = (
            "针对冻结 ground-truth logit 的目标条件化分析进一步显示：late attention 的 routing-to-total-path "
            "norm 比和 Value-only Jacobian discrepancy 均高于 early attention；attention residual 与 MLP "
            "residual 的 identity-path 层深趋势相反，空间改变也呈分化。这些比率和归一化误差描述局部 "
            "Jacobian 几何，不是贡献百分比。\n\n"
        )
        source = source.replace(abstract_anchor, abstract_insert + abstract_anchor, 1)
        source = source.replace(
            "第六，我们在两个模型的冻结 token-channel 样本中观察到层深\n相关的空间扩展，并由行为实验分别界定成分级与 reference-specific 空间效用。",
            "第六，我们在冻结 token-channel 样本中观察到层深相关的空间扩展，并把该 atlas 结果与目标条件化的 attention routing 和子层特异 residual 传递联系起来；行为实验另行界定成分级与 reference-specific 空间效用。", 1)
        discussion = """### 6.6 目标条件化 routing 与子层特异 residual 传递

M3 与 M4 描述局部 Jacobian 的两个层面。M3 测量抽样 post-block token-channel map 如何在输入坐标上扩展；M4 固定 ground-truth logit，并按 attention 和 residual 路径划分其 downstream-cotangent 加权输入 Jacobian。两个模型的 late attention 均表现出更大的 routing path-norm 比和 Value-only approximation discrepancy。Attention residual 的 late-minus-early 空间改变为正，MLP residual 则为负。每模型 16 个事前保留 raw maps 也显示 full 与 Value patch distribution 的距离随深度扩大。模型级 block 轨迹与 M3 atlas 在方向上相容，但目标与 cohort 不匹配，不能升级为逐单元或因果结论。

Saturation 结果依赖具体 summary。Mean-query saturation 在控制 head-output norm 与 downstream-cotangent norm 后，仍与 routing scale 和两类 approximation error 呈负相关。CLS-query 关系更弱；ViT-B 中充分调整后的 routing 与 Endpoint-error 检验未通过 post-hoc sensitivity Holm family。因此正文以 mean-query 为主结论，CLS-query 作为更窄的敏感性边界。
"""
        source = source.replace("\n## 7. 局限", "\n\n" + discussion + "\n## 7. 局限", 1)
        limitation = """M4 的每条路径 map 都以当前输入和冻结 ground-truth logit 的 downstream cotangent 为条件。Routing share 与 identity share 是相对路径 norm 尺度，不是最终 Endpoint 的可加百分比或因果效应；归一化 JVP discrepancy 是方向范数之比，不是遗漏百分比。Full–Value 空间桥接每模型仅使用 16 个事前保留 class-logit raw maps，属于描述性结果，且不与 100 图 token-channel M3 cohort 逐图配对。

"""
        source = source.replace("预注册 Path-IG calibration", limitation + "预注册 Path-IG calibration", 1)
        conclusion_header = "## 8. 结论"
        conclusion_anchor = "对于匹配 activation replacement"
        conclusion_insert = (
            "在冻结 ground-truth logit 下，late attention 更难由 Value-only Jacobian 近似；attention 与 MLP "
            "residual 的 identity-path 层深趋势相反。这些结论属于目标条件化的局部路径记账。\n\n"
        )
        before, conclusion = source.split(conclusion_header, 1)
        conclusion = conclusion.replace(conclusion_anchor, conclusion_insert + conclusion_anchor, 1)
        source = before + conclusion_header + conclusion
    return source


def main():
    jvp = rows("jvp_block_summary.csv"); saturation = rows("saturation_sensitivity_summary.csv")
    residual = rows("residual_spatial_summary.csv"); residual_tests = load(ROOT / "residual_spatial_tests.json")["tests"]
    bridge = rows("bridge_block_summary.csv"); trajectory = rows("m3_m4_trajectory_relations.csv")
    numeric = load(ROOT / "numeric_error_locations.json")["records"]
    original = load(M4 / "m4_tests.json")["tests"]
    def j(model, block, metric): return select(jvp, model_id=model, block=block, metric=metric)
    def s(model, sat, outcome, adjustment, phase="all"):
        return select(saturation, model_id=model, phase=phase, saturation=sat, outcome=outcome, adjustment=adjustment)
    def r(model, kind, metric): return select(residual, model_id=model, residual_type=kind, metric=metric)
    def b(model, block, metric): return select(bridge, model_id=model, block=block, metric=metric)
    jvp_lines = []
    for model in ("vit_b16", "deit_s16"):
        for metric in ("relative_error", "absolute_error", "full_response_denominator"):
            row = j(model, 11, metric)
            jvp_lines.append(f"| {model} | {metric} | {fmt(row['q10'])} | {fmt(row['q25'])} | {fmt(row['median'])} {ci(row['ci95'])} | {fmt(row['q75'])} | {fmt(row['q90'])} |")
    sat_lines = []
    for model in ("vit_b16", "deit_s16"):
        for sat in ("mean_query_saturation", "cls_query_saturation"):
            for outcome in ("routing_share", "frozen_attention_endpoint_error", "frozen_attention_jvp_error"):
                unadjusted = s(model, sat, outcome, "unadjusted")
                adjusted = s(model, sat, outcome, "adjust_output_and_cotangent_norm")
                sat_lines.append(f"| {model} | {sat.replace('_saturation','')} | {outcome} | {fmt(unadjusted['median'])} | {fmt(adjusted['median'])} {ci(adjusted['ci95'])} |")
    phase_lines = []
    for model in ("vit_b16", "deit_s16"):
        for phase in ("early", "middle", "late"):
            row = s(model, "mean_query_saturation", "frozen_attention_endpoint_error", "adjust_output_and_cotangent_norm", phase)
            phase_lines.append(f"| {model} | {phase} | {fmt(row['median'])} | {fmt(row['q10'])} | {fmt(row['q90'])} |")
    spatial_lines = []
    for model in ("vit_b16", "deit_s16"):
        for kind in ("attention", "mlp"):
            test = next(row for row in residual_tests if row["model_id"] == model and row["residual_type"] == kind)
            relation = r(model, kind, "branch_share_spatial_tv_spearman")
            spatial_lines.append(f"| {model} | {kind} | {fmt(test['median'])} {ci(test['ci95'])} | {test['positive']}/{test['negative']}/{test['ties']} | {sci(test['holm_p'])} | {fmt(relation['median'])} {ci(relation['ci95'])} |")
    cancellation_lines = []
    for model in ("vit_b16", "deit_s16"):
        test = next(row for row in original if row["model_id"] == model and row["domain"] == "residual" and row["residual_type"] == "mlp" and row["test_id"] == "late_minus_early_path_cancellation")
        cancellation_lines.append(f"| {model} | {fmt(test['median'])} {ci(test['ci95'])} | {test['positive']}/{test['negative']}/{test['ties']} | {sci(test['holm_p'])} |")
    bridge_lines = []
    for model in ("vit_b16", "deit_s16"):
        for block in (0, 5, 11):
            bridge_lines.append(f"| {model} | {block} | {fmt(b(model, block, 'full_entropy')['median'])} | {fmt(b(model, block, 'value_entropy')['median'])} | {fmt(b(model, block, 'full_minus_value_entropy')['median'])} | {fmt(b(model, block, 'full_value_patch_tv')['median'])} {ci(b(model, block, 'full_value_patch_tv')['ci95'])} |")
    trajectory_lines = [f"| {row['model_id']} | {row['m3_metric']} | {row['m4_bridge_metric']} | {fmt(row['block_trajectory_spearman'])} |" for row in trajectory]
    numeric_lines = [f"| {row['metric']} | {row['model_id']} | {row['sample_id']} | {row['block']} | {row['head'] or row['residual_type']} | {sci(row['value'])} |" for row in numeric]
    common_tables = {
        "jvp": "\n".join(jvp_lines), "sat": "\n".join(sat_lines), "phase": "\n".join(phase_lines),
        "spatial": "\n".join(spatial_lines), "cancellation": "\n".join(cancellation_lines),
        "bridge": "\n".join(bridge_lines), "trajectory": "\n".join(trajectory_lines), "numeric": "\n".join(numeric_lines),
    }
    zh = f"""# Endpoint M4 外部评价修订结果 R2

## 结论

M4 测量冻结 ground-truth logit 的目标条件化局部输入 Jacobian 路径记账；每个分量都包含该目标的下游 cotangent。Routing share 与 identity share 是两条路径 absolute Endpoint norm 的相对尺度，JVP error 是归一化 RMS 方向差异。它们分别描述路径尺度和方向响应偏差，不是函数值或因果贡献百分比。

三项主发现得到保留和收紧：late attention 的 routing relative path-norm scale 与 Value-only approximation error 高于 early attention；mean-query saturation 与 routing scale/error 呈负相关；attention residual 与 MLP residual 的 identity-path depth trend 相反。补充分析进一步排除了 block-11 JVP relative error 由近零分母普遍放大的解释，并为 attention residual 的空间改变提供了完整 400 图推断。

## JVP discrepancy 的分布和分母

| 模型 | Block-11 指标 | q10 | q25 | median [95% image-bootstrap CI] | q75 | q90 |
|---|---|---:|---:|---:|---:|---:|
{common_tables['jvp']}

Block 11 的归一化 RMS discrepancy 中位数为 0.9516 和 0.9636。对应 full-Jacobian response denominator 的图像级 head-median 分别为 14.9711（q10=5.5214）和 1.5155（q10=0.5719），而 absolute routing discrepancy 中位数为 13.8064 和 1.4228。因此高 relative error 伴随同量级 absolute discrepancy，并非由普遍近零分母造成。该比值应解释为方向响应差异与完整响应同量级，而非可加的遗漏或路径贡献分数。

## Saturation 稳健性

| 模型 | Saturation | Outcome | Unadjusted rho | 调整 output/cotangent norm 后 rho [95% CI] |
|---|---|---|---:|---:|
{common_tables['sat']}

Mean-query saturation 在两个模型和三个 outcome 上，控制 head projection-input activation norm 与 downstream cotangent norm 后仍保持负相关。CLS-query 关系更弱：ViT-B 的 fully adjusted routing-share 与 Endpoint-error 检验在 18 项 model-specific sensitivity Holm family 中未通过（Holm p 均为 0.0629），而 JVP error 及 DeiT-S 的三项关系仍通过。由此，正式结果应以 all-query saturation 为主，不把单一 CLS row 当作等价代理。

| 模型 | Phase | Fully adjusted mean-query saturation–Endpoint-error rho | q10 | q90 |
|---|---|---:|---:|---:|
{common_tables['phase']}

负相关主要集中在 early/middle blocks；late phase 接近零。这进一步限定了“sharpness 不是 routing proxy”：它是同图同层 head-level、且具有深度异质性的相关关系，不是 saturation 导致 routing 变化的因果证据。

## Residual 空间改变

空间指标为 output 与 identity 的归一化 14×14 patch absolute distribution total variation。

| 模型 | Residual | Late−early spatial TV [95% CI] | 正/负/平图像 | Holm p | Branch-share–TV rho [95% CI] |
|---|---|---:|---:|---:|---:|
{common_tables['spatial']}

两个模型的 attention residual 在 late blocks 相对 early blocks具有更大的 output–identity 空间差异，ViT-B/16 为 +0.0520，DeiT-S 为 +0.0282；MLP residual 分别为 −0.0187 和 −0.0462。四项 post-hoc spatial sensitivity tests 均通过 Holm。Branch relative path-norm scale 与 spatial TV 的图像内跨层相关也为正。因而“attention residual 更具空间重写特征”现在同时有 share 和空间分布证据；结论仍限定为目标条件化局部 Jacobian，而不是有限干预效应。

## MLP path cancellation 的模型差异

| 模型 | Late−early path cancellation [95% CI] | 正/负/平图像 | 原 M4 Holm p |
|---|---:|---:|---:|
{common_tables['cancellation']}

ViT-B/16 中 MLP identity–branch cancellation 随深度上升，DeiT-S 中下降；两项原冻结检验均通过校正。因此可写“呈相反的显著趋势”，但不能归纳为跨模型统一的 MLP cancellation 规律。

## M3–M4 描述性桥接

E3 事前保留的每模型 16 个 class-logit raw maps 允许比较完整 attention、routing 和 Value 路径的 patch absolute distributions。

| 模型 | Block | H(full) | H(value) | H(full)−H(value) | Full–Value patch TV [95% CI] |
|---|---:|---:|---:|---:|---:|
{common_tables['bridge']}

Full–Value patch TV 从 block 0 的 0.0550/0.0307 增至 block 11 的 0.1865/0.1811，说明深层 Value-only approximation 与完整 attention 的空间形态差异更大。Entropy difference 不是单调且在 ViT-B block 11 跨零，因此不支持“routing 可加地增加 entropy”。

| 模型 | M3 指标 | M4 bridge 指标 | 12-block median trajectory Spearman |
|---|---|---|---:|
{common_tables['trajectory']}

M3 entropy 与 Full–Value TV 的 block-trajectory rho 为 0.2727/0.4406，locality 为 −0.3357/−0.5105，distance 为 0.3357/0.5105。方向与 M3 spatial broadening 相容，但 M3 使用 100 图 token-channel targets，而 raw bridge 使用 16 图 class-logit targets；每模型仅 1 张图重叠。因此这些值只构成模型级深度轨迹桥接，不识别 routing 对 M3 空间扩展的因果效应。

## Cut、cotangent 与数值位置

Attention cut 位于每个 head 的 AV 输出、head concatenation 和 output projection 之前；concat、projection 和所有后续层的作用进入该 AV edge 的 downstream cotangent。Q/K 取 q/k normalization 后的实际 edge，Value 取 value projection edge。模型是 pre-LN timm ViT：第一处 residual 为 u + attention(norm1(u))，使用 clean g_r1；第二处为 r1 + MLP(norm2(r1))，使用 clean g_r2。Identity 与 branch 始终共享同一 residual-output cotangent。

| 数值诊断 | 模型 | 图像 | Block | Head/Residual | 最大误差 |
|---|---|---|---:|---|---:|
{common_tables['numeric']}

最大显式-vs-native JVP relative error 为 2.502e-5，位于 ViT-B/16 block 0 head 3；其余最大误差位置如表。闭合、实现正确性和行为含义继续分开报告。

## 修订后的一句话结论

在所测试模型、当前输入和冻结 ground-truth logit 下，late attention 的目标条件化局部输入 Jacobian 更难由固定 attention weights 的 Value-only 路径近似；attention residual 的 branch-relative path norm 与空间改变随深度增强，而 MLP residual 的 identity-relative path norm 增强。以上是局部 Jacobian 路径和空间分布结果，不是函数值贡献比例或有限因果重要性。
"""
    en = f"""# Endpoint M4 external-review revision R2

## Conclusion

M4 estimates target-conditioned local input-Jacobian path accounting for the frozen ground-truth logit; every component includes that target's downstream cotangent. Routing and identity shares are relative absolute-Endpoint path-norm scales, and JVP error is a normalized RMS directional discrepancy. They describe path scale and directional mismatch rather than percentages of a function value or causal effect.

Three findings remain after this refinement: late attention has larger routing relative path-norm scale and Value-only approximation error than early attention; mean-query saturation is negatively associated with routing scale/error; and attention versus MLP residuals have opposite identity-path depth trends. The review analysis also rules out generally near-zero block-11 denominators and adds full-cohort spatial inference for the attention-residual rewrite interpretation.

## JVP discrepancy distribution and denominator

| Model | Block-11 metric | q10 | q25 | median [95% image-bootstrap CI] | q75 | q90 |
|---|---|---:|---:|---:|---:|---:|
{common_tables['jvp']}

The block-11 normalized RMS discrepancy medians are 0.9516 and 0.9636. The corresponding image-level head-median full-Jacobian response denominators are 14.9711 (q10=5.5214) and 1.5155 (q10=0.5719), while the absolute routing discrepancies are 13.8064 and 1.4228. The high relative errors therefore accompany same-order absolute discrepancies rather than generally near-zero denominators. They indicate directional differences comparable in norm with the full response, not an additive omission or path-contribution fraction.

## Saturation sensitivity

| Model | Saturation | Outcome | Unadjusted rho | After output/cotangent-norm adjustment rho [95% CI] |
|---|---|---|---:|---:|
{common_tables['sat']}

Mean-query saturation remains negatively associated with all three outcomes in both models after adjustment for the head projection-input activation norm and downstream-cotangent norm. CLS-query relations are weaker: the fully adjusted ViT-B routing-share and Endpoint-error tests do not survive the 18-test model-specific sensitivity Holm family (both Holm p=0.0629), whereas its JVP-error relation and all three DeiT-S relations remain supported. The formal summary should therefore use all-query saturation; one CLS row is not an equivalent proxy.

| Model | Phase | Fully adjusted mean-query saturation–Endpoint-error rho | q10 | q90 |
|---|---|---:|---:|---:|
{common_tables['phase']}

The negative association is concentrated in early and middle blocks and approaches zero in the late phase. Thus attention sharpness is not a positive routing proxy under this within-image, within-layer head comparison; the association does not establish that saturation causes routing changes.

## Residual spatial change

The spatial metric is total variation between the output and identity normalized 14×14 patch absolute distributions.

| Model | Residual | Late−early spatial TV [95% CI] | Pos/neg/tie images | Holm p | Branch-share–TV rho [95% CI] |
|---|---|---:|---:|---:|---:|
{common_tables['spatial']}

Attention residuals have larger output–identity spatial differences in late than early blocks in both models (+0.0520 for ViT-B/16 and +0.0282 for DeiT-S), whereas MLP residual differences decrease (−0.0187 and −0.0462). All four post-hoc spatial sensitivity tests survive Holm correction. The within-image cross-block association between branch relative path-norm scale and spatial TV is also positive. The attention-residual rewrite description is therefore supported jointly by relative norm and spatial-distribution evidence, while remaining a target-conditioned local-Jacobian result rather than a finite intervention effect.

## Model-dependent MLP path cancellation

| Model | Late−early path cancellation [95% CI] | Pos/neg/tie images | Original M4 Holm p |
|---|---:|---:|---:|
{common_tables['cancellation']}

MLP identity–branch path cancellation increases with depth in ViT-B/16 and decreases in DeiT-S; both original frozen tests survive correction. The paper can state that the models exhibit significant opposite trends, but not a model-invariant MLP cancellation law.

## Descriptive M3–M4 bridge

The 16 pre-retained E3 class-logit raw maps per model permit comparison of complete-attention, routing, and Value patch absolute distributions.

| Model | Block | H(full) | H(value) | H(full)−H(value) | Full–Value patch TV [95% CI] |
|---|---:|---:|---:|---:|---:|
{common_tables['bridge']}

Full–Value patch TV rises from 0.0550/0.0307 at block 0 to 0.1865/0.1811 at block 11, indicating larger spatial-shape differences between late complete attention and its Value-only approximation. Entropy differences are nonmonotonic and cross zero for ViT-B at block 11, so the data do not support an additive claim that routing increases entropy.

| Model | M3 metric | M4 bridge metric | 12-block median trajectory Spearman |
|---|---|---|---:|
{common_tables['trajectory']}

M3 entropy versus Full–Value TV has block-trajectory rho 0.2727/0.4406; locality has −0.3357/−0.5105 and distance 0.3357/0.5105. These directions are compatible with M3 spatial broadening, but M3 uses 100-image token-channel targets whereas the raw bridge uses 16-image class-logit targets, with only one overlapping image per model. The result is a model-level depth-trajectory bridge and does not identify a causal effect of routing on the M3 broadening.

## Cut, cotangent, and numerical locations

The attention cut is each head's AV output before head concatenation and output projection; concatenation, projection, and all later layers enter the downstream cotangent at that AV edge. Q/K are the actual post-q/k-normalization edges and Value is the value-projection edge. The timm models are pre-LN: the first residual is u + attention(norm1(u)) with clean g_r1, and the second is r1 + MLP(norm2(r1)) with clean g_r2. Identity and branch use the same clean residual-output cotangent.

| Numerical diagnostic | Model | Image | Block | Head/Residual | Maximum error |
|---|---|---|---:|---|---:|
{common_tables['numeric']}

The largest explicit-versus-native JVP relative error is 2.502e-5 at ViT-B/16 block 0 head 3. The table locates the other maxima. Algebraic closure, implementation correctness, and behavioral interpretation remain separate evidence claims.

## Revised one-sentence conclusion

For the tested models, observed inputs, and frozen ground-truth logit, late attention's target-conditioned local input Jacobian is less accurately approximated by Value-only propagation with fixed attention weights; attention residuals show increasing branch-relative path norm and spatial change, whereas MLP residuals show increasing identity-relative path norm. These are local Jacobian path and spatial-distribution results, not function-value percentages or finite causal importance.
"""
    (REPO / "docs/endpoint_m4_review_results_20260919.md").write_text(zh)
    (PAPER / "zh/endpoint_m4_review_r2.md").write_text(zh)
    (PAPER / "en/endpoint_m4_review_r2.md").write_text(en)
    marker_zh = "### 5.8 M4：目标条件化 attention routing/value 与 residual 机制\n\n" + "\n\n".join(zh.split("\n\n")[2:]).replace("\n## ", "\n#### ")
    marker_en = "### 5.8 M4: Target-conditioned attention routing/value and residual mechanisms\n\n" + "\n\n".join(en.split("\n\n")[2:]).replace("\n## ", "\n#### ")
    revised_zh = revised_full_draft("zh", marker_zh); revised_en = revised_full_draft("en", marker_en)
    (PAPER / "zh/full_draft_m4_review_r2.md").write_text(revised_zh)
    (PAPER / "en/full_draft_m4_review_r2.md").write_text(revised_en)
    evidence = """# Endpoint M4 external-review R2 evidence matrix

| Review issue | Evidence | Decision |
|---|---|---|
| Target-conditioned scope | Frozen ground-truth logit and downstream cotangents in M4 records and implementation | ADOPTED |
| Path norm is not contribution percentage | Exact estimand definitions and revised wording | ADOPTED |
| Block-11 JVP denominator | Full-cohort q10–q90 denominator and absolute discrepancy | NOT_SMALL_DENOMINATOR_ARTIFACT |
| Mean-query saturation robustness | Output- and cotangent-norm partial rank; phase stratification | NEGATIVE_ASSOCIATION_RETained |
| CLS-query robustness | Fully adjusted ViT-B routing and Endpoint-error Holm p=0.0629 | QUALIFIED |
| Residual spatial rewrite | Full-cohort late–early spatial TV and branch-share relation | ATTENTION_INCREASE_MLP_DECREASE |
| MLP cancellation | Original M4 corrected tests | SIGNIFICANT_OPPOSITE_MODEL_TRENDS |
| M3–M4 bridge | 16 pre-retained class-logit raw maps/model and 12-block trajectory comparison | DESCRIPTIVE_ONLY |
| Causal claim | No finite intervention added | NOT_AUTHORIZED |
"""
    (PAPER / "notes/endpoint_m4_review_r2_evidence_matrix.md").write_text(evidence)
    decision = {
        "status": "M4_REVIEW_R2_COMPLETE_POST_HOC_SENSITIVITY",
        "original_M4_status_unchanged": "ENDPOINT_M4_PASS_EXECUTION_AND_AUDIT",
        "claims": {
            "target_conditioned_scope": "ADOPTED",
            "block11_jvp_small_denominator_artifact": "NOT_SUPPORTED",
            "mean_query_saturation_negative_association_after_adjustment": "SUPPORTED_SENSITIVITY",
            "cls_query_equivalent_robustness": "QUALIFIED",
            "attention_residual_late_spatial_change": "SUPPORTED_POST_HOC_SENSITIVITY",
            "mlp_residual_late_spatial_change": "SUPPORTED_DECREASE_POST_HOC_SENSITIVITY",
            "mlp_path_cancellation_model_invariant_trend": "NOT_SUPPORTED_OPPOSITE_MODEL_TRENDS",
            "routing_causes_M3_spatial_broadening": "NOT_SUPPORTED_DESCRIPTIVE_BRIDGE_ONLY"
        },
        "next_authorization": "STOP_AFTER_M4_REVIEW_R2"
    }
    (ROOT / "scientific_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    shutil.copyfile(ROOT / "figure_manifest.json", PAPER / "figure_manifest_m4_review_r2.json")
    print(json.dumps({"status": "PASS", "documents": 5, "jvp_rows": len(jvp_lines),
                      "saturation_rows": len(sat_lines), "spatial_rows": len(spatial_lines),
                      "bridge_rows": len(bridge_lines)}))


if __name__ == "__main__":
    main()
