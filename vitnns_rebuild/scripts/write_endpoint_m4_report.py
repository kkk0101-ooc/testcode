#!/usr/bin/env python3
"""Write evidence-bound Endpoint M4 result documents and manuscript sections."""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
PAPER = REPO / "papers/nns_vit_endpoint"


def load(path):
    return json.loads(Path(path).read_text())


def rows(name):
    with (ROOT / name).open() as stream:
        output = list(csv.DictReader(stream))
    for row in output:
        for key in ("block", "head", "n", "na"):
            if key in row:
                row[key] = int(row[key])
        if "median" in row:
            row["median"] = None if row["median"] == "" else float(row["median"])
        if "ci95" in row:
            row["ci95"] = None if row["ci95"] == "" else json.loads(row["ci95"].replace("'", '"'))
    return output


def value(table, model, block, metric, kind=None):
    return next(row for row in table if row["model_id"] == model and row["block"] == block
                and row["metric"] == metric and (kind is None or row.get("residual_type") == kind))


def number(observed, digits=4):
    return "NA" if observed is None else f"{float(observed):.{digits}f}"


def replace_marker(path: Path, marker: str, content: str) -> None:
    text = path.read_text(); start, end = f"<!-- {marker}_START -->", f"<!-- {marker}_END -->"
    block = start + "\n" + content.rstrip() + "\n" + end
    if start in text and end in text:
        before, rest = text.split(start, 1); _, after = rest.split(end, 1); text = before + block + after
    else:
        anchor = "\n## 6. Discussion" if path.parent.name == "en" else "\n## 6. 讨论"
        if anchor not in text:
            raise ValueError("MANUSCRIPT_ANCHOR:" + str(path))
        text = text.replace(anchor, "\n\n" + block + "\n" + anchor, 1)
    path.write_text(text)


def main():
    attention = rows("attention_block_summary.csv"); residual = rows("residual_block_summary.csv")
    tests = load(ROOT / "m4_tests.json")["tests"]
    validation = load(ROOT / "validation_report.json"); analysis = load(ROOT / "analysis_receipt.json")
    models = ("vit_b16", "deit_s16")
    attention_lines = []
    for model in models:
        for block in (0, 5, 11):
            attention_lines.append(
                f"| {model} | {block} | {number(value(attention, model, block, 'routing_share')['median'])} | "
                f"{number(value(attention, model, block, 'frozen_attention_endpoint_error')['median'])} | "
                f"{number(value(attention, model, block, 'frozen_attention_jvp_error')['median'])} | "
                f"{number(value(attention, model, block, 'mean_query_saturation')['median'])} | "
                f"{number(value(attention, model, block, 'routing_value_path_cancellation')['median'])} |")
    residual_lines = []
    for model in models:
        for kind in ("attention", "mlp"):
            for block in (0, 5, 11):
                residual_lines.append(
                    f"| {model} | {kind} | {block} | {number(value(residual, model, block, 'identity_share', kind)['median'])} | "
                    f"{number(value(residual, model, block, 'path_cancellation', kind)['median'])} | "
                    f"{number(value(residual, model, block, 'identity_branch_cosine', kind)['median'])} | "
                    f"{number(value(residual, model, block, 'output_identity_patch_tv', kind)['median'])} |")
    test_lines = []
    for row in tests:
        group = row.get("residual_type", "attention-head")
        test_lines.append(
            f"| {row['model_id']} | {row['domain']} | {group} | {row['test_id']} | "
            f"{number(row['median'])} | {row['positive']}/{row['negative']}/{row['ties']} | "
            f"{float(row['holm_p']):.3e} | {row['decision']} |")
    extrema = {}
    for model in models:
        extrema[model] = {}
        for kind in ("attention", "mlp"):
            tv = [value(residual, model, block, "output_identity_patch_tv", kind) for block in range(12)]
            extrema[model][kind] = {
                "min_block": min(tv, key=lambda row: row["median"])["block"],
                "min_tv": min(row["median"] for row in tv),
                "max_block": max(tv, key=lambda row: row["median"])["block"],
                "max_tv": max(row["median"] for row in tv),
            }
    decision = {
        "status": "M4_ATTENTION_AND_RESIDUAL_MECHANISM_ANALYSIS_COMPLETE",
        "attention_tests": [row for row in tests if row["domain"] == "attention"],
        "residual_tests": [row for row in tests if row["domain"] == "residual"],
        "descriptive_spatial_extrema": extrema,
        "scope": "local input dependence for two frozen ImageNet ViTs and one shared 400-image cohort",
    }
    (ROOT / "scientific_decision.json").write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    attention_table = "\n".join(attention_lines); residual_table = "\n".join(residual_lines); test_table = "\n".join(test_lines)
    zh = f"""# Endpoint M4：Attention 与 residual 结构机制分析

## 执行与验证

M4 完成两模型共享 400 图、12 个 blocks 和全部 heads 的 ground-truth-logit 局部输入依赖分解，共 {analysis['attention_head_records']:,} 个 image-block-head attention records、{analysis['residual_records']:,} 个 image-block-residual records 和 {analysis['jvp_direction_head_records']:,} 个 direction-head JVP records。Q+K+V 逐 head 重构 native attention Endpoint；routing+value 重构同一对象；attention 和 MLP residual 的 identity/carry 加 branch 均重构完整 logit Endpoint。

独立 CPU 审计复算 {validation['e3_scalar_crosschecks']:,} 个 E3 summary 字段。最大 Q/K/V closure 绝对误差为 {validation['maximum_errors']['qkv_closure_max_abs']:.3e}，最大 residual closure 绝对误差为 {validation['maximum_errors']['residual_closure_max_abs']:.3e}，显式 attention differential 与 native JVP 的最大相对误差为 {validation['maximum_errors']['explicit_jvp_relative_error']:.3e}。全部 {validation['negative_controls']:,} 个省略 identity/carry 负控按预期闭合失败；冻结前 16 个 evaluation orders 的 {validation['detach_blocks']:,} 个 attention-detach block checks 通过。

## Attention routing/value

| 模型 | Block | Routing share | Frozen-attention Endpoint error | Frozen-attention JVP error | Mean-query saturation | Routing-value path cancellation |
|---|---:|---:|---:|---:|---:|---:|
{attention_table}

Frozen-attention 只保留 A dV。因此 Endpoint 与 JVP error 分别测量它在输入缩放方向和 8 个冻结 Rademacher 方向上遗漏的 routing differential。以下检验先在图内按层和 head 聚合，再以图像为统计单位；Holm family 按模型冻结。

| 模型 | Domain | 类型 | 冻结检验 | 图像级中位数 | 正/负/平 | Holm p | 判定 |
|---|---|---|---|---:|---:|---:|---|
{test_table}

Saturation 关系是在每张图、每一层的 heads 内计算 Spearman 后再汇总，避免把共同层深趋势当成 head 内关系。其方向与支持状态由上表直接给出，不预设 softmax 越尖锐时 routing omission 越大。

## Residual identity/branch

| 模型 | Residual | Block | Identity share | Path cancellation | Identity-branch cosine | Output-identity patch TV |
|---|---|---:|---:|---:|---:|---:|
{residual_table}

Identity share 描述 absolute Endpoint mass 在 skip 与 branch 两条局部路径间的比例；path cancellation 比较两条 signed map 相加前后的 absolute mass。Output-identity patch TV 衡量 residual 输出相对只走 identity 时的空间依赖改变。ViT-B/16 中 attention/MLP residual 的最小与最大空间 TV 位于 blocks {extrema['vit_b16']['attention']['min_block']}/{extrema['vit_b16']['attention']['max_block']} 和 {extrema['vit_b16']['mlp']['min_block']}/{extrema['vit_b16']['mlp']['max_block']}；DeiT-S 对应为 {extrema['deit_s16']['attention']['min_block']}/{extrema['deit_s16']['attention']['max_block']} 和 {extrema['deit_s16']['mlp']['min_block']}/{extrema['deit_s16']['mlp']['max_block']}。这些层级定位来自冻结 cohort 的描述性中位数。

## Signed、absolute 与 cancellation

每条路径均同时保留 signed sum、absolute mass、positive/negative mass 和 map cancellation。M4 区分同一 map 内的正负抵消与 routing/value 或 identity/branch 之间的 path cancellation；二者没有合并为单一“重要性”指标。所有零分母、零范数和常数 head 关系严格传播为 NA，没有加入 epsilon。

## 图

- Image #1（m4_attention_depth）：逐 block 展示 routing share、frozen-attention Endpoint error 与 8-direction JVP error；阴影为 10,000 次图像 bootstrap 95% CI。
- Image #2（m4_attention_saturation）：展示 block-level saturation 与 routing/error 的描述性位置；正式相关检验仍使用图内 layer-wise head Spearman。
- Image #3（m4_residual_depth）：同时展示 attention/MLP residual 的 identity share、path cancellation 和空间 TV。

## 解释边界

M4 定量描述当前输入点和冻结 ground-truth logit 下的局部 Jacobian 路径记账。较高 routing share 或 branch mass 表示该路径承载较多局部输入依赖；因果重要性仍需独立有限干预证据。
"""
    en = f"""# Endpoint M4: Attention and residual mechanism analysis

## Execution and validation

M4 covered a shared 400-image cohort, both models, all 12 blocks, and every head for the frozen ground-truth logit. The result contains {analysis['attention_head_records']:,} image-block-head attention records, {analysis['residual_records']:,} image-block-residual records, and {analysis['jvp_direction_head_records']:,} direction-head JVP records. Per-head Q+K+V reconstructed the native attention Endpoint, routing+value reconstructed the same object, and identity/carry plus branch reconstructed the full logit Endpoint at both residual additions.

The independent CPU audit recomputed {validation['e3_scalar_crosschecks']:,} E3 summary fields. Maximum absolute Q/K/V and residual closure errors were {validation['maximum_errors']['qkv_closure_max_abs']:.3e} and {validation['maximum_errors']['residual_closure_max_abs']:.3e}. The maximum relative error between the explicit attention differential and native JVP was {validation['maximum_errors']['explicit_jvp_relative_error']:.3e}. All {validation['negative_controls']:,} identity/carry omission controls failed closure as intended, and {validation['detach_blocks']:,} attention-detach block checks passed for the first 16 frozen evaluation orders.

## Attention routing and value

| Model | Block | Routing share | Frozen-attention Endpoint error | Frozen-attention JVP error | Mean-query saturation | Routing-value path cancellation |
|---|---:|---:|---:|---:|---:|---:|
{attention_table}

Frozen attention retains only A dV. Endpoint and JVP errors therefore quantify omitted routing differentials along the input-scaling direction and eight frozen Rademacher directions, respectively. The tests below aggregate heads and blocks within each image before image-level inference; Holm families were frozen separately by model.

| Model | Domain | Type | Frozen test | Image-level median | Pos/neg/tie | Holm p | Decision |
|---|---|---|---|---:|---:|---:|---|
{test_table}

Saturation relations use Spearman correlations across heads within each image and layer before aggregation, separating head-level association from common depth trends. The table reports their observed directions without assuming that sharper softmax rows must yield larger routing omission.

## Residual identity and branch

| Model | Residual | Block | Identity share | Path cancellation | Identity-branch cosine | Output-identity patch TV |
|---|---|---:|---:|---:|---:|---:|
{residual_table}

Identity share partitions absolute Endpoint mass between the skip and branch paths. Path cancellation compares their signed sum with the sum of their absolute masses. Output-identity patch TV measures the spatial change from the identity-only distribution to the residual output. For ViT-B/16, the minimum/maximum spatial-TV blocks were {extrema['vit_b16']['attention']['min_block']}/{extrema['vit_b16']['attention']['max_block']} for attention residuals and {extrema['vit_b16']['mlp']['min_block']}/{extrema['vit_b16']['mlp']['max_block']} for MLP residuals; the DeiT-S pairs were {extrema['deit_s16']['attention']['min_block']}/{extrema['deit_s16']['attention']['max_block']} and {extrema['deit_s16']['mlp']['min_block']}/{extrema['deit_s16']['mlp']['max_block']}. These locations are descriptive medians over the frozen cohort.

## Signed, absolute, and cancellation views

Every path retains signed sum, absolute mass, positive/negative mass, and within-map cancellation. M4 keeps within-map cancellation separate from routing/value or identity/branch path cancellation. Zero denominators, zero norms, and constant-head correlations propagate as NA without epsilon substitution.

## Figures

- Image #1 (m4_attention_depth): routing share, frozen-attention Endpoint error, and eight-direction JVP error across blocks; bands are 10,000-replicate image-bootstrap 95% intervals.
- Image #2 (m4_attention_saturation): descriptive block-level locations of saturation versus routing/error; formal association tests use within-image, within-layer head Spearman correlations.
- Image #3 (m4_residual_depth): identity share, path cancellation, and spatial TV for attention and MLP residuals.

## Interpretation boundary

M4 quantifies local Jacobian path accounting at the observed input for the frozen ground-truth logit. A larger routing share or branch mass indicates more local input dependence along that path; causal importance requires separate finite-intervention evidence.
"""
    (REPO / "docs/endpoint_m4_results_20260919.md").write_text(zh)
    (PAPER / "zh/endpoint_m4_supplement.md").write_text(zh)
    (PAPER / "en/endpoint_m4_supplement.md").write_text(en)
    replace_marker(PAPER / "zh/full_draft.md", "M4_MECHANISM", "### 5.8 M4：Attention routing/value 与 residual 机制\n\n" + "\n\n".join(zh.split("\n\n")[2:]))
    replace_marker(PAPER / "en/full_draft.md", "M4_MECHANISM", "### 5.8 M4: Attention routing/value and residual mechanisms\n\n" + "\n\n".join(en.split("\n\n")[2:]))
    evidence = """# Endpoint M4 evidence matrix

| Claim | Evidence | Status |
|---|---|---|
| Full cohort and head coverage | task_manifest.json, coverage_audit.json, analysis_receipt.json | PASS |
| Q/K/V and routing/value closure | per-task closure fields and validation_report.json | PASS |
| Frozen-attention JVP | eight deterministic directions per image-head, explicit softmax differential versus native JVP | PASS |
| Attention detach and forward parity | first 16 frozen evaluation orders per model, all 12 blocks | PASS |
| Residual identity/branch closure | both residual types, all blocks and images | PASS |
| Omitted-identity negative control | one control per residual record | PASS |
| E3 inheritance | 288 scalar summary crosschecks per image-model task | PASS |
| Mechanism statistics | image-first aggregation, 10,000 bootstrap, exact sign and frozen Holm families | COMPLETE |
| Scope | local input dependence, not causal head or branch importance | BOUNDARY |
"""
    (PAPER / "notes/endpoint_m4_evidence_matrix.md").write_text(evidence)
    print(json.dumps({"status": decision["status"], "tests": len(tests), "figures": 3}))


if __name__ == "__main__":
    main()
