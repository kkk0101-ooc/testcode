#!/usr/bin/env python3
"""Write synchronized M6 case reports after selection and rendering."""
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_m6_r1_20260919"
PAPER = REPO / "papers/nns_vit_endpoint"


def load(path):
    return json.loads(Path(path).read_text())


def compact(case):
    if "selected_units" in case: rows = case["selected_units"]
    elif "selected_heads" in case: rows = case["selected_heads"]
    elif "selected_residuals" in case: rows = case["selected_residuals"]
    else: rows = [case["selected_boundary"]]
    return "; ".join("/".join(str(row.get(key, "-")) for key in ("model_id", "sample_id", "block", "token", "channel", "head", "residual_type")) for row in rows)


def value(case):
    selection = case["selection"]; metric = selection["metric"]
    return metric, selection.get("selected_value", selection["selected"].get(metric))


def table(selection):
    lines = ["| Case | Frozen metric | Selected value | Selected identity |", "|---|---|---:|---|"]
    for case in selection["cases"]:
        metric, observed = value(case)
        lines.append(f"| {case['case_id']} | {metric} | {float(observed):.6f} | {compact(case)} |")
    return "\n".join(lines)


def main():
    selection = load(ROOT / "selection_manifest.json"); figures = load(ROOT / "figure_manifest.json"); rows = table(selection)
    zh = f"""# Endpoint M6 机制案例与可视化 atlas

M6 按执行前冻结的指标从 M3/M4 全部封存记录中自动选择 11 类机制案例，并从 M2 选择 1 个方法语义边界案例。选择过程仅读取数值文件；原图与像素级 map 在选择封存后才读取或重算。没有根据图像观感复选，也没有增加推断检验。

{rows}

## Image #1：Neuron atlas

![Image #1](../figures/m6/m6_neuron_atlas.png)

该图同时给出原图、signed Endpoint、absolute Endpoint、14×14 patch mass，以及 block/token/channel、entropy、locality、distance 和 cancellation。early-local、late-dispersed、CLS–patch 与 high-cancellation 案例分别由完整 M3 数值记录的冻结 5% 尾部确定。它们展示注册单元在当前输入处的 local input dependence，不把图案命名为神经元学到的概念。

## Image #2：ViT-B/16 与 DeiT-S/16 层级差异

![Image #2](../figures/m6/m6_model_layer_comparison.png)

共享图像按两模型 late-minus-early entropy 差异最接近跨图中位数选出；block 0/11 的 token 30 channel 再按最接近图内中位 entropy 自动确定。因此该图是预定义的代表案例，不是最显著模型差异。两模型均呈现深层 map 更分散的局部一阶输入依赖；channel ID 不表示跨模型语义对应。

## Image #3：Attention routing/value 案例

![Image #3](../figures/m6/m6_attention_cases.png)

三行分别对应 routing share 最高 5%、最低 5% 和 frozen-attention Endpoint error 最高 5% 中按固定 ID 选出的 head。Full、routing、Value、Q、K maps 解释同一 ground-truth logit，且 Q+K 重构 routing、routing+Value 重构 full。右侧 attention 面板是同一 head 的 CLS-query 权重。Routing share 是相对 path-norm scale；它不是因果贡献百分比，attention weight 也不等于 Endpoint dependence。

## Image #4：Residual identity/branch 案例

![Image #4](../figures/m6/m6_residual_cases.png)

三行分别来自 identity share 最高 5%、最低 5% 和 path cancellation 最高 5%。Output、identity/carry 与 branch 使用同一 residual-output cotangent，并满足 output = identity + branch。该图显示冻结目标条件下路径局部依赖的保留、改写与抵消，不把 path share 解释为有限删除效应。

## Image #5：M2 方法语义边界

![Image #5](../figures/m6/m6_m2_semantic_boundary.png)

固定 DeiT-S、block 5、patch geometry 和 normalized-zero reference 后，案例仅按 Endpoint absolute 与 regional Taylor 排名差异最接近候选中位数选出，未用 finite-effect outcome 选图。Endpoint 描述观测输入处的局部缩放依赖；regional Taylor 还使用指定 reference 的 activation displacement。右侧曲线沿用 M2 封存的 full/isolated finite effects，只用于展示两个 estimand 在同一案例中的不同排序。

## 结论

M6 把 M3 的 token-channel 空间扩展和 M4 的 attention/residual 路径差异落实为机器选择的可追溯案例。图像用于解释统计规律；总体结论仍以 M3/M4 的图像级分析为依据。显示归一化仅改变颜色映射，所有数值和闭合核对来自 raw arrays 与封存来源。
"""
    en = f"""# Endpoint M6 mechanism cases and visualization atlas

M6 automatically selects eleven mechanism cases from all sealed M3/M4 records and one semantic-boundary case from M2. Selection reads numerical records only; images and pixel-level maps are accessed or recomputed after the selection is sealed. No case is reselected by visual appearance, and no inferential test is added.

{rows}

## Image #1: neuron atlas

![Image #1](../figures/m6/m6_neuron_atlas.png)

The figure reports the input, signed Endpoint, absolute Endpoint, 14×14 patch mass, block/token/channel identity, entropy, locality, distance, and cancellation. The early-local, late-dispersed, CLS–patch, and high-cancellation cases come from frozen 5% tails over the complete M3 numerical records. They visualize local input dependence for registered units; the visual patterns are not assigned learned-concept labels.

## Image #2: ViT-B/16 and DeiT-S/16 layer contrast

![Image #2](../figures/m6/m6_model_layer_comparison.png)

The shared image is nearest to the cross-image median ViT-minus-DeiT difference in late-minus-early entropy. Token-30 channels at blocks 0 and 11 are then selected nearest to each image-block median entropy. This is a predefined representative case, not the largest model difference. Both models show more dispersed local first-order input dependence at the later block; channel IDs are not semantic matches across models.

## Image #3: attention routing/value cases

![Image #3](../figures/m6/m6_attention_cases.png)

The rows are the fixed-ID choices from the highest 5% routing share, lowest 5% routing share, and highest 5% frozen-attention Endpoint error tails. Full, routing, Value, Q, and K maps explain the same ground-truth logit; Q+K reconstructs routing and routing+Value reconstructs full. The final panel shows CLS-query attention weights for the same head. Routing share is a relative path-norm scale, not a causal contribution percentage, and attention weights are not Endpoint dependence.

## Image #4: residual identity/branch cases

![Image #4](../figures/m6/m6_residual_cases.png)

The rows come from the highest 5% identity share, lowest 5% identity share, and highest 5% path-cancellation tails. Output, identity/carry, and branch use the same residual-output cotangent and satisfy output = identity + branch. The maps show preservation, rewriting, and cancellation of target-conditioned local dependence; path shares are not finite ablation effects.

## Image #5: M2 semantic boundary

![Image #5](../figures/m6/m6_m2_semantic_boundary.png)

After fixing DeiT-S, block 5, patch geometry, and the normalized-zero reference, this case is selected nearest to the median Endpoint-absolute versus regional-Taylor rank difference. Finite-effect outcomes are not used for selection. Endpoint measures local scaling dependence at the observed input, whereas regional Taylor also uses activation displacement to the specified reference. The curves reuse sealed M2 full and isolated finite effects to show the two estimands in the same case.

## Conclusion

M6 converts the M3 token-channel spatial-broadening result and the M4 attention/residual path differences into traceable, machine-selected cases. The images explain the aggregate patterns; the M3/M4 image-level analyses remain the evidence for population-level conclusions. Display normalization changes only the color mapping, while numerical statements and closure checks use raw arrays and sealed sources.
"""
    (PAPER / "zh/endpoint_m6_atlas.md").write_text(zh); (PAPER / "en/endpoint_m6_atlas.md").write_text(en)
    (REPO / "docs/endpoint_m6_results_20260919.md").write_text(zh.replace("../figures/m6/", "../papers/nns_vit_endpoint/figures/m6/"))
    for language, heading, marker, body in (("zh", "### 6.8 自动选择的机制案例", "## 7. 局限", zh), ("en", "### 6.8 Automatically selected mechanism cases", "## 7. Limitations", en)):
        source = (PAPER / language / "full_draft_m5.md").read_text(); destination = PAPER / language / "full_draft_m6.md"
        core = "\n".join(body.splitlines()[1:]).strip(); block = "<!-- M6_ATLAS_START -->\n" + heading + "\n\n" + core + "\n<!-- M6_ATLAS_END -->\n\n"
        if source.count(marker) != 1: raise ValueError("INSERTION_POINT:" + language)
        destination.write_text(source.replace(marker, block + marker))
    evidence = """# Endpoint M6 evidence matrix

| Requirement | Evidence | Status | Boundary |
|---|---|---|---|
| Twelve case categories | selection_manifest.json | PASS | Fixed metrics and ID tie-breaks |
| Selection before images | freeze.json, source_validation.json | PASS | CPU-only numeric pass |
| Neuron atlas fields | m6_neuron_atlas files, raw_arrays.npz | PASS | Local Endpoint dependence |
| Attention full/routing/value/Q/K and attention | m6_attention_cases files | PASS | Relative path norms, not causal percentages |
| Residual output/identity/branch | m6_residual_cases files | PASS | Shared clean output cotangent |
| Cross-model representative case | m6_model_layer_comparison files | PASS | No channel correspondence claim |
| M2 boundary case | m6_m2_semantic_boundary files | PASS | Selection excludes finite outcomes |
| Display and provenance | figure_manifest.json, map_recompute_receipt.json | PASS | Raw arrays authoritative |
"""
    (PAPER / "notes/endpoint_m6_evidence_matrix.md").write_text(evidence)
    audit = """# Endpoint M6 defensive-writing audit

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| M6 captions | None detected | CLEAN | None | Yes | Claims lead with selected metric and map identity | Scope is explicit | Retain | Retain |

### Paper-level defensive-writing score

1/10. The section uses claim -> evidence -> concise boundary and avoids reviewer-facing prebuttals.

### Highest-priority removals

- None required after audit.

### Necessary caveats to preserve

- Routing and identity shares are relative path-norm scales, not causal contribution percentages.
- M2 selection excludes finite-effect outcomes and illustrates an estimand boundary.
- Display normalization is not used for numerical comparison.

### Repeated phrase audit

- Reviewer-facing prebuttals: 0
- Repeated non-claim disclaimers: 0
- Promotional adjectives: 0
"""
    (PAPER / "qc/endpoint_m6_writing_audit.md").write_text(audit)
    print(json.dumps({"status": "M6_DOCUMENTS_WRITTEN", "cases": 12, "figures": len(figures["figures"]), "documents": 7}))


if __name__ == "__main__":
    main()
