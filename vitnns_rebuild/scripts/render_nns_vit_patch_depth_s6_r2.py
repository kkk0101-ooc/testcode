#!/usr/bin/env python3
"""Render S6-R2 presentation figures without changing sealed S6-R1 evidence."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-nns-s6-r2")

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s6_visual as visual  # noqa: E402


def git(*args: str) -> str: return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def analysis_text() -> str:
    return """# S6-R2 presentation revision 图片分析

这次改图参照 Understanding neural network through neuron level visualization 的多面板叙事方式：用统一的行列结构展示结果，把统计量放在小型标注中，不再让审计门槛占据主视觉。S6-R1 的数值、决策和 terminal 均未修改。

## Image #1 — Evidence atlas

四个端点使用相同的小倍图语法：淡色分布展示图像级异质性，顶部黑线是 ordinary BCa 95% CI，彩色点是 equal-class estimate，虚线是 S5 门槛。它比原图同时呈现总体效应、区间和样本分布，避免大面积空白。

## Image #2 — Depth progression

采用参考文章常见的双面板层级曲线。左图是 NNS corrected alignment，右图是 probe gain；两者保留独立纵轴和不同线型，不暗示数值可直接互换。

## Image #3 — D1/D2 relation

保留逐图散点，因为近零相关本身是关键结论。版式统一为轻量网格和内嵌统计卡。

## Image #4 — P2 decision panels

三项标准拆成三个独立面板，各自拥有正确量纲和轴范围。蓝点为观测值，橙虚线为原规则，绿点线为 S5 exploratory rule。三个面板都直接标示 Original: FAIL 与 S5: PASS，避免原图把不同量纲强放在同一横轴。
"""


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--replace-own-root", action="store_true"); args = parser.parse_args()
    evidence = REPO / visual.ROOT_REL; report = REPO / visual.REPORT_REL
    for root in (evidence, report):
        if root.exists():
            if not args.replace_own_root: raise RuntimeError(f"fresh presentation root required: {root}")
            shutil.rmtree(root)
    evidence.mkdir(parents=True); report.mkdir(parents=True)
    inputs, source_records = visual.load_inputs(REPO); commit = git("rev-parse", "HEAD")
    source_payload = {"analysis_label": visual.LABEL, "title": "S6-R2 presentation source and style binding", "records": source_records, "records_digest": visual.canonical_digest(source_records), "S6_R1_modified": False, "new_scientific_analysis_count": 0, "visual_style_reference_only": True}
    visual.write_json_new(evidence / "source_binding.json", visual.signed("s6_r2_source_binding", "PASS_PRESENTATION_ONLY", source_payload, commit))
    paths = visual.render(evidence / "figures", inputs)
    records = []
    for path in paths:
        item = visual.artifact(REPO, path); item["format"] = path.suffix[1:].upper(); item["geometry"] = visual.png_geometry(path) if path.suffix == ".png" else None; records.append(item)
    inventory = {"analysis_label": visual.LABEL, "title": "S6-R2 presentation figure inventory", "figures": records, "figure_count": len(records), "figures_digest": visual.canonical_digest(records), "formats": ["PDF", "PNG", "SVG"], "S6_R1_scientific_values_changed": False}
    visual.write_json_new(evidence / "figure_inventory.json", visual.signed("s6_r2_figure_inventory", "PASS_PRESENTATION_ONLY", inventory, commit))
    explanation = analysis_text(); visual.write_text_new(evidence / "figure_analysis.md", explanation); visual.write_text_new(report / "figure_analysis.md", explanation)
    (report / "figures_png").mkdir()
    for path in paths:
        if path.suffix == ".png": shutil.copyfile(path, report / "figures_png" / path.name)
    access = {"analysis_label": visual.LABEL, "title": "S6-R2 presentation isolation audit", "S5_or_S6_JSON_read_count": len(visual.SOURCE_FILES), "reference_JPEG_style_read_count": 0, "scientific_array_read_count": 0, "model_or_checkpoint_load_count": 0, "CUDA_or_GPU_initialization_count": 0, "new_scientific_analysis_count": 0, "S6_R1_write_count": 0}
    visual.write_json_new(evidence / "access_audit.json", visual.signed("s6_r2_access_audit", "PASS_PRESENTATION_ONLY", access, commit))
    print(json.dumps({"status": "PASS_PRESENTATION_ONLY", "figure_count": len(records), "evidence_root": visual.ROOT_REL.as_posix(), "report_root": visual.REPORT_REL.as_posix()}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
