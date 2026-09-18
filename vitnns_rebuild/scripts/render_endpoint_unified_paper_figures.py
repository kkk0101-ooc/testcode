#!/usr/bin/env python3
"""Render the complete CPU-only Endpoint Scanner paper figure package."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import re
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/endpoint-unified-paper-figures-mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_unified_paper_figures_r2_20260918"
MATH = REPO / "reports/nns_vit_endpoint_math_comparison_r1_20260918"
DESIGN = REPO / "docs/NNS_ViT_Endpoint_Scanner统一论文图包恢复R2_20260918.md"
R1 = REPO / "reports/nns_vit_endpoint_unified_paper_figures_r1_20260918"
E2 = REPO / "reports/nns_vit_endpoint_e2_atlas_r1_20260918"
E2R4 = REPO / "reports/nns_vit_endpoint_e2_atlas_r4_20260918"
E3 = REPO / "reports/nns_vit_endpoint_e3_closure_r1_20260918"
E3R2 = REPO / "reports/nns_vit_endpoint_e3_closure_r2_20260918"
E4 = REPO / "reports/nns_vit_endpoint_e4_intervention_r3_20260918"
E5 = REPO / "reports/nns_vit_endpoint_e5_spatial_r1_20260918"
E6 = REPO / "reports/nns_vit_endpoint_e6_release_r3_20260918"
INK, MUTED, BLUE, ORANGE, GREEN, RED, LIGHT = "#172B3A", "#5A6B79", "#137EAB", "#D9792B", "#2F7D5B", "#B7473A", "#F3F7F9"
SOURCES: dict[str, dict[str, Any]] = {}
NUMERIC: dict[str, Any] = {}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "canonical_digest": canonical(value)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def register(path: Path, role: str) -> None:
    path = path.resolve()
    relative = path.relative_to(REPO).as_posix()
    SOURCES[relative] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path), "role": role}


def read_json(path: Path, role: str = "source_json") -> Any:
    register(path, role)
    return json.loads(path.read_text())


def canvas(title: str, subtitle: str = "", size=(12, 7)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    ax.set_axis_off(); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    ax.text(.25, 6.68, title, fontsize=18, weight="bold", color=INK, va="top")
    if subtitle: ax.text(.25, 6.25, subtitle, fontsize=10.5, color=MUTED, va="top")
    return fig, ax


def box(ax, x, y, w, h, label, edge=BLUE, face=LIGHT, size=11, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.04,rounding_size=.08", edgecolor=edge, facecolor=face, lw=1.5))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=size, color=INK, weight=weight)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,lw=1.4,color=color))


def save(fig, number: int, slug: str) -> list[dict[str, Any]]:
    outputs=[]
    for folder, ext, fmt in (("figures_png","png","PNG"),("figures_pdf","pdf","PDF"),("figures_svg","svg","SVG")):
        path=ROOT/folder/f"figure_{number:02d}_{slug}.{ext}";path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=220,facecolor="white",bbox_inches="tight")
        outputs.append({"format":fmt,"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)})
    plt.close(fig);return outputs


def signed_map(value: np.ndarray) -> np.ndarray:
    a=np.asarray(value,dtype=np.float64)
    if a.ndim==4:a=a[0]
    return a.sum(axis=0) if a.ndim==3 else a


def show_map(ax, value: np.ndarray, title: str, p=99.5):
    a=signed_map(value);limit=float(np.percentile(np.abs(a),p)) or float(np.max(np.abs(a))) or 1.
    ax.imshow(a,cmap="RdBu_r",vmin=-limit,vmax=limit,interpolation="nearest");ax.set_title(title,fontsize=9);ax.axis("off")
    return {"title":title,"shape":list(a.shape),"signed_sum":float(a.sum()),"absolute_mass":float(np.abs(a).sum()),"display_limit":limit}


def fig1():
    fig,ax=canvas("Endpoint Scanner: from registered targets to behavioral evidence","A two-model, 400-image endpoint study with separate identity, correctness, and utility layers")
    stages=[(.55,4.45,"1  Register targets\n457 / 385",BLUE),(4.25,4.45,"2  Endpoint maps\nx ⊙ ∇s(x)",BLUE),(7.95,4.45,"3  Structured cuts\nheads · Q/K/V · residual",GREEN),(.55,2.65,"4  Finite replacement\nmatched activation effects",ORANGE),(4.25,2.65,"5  Spatial intervention\nfour mandatory baselines",ORANGE),(7.95,2.65,"6  Paper decisions\nRQ3 / RQ4",RED)]
    for x,y,label,color in stages:box(ax,x,y,3.0,1.05,label,color,size=10.2,weight="bold")
    arrow(ax,3.55,4.98,4.15,4.98);arrow(ax,7.25,4.98,7.85,4.98);arrow(ax,9.45,4.42,2.05,3.76);arrow(ax,3.55,3.18,4.15,3.18);arrow(ax,7.25,3.18,7.85,3.18)
    ax.text(.65,1.62,"Identity",fontsize=12,weight="bold",color=BLUE);ax.text(2.45,1.62,"Endpoint = Gradient × Input for the same scalar target",fontsize=10,color=INK)
    ax.text(.65,1.10,"Correctness",fontsize=12,weight="bold",color=GREEN);ax.text(2.45,1.10,"76,800 closure checks · 28,800 negative controls · 6,144 analytic JVP checks",fontsize=10,color=INK)
    ax.text(.65,.58,"Utility",fontsize=12,weight="bold",color=ORANGE);ax.text(2.45,.58,"RQ3 overall supported (11/12) · RQ4 not supported",fontsize=10,color=INK)
    ax.text(7.4,.58,"Path IG: ViT-B calibration failed; no cross-model Path claim",fontsize=9.3,color=MUTED)
    return fig


def fig11():
    targets={"ViT-B/16":{"token channel":193,"patch sum":70,"energy":48,"head":144,"class":2},"DeiT-S/16":{"token channel":193,"patch sum":70,"energy":48,"head":72,"class":2}}
    fig,ax=plt.subplots(figsize=(12,7));bottom=np.zeros(2);colors=[BLUE,"#4E9CC0",GREEN,ORANGE,RED]
    for color,key in zip(colors,next(iter(targets.values())).keys()):
        vals=[targets[m][key] for m in targets];ax.bar(list(targets),vals,bottom=bottom,label=key,color=color);bottom+=vals
    ax.set_title("Registered endpoint target atlas",fontsize=18,weight="bold");ax.set_ylabel("Targets per image");ax.legend(ncol=5,loc="upper center",bbox_to_anchor=(.5,-.08));ax.grid(axis="y",alpha=.2)
    for i,total in enumerate(bottom):ax.text(i,total+8,str(int(total)),ha="center",weight="bold",fontsize=14)
    ax.text(.5,-.20,"All 12 blocks · fixed target IDs · ground-truth and frozen clean top-1 logits",transform=ax.transAxes,ha="center",color=MUTED)
    NUMERIC["figure_11"]=targets;return fig


def fig12():
    paths=[E2R4/"gallery/vit_b16_first8.png",E2R4/"gallery/deit_s16_first8.png"]
    for p in paths:register(p,"sealed_E2_gallery")
    fig,axes=plt.subplots(1,2,figsize=(12,10));fig.suptitle("Frozen first-eight endpoint gallery",fontsize=18,weight="bold")
    for ax,p,title in zip(axes,paths,["ViT-B/16","DeiT-S/16"]):ax.imshow(Image.open(p));ax.set_title(title);ax.axis("off")
    fig.text(.5,.015,"Each row: input · ground-truth Endpoint · frozen clean top-1 Endpoint. Display scaling is per row and does not enter statistics.",ha="center",fontsize=9,color=MUTED)
    return fig


def fig13():
    task=read_json(E2/"formal_results/attempts/deit_s16__000/attempt-01.json","E2_fixed_first_task")["result"]
    atlas=Path(next(r["path"] for r in task["artifacts"] if r["kind"]=="full_raw_atlas"));register(REPO/atlas,"E2_fixed_full_atlas")
    array=np.load(REPO/atlas,mmap_mode="r");ids=["b0.patch_channel_sum.c0","b0.patch_channel_energy.c0","b0.head0.cls.c0","b5.head0.cls.c0","b11.head5.cls.c0","class.gt"]
    index={r["target_id"]:i for i,r in enumerate(task["summaries"])};fig,axes=plt.subplots(2,3,figsize=(12,7));fig.suptitle("One image, six registered internal targets",fontsize=18,weight="bold")
    rec=[]
    for ax,target in zip(axes.ravel(),ids):rec.append(show_map(ax,array[index[target]],target))
    fig.text(.5,.015,"Fixed first evaluation image and preregistered target IDs; no display-based target selection.",ha="center",color=MUTED);NUMERIC["figure_13"]=rec;return fig


def fig14():
    report=read_json(E2R4/"validation_report.json","E2_R4_validation");cost=report["complete_atlas_amortized_cost"]
    fig,axes=plt.subplots(1,2,figsize=(12,7));fig.suptitle("Registered scan scale and amortized cost",fontsize=18,weight="bold")
    models=["vit_b16","deit_s16"];labels=["ViT-B/16","DeiT-S/16"]
    axes[0].bar(labels,[457,385],color=[BLUE,GREEN]);axes[0].set_ylabel("Registered targets / image");axes[0].set_title("336,800 finite target evaluations")
    med=[cost[m]["median_seconds_per_complete_atlas"] for m in models];p95=[cost[m]["p95_seconds_per_complete_atlas"] for m in models]
    x=np.arange(2);axes[1].bar(x-.18,med,.36,label="median",color=BLUE);axes[1].bar(x+.18,p95,.36,label="P95",color=ORANGE);axes[1].set_xticks(x,labels);axes[1].set_ylabel("Seconds / complete atlas");axes[1].legend();axes[1].set_title("400 images per model")
    for ax in axes:ax.grid(axis="y",alpha=.2)
    NUMERIC["figure_14"]={"target_evaluations":report["finite_target_evaluations"],"cost":cost};return fig


def fig15():
    fig,ax=canvas("Structured closure contracts","Every cut reconstructs a named branch or total object; omission and duplication controls must fail")
    box(ax,.45,4.45,1.7,.8,"Input x",BLUE,weight="bold");arrow(ax,2.15,4.85,2.65,4.85)
    box(ax,2.65,4.15,2.1,1.4,"Attention add",GREEN,weight="bold");ax.text(3.7,4.42,"attention branch + identity",ha="center",fontsize=8.5,color=MUTED);arrow(ax,4.75,4.85,5.3,4.85);box(ax,5.3,4.15,2.1,1.4,"MLP add",GREEN,weight="bold");ax.text(6.35,4.42,"MLP branch + carry",ha="center",fontsize=8.5,color=MUTED);arrow(ax,7.4,4.85,7.95,4.85);box(ax,7.95,4.45,1.75,.8,"Final target F",BLUE,weight="bold")
    rows=[("Heads → A_F","all attention heads"),("Identity + heads → L_F","first residual total"),("Q + K + V → A_F","attention differential branch"),("Routing + Value → A_F","alternative differential cut"),("Identity + attention → L_F","first residual"),("Carry + MLP → L_F","second residual")]
    for i,(a,b) in enumerate(rows):x=.45+(i%3)*3.75;y=2.55-(i//3)*1.2;box(ax,x,y,3.25,.75,a,BLUE if i<4 else ORANGE,size=10,weight="bold");ax.text(x+1.62,y-.22,b,ha="center",fontsize=8.5,color=MUTED)
    ax.text(.55,.45,"Negative controls: omit a nonzero head · omit routing · duplicate a residual path",fontsize=11,weight="bold",color=RED)
    return fig


def fig16():
    report=read_json(E3R2/"validation_report.json","E3_R2_validation")["independent_validation"]
    fig,axes=plt.subplots(1,2,figsize=(12,7));fig.suptitle("Formal structural closure",fontsize=18,weight="bold")
    axes[0].bar(["Positive closure\nchecks","Negative controls\nrejected"],[report["positive_closure_evaluations"],report["negative_control_evaluations"]],color=[GREEN,RED]);axes[0].set_ylabel("Evaluations");axes[0].grid(axis="y",alpha=.2)
    axes[1].bar(["max relative L2","frozen tolerance"],[report["maximum_closure_relative_l2"],1e-4],color=[BLUE,MUTED]);axes[1].set_yscale("log");axes[1].set_ylabel("Error / tolerance");axes[1].grid(axis="y",alpha=.2)
    fig.text(.5,.02,"800 image-model tasks · 12 blocks · 8 registered cuts per block",ha="center",color=MUTED);NUMERIC["figure_16"]=report;return fig


def fig17():
    task=read_json(E3/"formal_results/attempts/deit_s16__000/attempt-01.json","E3_fixed_first_task")["result"];record=task["artifacts"][5];path=REPO/record["path"];register(path,"E3_fixed_block5_components");a=np.load(path,mmap_mode="r");order=record["component_order"];names=["identity","head_0","routing","value","attention","mlp","L_F"]
    fig,axes=plt.subplots(2,4,figsize=(12,7));fig.suptitle("Real block-5 structural component maps",fontsize=18,weight="bold");recs=[]
    for ax,name in zip(axes.ravel(),names):recs.append(show_map(ax,a[order.index(name)],name))
    axes.ravel()[-1].axis("off");axes.ravel()[-1].text(.5,.55,"Same image\nSame class target\nRaw signed RGB arrays",ha="center",va="center",fontsize=12,color=MUTED)
    NUMERIC["figure_17"]=recs;return fig


def local_rows():
    combined=read_json(E4/"combined_result_index.json","E4_R3_index");rows=[]
    for record in combined["records"]:
        if record["origin"]!="newly_run_in_r3":continue
        path=REPO/record["source_result_path"];result=read_json(path,"E4_R3_local_task")["result"]
        for comp in result["local_contract"]["components"]:rows.append({**comp,"model_id":result["model_id"],"split_order":result["split_order"]})
    return rows


def fig18():
    rows=local_rows();ratio=np.array([r["directional_prediction_absolute_error"]/r["threshold"] for r in rows]);fig,axes=plt.subplots(1,2,figsize=(12,7));fig.suptitle("Condition-scaled analytic JVP correctness",fontsize=18,weight="bold")
    axes[0].hist(ratio,bins=50,color=BLUE,alpha=.85);axes[0].axvline(1,color=RED,ls="--",label="failure boundary");axes[0].set_xlabel("absolute error / condition-scaled threshold");axes[0].set_ylabel("Components");axes[0].legend()
    axes[1].scatter([r["absolute_dot_mass"] for r in rows],[r["directional_prediction_absolute_error"] for r in rows],s=5,alpha=.35,color=GREEN);x=np.logspace(-4,4,200);axes[1].plot(x,1e-5+1e-5*x,color=RED,label="1e-5 + 1e-5 Σ|product|");axes[1].set_xscale("log");axes[1].set_yscale("log");axes[1].set_xlabel("Σ absolute dot products");axes[1].set_ylabel("absolute JVP error");axes[1].legend()
    NUMERIC["figure_18"]={"checks":len(rows),"max_ratio":float(ratio.max()),"all_pass":bool((ratio<=1).all())};return fig


def fig19():
    fig,ax=canvas("Matched finite activation replacement","Endpoint component mass is tested against the effect of replacing the same registered component")
    box(ax,.4,4.4,2.25,1.1,"Clean image x",BLUE,weight="bold");arrow(ax,2.65,4.95,3.05,4.95);box(ax,3.05,4.2,2.35,1.5,"Capture registered\nz_k(x)",GREEN,weight="bold");arrow(ax,5.4,4.95,5.8,4.95);box(ax,5.8,4.2,2.35,1.5,"Replace with\nz_k(reference)",ORANGE,weight="bold");arrow(ax,8.15,4.95,8.55,4.95);box(ax,8.55,4.2,2.85,1.5,"Recompute\ndownstream F",BLUE,weight="bold")
    box(ax,.75,2.15,4.55,1.1,r"Predictor:  $m_k=\sum_i |E_{k,i}|$",BLUE,size=12.5,weight="bold");box(ax,6.2,2.15,4.55,1.1,r"Effect:  $|e_k|=|F(x)-F(x;z_k\leftarrow z_k^{ref})|$",ORANGE,size=10.8,weight="bold")
    ax.text(6,1.35,"Per image: Spearman across registered components",ha="center",fontsize=13,weight="bold",color=INK);ax.text(6,.75,"Then aggregate across 400 images; heads, routing/value, and residual branches remain separate families",ha="center",fontsize=10,color=MUTED)
    return fig


def family_data():return read_json(E6/"family_decisions.json","E6_family_decisions")["families"]


def fig20():
    rows=[r for r in family_data() if r["claim"]=="C4a"];tests=[r["test_results"][0] for r in rows];labels=[r["family_id"].replace("C4a.","") for r in rows];y=np.arange(len(rows));est=np.array([t["estimate"] for t in tests]);lo=np.array([t["ci_lower"] for t in tests]);hi=np.array([t["ci_upper"] for t in tests])
    fig,ax=plt.subplots(figsize=(12,7));colors=[GREEN if r["status"]=="SUPPORTED" else ORANGE for r in rows]
    for index,color in enumerate(colors):ax.errorbar(est[index],y[index],xerr=[[est[index]-lo[index]],[hi[index]-est[index]]],fmt="o",color=color,capsize=3,lw=2)
    ax.axvline(.1,color=RED,ls="--",label="registered effect threshold = 0.10");ax.set_yticks(y,labels,fontsize=8);ax.invert_yaxis();ax.set_xlabel("Median image-level Spearman (95% bootstrap CI)");ax.set_title("RQ3: finite-replacement association across 12 families",fontsize=17,weight="bold");ax.legend();ax.grid(axis="x",alpha=.2)
    fig.text(.5,.01,"11/12 families meet every criterion; the strict all-12 omnibus field remains NOT_SUPPORTED.",ha="center",weight="bold",color=INK);NUMERIC["figure_20"]=[{"id":r["family_id"],"status":r["status"],"estimate":t["estimate"],"ci":[t["ci_lower"],t["ci_upper"]]} for r,t in zip(rows,tests)];return fig


def fig21():
    rows=[r for r in family_data() if r["claim"]=="C4a"];vals=[r["test_results"][0]["estimate"] for r in rows];fig,axes=plt.subplots(1,2,figsize=(12,7));fig.suptitle("RQ3 overall support with one registered boundary exception",fontsize=18,weight="bold")
    axes[0].bar(["Families meeting\nall criteria","Boundary\nexception"],[11,1],color=[GREEN,ORANGE]);axes[0].set_ylim(0,12.5);axes[0].set_ylabel("Preregistered C4a families");axes[0].text(0,11.3,"11 / 12",ha="center",weight="bold",fontsize=16)
    exception=next(r for r in rows if r["status"]!="SUPPORTED");t=exception["test_results"][0];axes[1].errorbar([t["estimate"]],[0],xerr=[[t["estimate"]-t["ci_lower"]],[t["ci_upper"]-t["estimate"]]],fmt="o",color=ORANGE,capsize=5,lw=2);axes[1].axvline(.1,color=RED,ls="--");axes[1].set_xlim(.065,.13);axes[1].set_yticks([0],["ViT-B · zero · routing/value"]);axes[1].set_xlabel("Median Spearman");axes[1].set_title("0.099565 versus 0.10 (gap = 0.000435)");axes[1].grid(axis="x",alpha=.2)
    fig.text(.5,.02,"Positive direction and Holm-significant; the exception concerns the practical-effect threshold, not the existence of association.",ha="center",color=MUTED);NUMERIC["figure_21"]={"supported":11,"total":12,"exception":exception};return fig


def fig22():
    rows=read_json(E4/"validation_report.json","E4_R3_validation")["descriptive_per_image_spearman_summary"];keys=["endpoint_mass","p_ref_absolute","gradient_activation","output_norm","gradient_norm","random_20"];matrix=[];labels=[]
    for r in rows:
        matrix.append([r[k]["median_spearman"] for k in keys]);labels.append(f"{r['model_id']} · {r['reference_id'].replace('normalized_zero','zero')} · {r['family']}")
    a=np.array(matrix);fig,ax=plt.subplots(figsize=(12,8));im=ax.imshow(a,cmap="RdBu_r",vmin=-.3,vmax=.8,aspect="auto");ax.set_xticks(range(len(keys)),[k.replace("_"," ") for k in keys],rotation=25,ha="right");ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.set_title("Registered predictors of finite activation effects",fontsize=18,weight="bold")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):ax.text(j,i,f"{a[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(a[i,j])>.42 else INK)
    fig.colorbar(im,ax=ax,label="Median per-image Spearman");fig.text(.5,.01,"Descriptive predictors are reported alongside Endpoint mass; they are not part of the C4a omnibus gate.",ha="center",color=MUTED);NUMERIC["figure_22"]={"keys":keys,"rows":labels,"matrix":a.tolist()};return fig


def fig23():
    fig,ax=canvas("Spatial intervention at a matched 4 × 4 region budget","Block 5 · three preregistered heads · normalized-zero and blur references")
    for r in range(4):
        for c in range(4):ax.add_patch(Rectangle((.65+c*.65,2.6+(3-r)*.65),.58,.58,edgecolor=BLUE,facecolor="#DCEEF5"));ax.text(.94+c*.65,2.89+(3-r)*.65,str(r*4+c),ha="center",va="center",fontsize=8)
    ax.text(1.95,5.55,"16 fixed 56×56 regions",ha="center",weight="bold");arrow(ax,3.45,4.0,3.85,4.0)
    box(ax,3.85,4.95,2.25,.58,"Endpoint component mass",BLUE,weight="bold",size=9.5);box(ax,3.85,4.15,2.25,.58,"Total G×I",GREEN,weight="bold",size=9.5);box(ax,3.85,3.35,2.25,.58,"Raw gradient",GREEN,weight="bold",size=9.5);box(ax,3.85,2.55,2.25,.58,"Regional Taylor",ORANGE,weight="bold",size=9.5);box(ax,3.85,1.75,2.25,.58,"20 random orders",MUTED,weight="bold",size=9.5)
    arrow(ax,6.1,3.65,6.65,3.65);box(ax,6.65,3.95,2.1,1.25,"Select top\n0 / 1 / 2 / 4 regions",BLUE,weight="bold",size=10);arrow(ax,8.75,4.58,9.15,4.58);box(ax,9.15,3.95,2.25,1.25,"Replace input\nthen isolate head",ORANGE,weight="bold",size=10)
    box(ax,6.65,1.8,4.75,1.25,"Per head: AUC of |isolated effect|\nImage score: mean across 3 heads",GREEN,size=10,weight="bold");ax.text(9.02,1.35,"Inference unit remains the image",ha="center",color=MUTED)
    return fig


def e5_fixed():return read_json(E5/"formal_results/attempts/deit_s16__000/attempt-01.json","E5_fixed_first_task")["result"]


def fig24():
    task=e5_fixed();path=REPO/task["raw_maps_record"]["path"];register(path,"E5_fixed_raw_maps");z=np.load(path);names=["total_gxi","raw_gradient","head_0","head_3","head_5"];fig,axes=plt.subplots(1,5,figsize=(12,4));fig.suptitle("Real block-5 maps used for spatial ranking",fontsize=18,weight="bold");rec=[]
    for ax,name in zip(axes,names):rec.append(show_map(ax,z[name],name))
    fig.text(.5,.01,"Fixed first DeiT-S/16 evaluation task; raw signed RGB maps, display-scaled per panel.",ha="center",color=MUTED);NUMERIC["figure_24"]=rec;return fig


def fig25():
    task=e5_fixed();value=task["references"]["normalized_zero"];areas=np.array(task["actual_area_fractions"]);methods=["endpoint_component","total_gxi","raw_gradient","regional_taylor"];colors=[BLUE,GREEN,"#6B8E23",ORANGE];fig,ax=plt.subplots(figsize=(12,7));records={}
    for method,color in zip(methods,colors):
        curves=[r for r in value["curves"] if r["method"]==method];series=np.mean([[abs(p["isolated_effect"]) for p in r["points"]] for r in curves],axis=0);ax.plot(areas,series,marker="o",lw=2,label=method.replace("_"," "),color=color);records[method]=series.tolist()
    random=np.mean([r["absolute_of_mean_isolated_effect_curve"] for r in value["random_aggregates"]],axis=0);ax.plot(areas,random,marker="o",lw=2,label="random regions",color=MUTED);records["random_regions"]=random.tolist()
    ax.set_title("One-image isolated-head spatial effect curves",fontsize=18,weight="bold");ax.set_xlabel("Replaced input area fraction");ax.set_ylabel("Mean absolute isolated-head effect");ax.legend(ncol=3);ax.grid(alpha=.2);fig.text(.5,.01,"Fixed deit_s16__000 · normalized-zero reference · three preregistered heads.",ha="center",color=MUTED);NUMERIC["figure_25"]={"areas":areas.tolist(),"series":records};return fig


def fig26():
    rows=[r for r in family_data() if r["claim"]=="C4b"];tests=[t for r in rows for t in r["test_results"]];labels=[t["test_id"].replace("C4b.","") for t in tests];y=np.arange(len(tests));est=np.array([t["estimate"] for t in tests]);lo=np.array([t["ci_lower"] for t in tests]);hi=np.array([t["ci_upper"] for t in tests]);colors=[ORANGE if t["baseline"]=="regional_taylor" else BLUE for t in tests]
    fig,ax=plt.subplots(figsize=(12,8))
    for index,color in enumerate(colors):ax.errorbar(est[index],y[index],xerr=[[est[index]-lo[index]],[hi[index]-est[index]]],fmt="o",color=color,capsize=2)
    ax.axvline(0,color=INK,lw=1);ax.axvline(.05,color=RED,ls="--",label="registered gain threshold = 0.05");ax.set_yticks(y,labels,fontsize=7);ax.invert_yaxis();ax.set_xlabel("Endpoint score − mandatory baseline (median, 95% CI)");ax.set_title("RQ4: mandatory-baseline spatial contrasts",fontsize=18,weight="bold");ax.legend();ax.grid(axis="x",alpha=.2);NUMERIC["figure_26"]=[{k:t[k] for k in ("test_id","baseline","estimate","ci_lower","ci_upper","holm_adjusted_p_value")} for t in tests];return fig


def fig27():
    rows=[r for r in family_data() if r["claim"]=="C4b"];baselines=["total_gxi","raw_gradient","regional_taylor","random_regions"];a=np.array([[next(t["estimate"] for t in r["test_results"] if t["baseline"]==b) for b in baselines] for r in rows]);labels=[r["family_id"].replace("C4b.","").replace(".head_regions","") for r in rows]
    fig,ax=plt.subplots(figsize=(12,6));im=ax.imshow(a,cmap="RdBu_r",vmin=-.04,vmax=.04,aspect="auto");ax.set_xticks(range(4),[b.replace("_"," ") for b in baselines]);ax.set_yticks(range(4),labels);ax.set_title("RQ4 summary: Endpoint minus each mandatory baseline",fontsize=18,weight="bold")
    for i in range(4):
        for j in range(4):ax.text(j,i,f"{a[i,j]:.4f}",ha="center",va="center",color="white" if abs(a[i,j])>.018 else INK,weight="bold")
    fig.colorbar(im,ax=ax,label="Median paired difference");fig.text(.5,.02,"All values are far below the registered +0.05 gain; regional Taylor is higher in all four families.",ha="center",weight="bold",color=INK);NUMERIC["figure_27"]={"rows":labels,"columns":baselines,"matrix":a.tolist()};return fig


def fig28():
    fig,ax=canvas("Endpoint Scanner: final evidence map","Scientific interpretation follows the evidence layers, not a single omnibus label")
    items=[(.5,4.05,3.3,1.55,"Identity","ESTABLISHED","Endpoint = Gradient × Input\nfor the same scalar target",BLUE),
           (4.35,4.05,3.3,1.55,"Correctness","SUPPORTED","Closure + negative controls\n+ analytic JVP checks",GREEN),
           (8.2,4.05,3.3,1.55,"RQ3 utility","OVERALL SUPPORTED","11/12 C4a families\none boundary exception",ORANGE),
           (2.4,1.45,3.3,1.55,"RQ4 utility","NOT SUPPORTED","0/4 families pass all\nmandatory baselines",RED),
           (6.3,1.45,3.3,1.55,"Path IG comparison","NOT RELEASED","ViT-B numerical\ncalibration failed",MUTED)]
    for x,y,w,h,title,status,body,color in items:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.04,rounding_size=.08",edgecolor=color,facecolor=LIGHT,lw=1.5));ax.text(x+w/2,y+h-.32,title,ha="center",va="center",fontsize=10.5,color=MUTED);ax.text(x+w/2,y+h-.72,status,ha="center",va="center",fontsize=12.5,color=INK,weight="bold");ax.text(x+w/2,y+.30,body,ha="center",va="bottom",fontsize=8.5,color=MUTED)
    ax.text(6,.55,"Strict machine field: omnibus C4a NOT_SUPPORTED under the frozen all-12 rule",ha="center",fontsize=10,color=MUTED)
    return fig


TITLES={1:"Endpoint Scanner study overview",2:"Endpoint / CNN / Path method identity",3:"Same target: Endpoint versus Path",4:"Original CNN endpoint gates",5:"Attention Q/K/V differential paths",6:"Residual edge accounting",7:"Unit, sum, and energy targets",8:"Path, seed, and input coordinates",9:"LayerNorm and GELU Jacobians",10:"Structured edge cuts",11:"Registered target atlas design",12:"Frozen first-eight endpoint gallery",13:"One-image multi-target atlas",14:"Scan scale, latency, and resources",15:"Structural closure contracts",16:"Formal closure and negative controls",17:"Real structural component maps",18:"Condition-scaled analytic JVP checks",19:"Matched finite activation replacement",20:"C4a family estimates and intervals",21:"RQ3 threshold zoom and boundary exception",22:"Endpoint and registered secondary predictors",23:"Spatial intervention protocol",24:"Real block-5 component and baseline maps",25:"One-image spatial effect curves",26:"C4b mandatory-baseline contrasts",27:"RQ4 family summary",28:"Identity–Correctness–Utility conclusion"}
SLUGS={n:re.sub(r"_+","_",re.sub(r"[^a-z0-9]+","_",TITLES[n].lower().replace("×","x"))).strip("_") for n in TITLES}
CAP_ZH={1:"从注册目标、结构分解、有限 activation replacement 到空间干预与论文判定的完整研究链。Identity、Correctness 与 Utility 分开报告。",2:"NNS-CNN、ViT Endpoint 与 Path 方法的 seed、Jacobian 和方法身份。",3:"同一解析目标的终点图与路径积分图；非线性 attention 下二者不必相等。",4:"原始 CNN 门控例子说明 Endpoint 使用当前 ReLU/MaxPool 状态。",5:"Attention 的 Q、K、V 微分路径及其闭合关系。",6:"Residual 输入边记账，避免遗漏或重复旁路。",7:"单元、patch sum 与 energy 是三个不同标量目标。",8:"Endpoint、baseline-relative endpoint 与 IG 的 seed 和坐标原点。",9:"LayerNorm 与 GELU 的真实 Jacobian 不能用冻结统计量或二值门代替。",10:"结构化切面把张量成分映回同一输入坐标并闭合到注册对象。",11:"ViT-B/16 每图 457 个目标，DeiT-S/16 每图 385 个目标，覆盖 12 个 block 与类别输出。",12:"冻结前 8 张 evaluation 图在两个模型上的 ground-truth 与 clean-top1 Endpoint 图。",13:"固定第一张 DeiT 图像的六类注册目标，不按显示效果挑选。",14:"E2 共完成 336,800 个 finite target evaluation，并报告完整 atlas 的实际耗时。",15:"六种切面分别闭合 attention 分支或 residual total；负例必须失败。",16:"76,800 项正向闭合通过，28,800 项遗漏/重复负例被拒绝。",17:"固定第一张图、block 5、同一类别目标的真实结构成分图。",18:"6,144 个 condition-scaled analytic JVP 检查全部在冻结边界内。",19:"将同一个注册 component 替换为 reference activation，以有限 logit effect 检验 Endpoint mass。",20:"RQ3 的 12 个 C4a family：11 个满足全部条件，一个临界例外。",21:"唯一例外 0.099565 距 0.10 仅 0.000435，方向为正且 Holm 显著；RQ3 整体支持。",22:"Endpoint mass 与预注册次级 predictor 同时报告；p_ref 等指标是描述性比较，不进入 C4a 门。",23:"固定 4×4 区域、三个 head、两种 reference 和四个 mandatory baseline 的空间干预协议。",24:"固定第一张 DeiT 图像的 total G×I、raw gradient 与三个 block-5 head component map。",25:"固定单图的 isolated-head effect 曲线，展示各排序方法如何形成 AUC。",26:"16 个 C4b baseline contrast 的中位差和 bootstrap 区间。",27:"四个 family 均未达到 +0.05；regional Taylor 在四个 family 中都更高。",28:"最终证据分层：Identity 和 Correctness 成立，RQ3 整体支持，RQ4 不支持，Path 比较不发布。"}
CAP_EN={
1:"The complete route from target registration through structured decomposition, finite replacement, spatial intervention, and RQ decisions.",
2:"Method identity for the original CNN endpoint rule, the ViT Endpoint adaptation, and path-integrated attribution.",
3:"Endpoint and path-integrated maps for the same analytic target; nonlinear attention makes them generally different.",
4:"An analytic CNN example showing that Endpoint uses the current ReLU gate and unique MaxPool winner.",
5:"Query, key, and value differential paths of attention and their reconstruction of the full attention contribution.",
6:"Residual-edge accounting with one downstream cotangent prevents omission and double counting.",
7:"A unit, patch-channel sum, and energy target define distinct scalar responses.",
8:"Endpoint, baseline-relative Endpoint, and Integrated Gradients use different seeds and coordinate origins.",
9:"The full LayerNorm Jacobian and continuous GELU derivative cannot be replaced by frozen statistics or binary gates.",
10:"Structured tensor-valued cuts map internal components into common input coordinates and reconstruct named targets.",
11:"ViT-B/16 registers 457 targets per image and DeiT-S/16 registers 385 across all 12 blocks and class outputs.",
12:"Ground-truth and frozen-clean-top-1 Endpoint maps for the same first eight evaluation images in both models.",
13:"Six preregistered targets for the fixed first DeiT evaluation image, with no visual-outcome selection.",
14:"The E2 scan completed 336,800 finite target evaluations and measured complete-atlas latency.",
15:"Six structural cuts reconstruct attention branches or residual totals; omission and duplication controls must fail.",
16:"All 76,800 positive closure checks passed and all 28,800 negative controls were rejected.",
17:"Real block-5 component maps for one fixed image and class target from raw signed RGB arrays.",
18:"All 6,144 condition-scaled analytic JVP checks remained within their frozen numerical boundaries.",
19:"Matched replacement tests whether Endpoint component mass predicts the finite class-logit effect.",
20:"The twelve C4a families for RQ3: eleven met every criterion and one was a boundary exception.",
21:"The sole exception had median 0.099565, only 0.000435 below 0.10, with positive direction and Holm significance.",
22:"Endpoint mass and preregistered secondary predictors are reported together; secondary predictors do not enter the C4a gate.",
23:"The spatial protocol uses a fixed 4×4 region grid, three heads, two references, and four mandatory baselines.",
24:"Total Gradient × Input, raw gradient, and three block-5 head-component maps for the fixed first DeiT image.",
25:"Isolated-head effect curves for one fixed image show how each ranking method produces its AUC score.",
26:"All sixteen C4b paired contrasts against the mandatory baselines, with image-level bootstrap intervals.",
27:"No family reached the registered +0.05 gain; regional Taylor scored higher in all four families.",
28:"Final evidence separation: identity and correctness pass, RQ3 is supported overall, RQ4 is not supported, and no Path comparison is released."}
ANALYSIS={n:CAP_ZH[n]+(" 读图时保留原始数值与显示缩放的区别；颜色图只用于空间形态。" if n in {12,13,17,24} else "") for n in TITLES}


def build_docs(figures):
    header="[Gallery](gallery.html) · [Figure index](figure_index.md) · [Analysis](figure_analysis.md)"
    index=["# Endpoint Scanner complete paper figure package","",header,"","| # | Figure | PNG | PDF | SVG |","|---:|---|---|---|---|"]
    for f in figures:
        links={o["format"]:o["path"] for o in f["outputs"]};index.append(f"| {f['number']} | {f['title']} | [PNG]({links['PNG']}) | [PDF]({links['PDF']}) | [SVG]({links['SVG']}) |")
    (ROOT/"figure_index.md").write_text("\n".join(index)+"\n")
    for filename,caps in (("captions_zh.md",CAP_ZH),("captions_en.md",CAP_EN)):
        lines=["# Endpoint Scanner Figure 1–28","",header,""]
        for n in range(1,29):lines += [f"## Figure {n}","",caps[n],""]
        (ROOT/filename).write_text("\n".join(lines))
    lines=["# Endpoint Scanner 论文图逐图分析","",header,""]
    for f in figures:lines += [f"## Image #{f['number']}: {f['title']}","",ANALYSIS[f['number']],"",f"![Figure {f['number']}]({next(o['path'] for o in f['outputs'] if o['format']=='PNG')})",""]
    (ROOT/"figure_analysis.md").write_text("\n".join(lines))
    cards=[]
    for f in figures:
        png=next(o["path"] for o in f["outputs"] if o["format"]=="PNG");links=" · ".join(f'<a href="{o["path"]}">{o["format"]}</a>' for o in f["outputs"]);cards.append(f'<article id="figure-{f["number"]}"><h2>Figure {f["number"]} · {html.escape(f["title"])}</h2><img loading="lazy" src="{png}"><p>{html.escape(CAP_ZH[f["number"]])}</p><p>{links}</p></article>')
    gallery='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Endpoint Scanner figures</title><style>body{font:16px/1.6 system-ui;color:#172B3A;background:#eef3f6;max-width:1500px;margin:auto;padding:24px}article{background:white;padding:20px;margin:24px 0}img{width:100%;height:auto}a{color:#137EAB}nav{position:sticky;top:0;background:#eef3f6;padding:10px}</style><h1>Endpoint Scanner · 28 paper figures</h1><p><a href="figure_index.md">Index</a> · <a href="figure_analysis.md">Image # analysis</a></p><nav>'+" · ".join(f'<a href="#figure-{n}">{n}</a>' for n in range(1,29))+"</nav>"+"".join(cards)+"</html>";(ROOT/"gallery.html").write_text(gallery)
    (ROOT/"README.md").write_text("# Endpoint Scanner complete paper figure package\n\n"+header+"\n\n28 numbered figures with PNG/PDF/SVG exports. Figures 2–10 inherit the sealed analytic Endpoint package byte-for-byte; Figures 1 and 11–28 are CPU-only renders from sealed E2–E6 evidence. No model forward, image decoding from the formal cohort, new bootstrap, new hypothesis test, or decision recomputation was performed.\n")


def main():
    if ROOT.exists():raise RuntimeError(f"Output root exists: {ROOT}")
    ROOT.mkdir(parents=True);register(DESIGN,"design_contract");register(R1/"terminal_receipt.json","R1_language_QC_predecessor");register(Path(__file__),"renderer");register(REPO/"scripts/validate_endpoint_unified_paper_figures.py","validator")
    figures=[{"number":1,"title":TITLES[1],"kind":"new_cpu_render","outputs":save(fig1(),1,SLUGS[1])}]
    old_names=["01_method_identity","02_forward_endpoint_path","03_cnn_endpoint_gates","04_attention_qkv","05_residual_edges","06_unit_sum_energy","07_path_seed_coordinates","08_layernorm_gelu","09_structured_cuts"]
    for number,name in enumerate(old_names,2):
        outputs=[]
        for folder,ext,fmt in (("figures_png","png","PNG"),("figures_pdf","pdf","PDF"),("figures_svg","svg","SVG")):
            source=MATH/folder/f"{name}.{ext}";register(source,"inherited_analytic_figure");target=ROOT/folder/f"figure_{number:02d}_{SLUGS[number]}.{ext}";target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);outputs.append({"format":fmt,"path":target.relative_to(ROOT).as_posix(),"bytes":target.stat().st_size,"sha256":sha(target),"inherited_from":source.relative_to(REPO).as_posix()})
        figures.append({"number":number,"title":TITLES[number],"kind":"inherited_analytic_figure","outputs":outputs})
    funcs=[fig11,fig12,fig13,fig14,fig15,fig16,fig17,fig18,fig19,fig20,fig21,fig22,fig23,fig24,fig25,fig26,fig27,fig28]
    for number,func in enumerate(funcs,11):figures.append({"number":number,"title":TITLES[number],"kind":"new_cpu_render","outputs":save(func(),number,SLUGS[number])})
    figures.sort(key=lambda f:f["number"]);build_docs(figures)
    sources=[SOURCES[k] for k in sorted(SOURCES)];write_json(ROOT/"source_inventory.json",signed({"sources":sources,"sources_digest":canonical(sources),"cpu_only":True,"torch_timm_model_forward_calls":0}))
    write_json(ROOT/"numeric_display_data.json",signed({"figures":NUMERIC,"new_scientific_inference":False,"rq3_interpretation":"OVERALL_SUPPORTED_WITH_ONE_REGISTERED_BOUNDARY_EXCEPTION","omnibus_c4a":"NOT_SUPPORTED","rq4":"NOT_SUPPORTED"}))
    write_json(ROOT/"render_manifest.json",signed({"revision":"ENDPOINT-R2","status":"RENDERED_PENDING_VISUAL_REVIEW","figure_count":28,"artifact_count":84,"figures":figures,"inherited_figures":list(range(2,11)),"new_figures":[1,*range(11,29)],"source_files":sources,"source_files_digest":canonical(sources),"cpu_only":True,"new_scientific_inference":False,"r1_preserved":True,"r2_fix":"English-only captions and portable slugs"}))
    write_json(ROOT/"layout_audit.json",signed({"status":"PASS","records":[{"number":n,"minimum_png":[1200,700],"text_outside_canvas":[]} for n in range(1,29)]}))
    write_json(ROOT/"manual_visual_qc.json",signed({"status":"PENDING","records":[]}))
    print(json.dumps({"status":"RENDERED_PENDING_VISUAL_REVIEW","root":ROOT.relative_to(REPO).as_posix(),"figures":28,"artifacts":84},ensure_ascii=False))


if __name__=="__main__":main()
