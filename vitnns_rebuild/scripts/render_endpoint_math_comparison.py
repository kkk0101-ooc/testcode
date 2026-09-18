#!/usr/bin/env python3
"""CPU-only analytic Endpoint/CNN/Path figures. No trained-model inference."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/endpoint-math-comparison-mpl")
import argparse
import hashlib
import html
import json
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "reports/nns_vit_forward_learning_image_comparison_20260917"
DEFAULT = REPO / "reports/nns_vit_endpoint_math_comparison_r1_20260918"
PLAN = REPO / "docs/NNS_ViT_Endpoint_Scanner论文思路与规划与步骤_20260918.md"
MATH = REPO / "docs/NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md"
INK, MUTED, BLUE, ORANGE, GREEN = "#172B3A", "#596B7C", "#126B9B", "#B65D22", "#327653"
X = np.array([1., 1., 3., 3.])
P = np.array([[.5, .5, 0., 0.], [0., 0., .5, .5]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def chain(x):
    h = P @ x
    scores = np.outer(h, h)
    a = np.exp(scores - scores.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)
    o = a @ h
    jq, jk, jv = np.zeros((2, 4)), np.zeros((2, 4)), a @ P
    for t in range(2):
        js = np.diag(a[t]) - np.outer(a[t], a[t])
        jq[t] = h @ js @ np.outer(h, P[t])
        jk[t] = h @ js @ (h[t] * P)
    jo = jq + jk + jv
    values = {"x": x, "h": h, "o": o, "r": h + o, "a": a}
    jac = {"x": np.eye(4), "h": P, "o": jo, "r": P + jo}
    parts = {"Q": jq, "K": jk, "V": jv}
    return values, jac, parts


def integrate(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    maps = {k: np.zeros_like(v) for k, v in chain(X)[1].items()}
    parts = {k: np.zeros((2, 4)) for k in ("Q", "K", "V")}
    for a, w in zip((nodes + 1) / 2, weights / 2):
        _, j, p = chain(a * X)
        for k in maps:
            maps[k] += w * j[k] * X
        for k in parts:
            parts[k] += w * p[k] * X
    return maps, parts


def forward_only(x):
    h = P @ x
    scores = np.outer(h, h)
    a = np.exp(scores - scores.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)
    return {"x": x, "h": h, "o": a @ h, "r": h + a @ h}


def ln(u):
    gamma, beta, eps = np.array([1., 1.2, .8, 1.1]), np.array([.2, -.1, .3, 0.]), .2
    centered = u - u.mean()
    sigma = np.sqrt(np.mean(centered ** 2) + eps)
    hat = centered / sigma
    j = gamma[:, None] / sigma * (np.eye(4) - np.ones((4, 4)) / 4 - np.outer(hat, hat) / 4)
    return gamma * hat + beta, j, gamma / sigma


def gelu(u):
    phi = np.exp(-u * u / 2) / np.sqrt(2 * np.pi)
    cdf = .5 * (1 + np.array([math.erf(float(v) / np.sqrt(2)) for v in np.atleast_1d(u)]))
    return u * cdf, cdf + u * phi


def compute():
    values, jac, parts = chain(X)
    end = {k: v * X for k, v in jac.items()}
    pend = {k: v * X for k, v in parts.items()}
    path, ppath = integrate(128)
    path2, ppath2 = integrate(256)
    checks = {}
    checks["quadrature_128_vs_256_max_abs"] = max(float(np.max(np.abs(path[k]-path2[k]))) for k in path)
    checks["path_component_quadrature_max_abs"] = max(float(np.max(np.abs(ppath[k]-ppath2[k]))) for k in ppath)
    finite_errors = []
    eps = 1e-5
    for k in jac:
        fd = np.stack([(forward_only(X + eps*np.eye(4)[i])[k] - forward_only(X - eps*np.eye(4)[i])[k]) / (2*eps) for i in range(4)], axis=-1)
        finite_errors.append(np.max(np.abs(fd - jac[k])))
    checks["analytic_jacobian_finite_difference_max_abs"] = float(max(finite_errors))
    checks["endpoint_qkv_closure_max_abs"] = float(np.max(np.abs(sum(pend.values()) - end["o"])))
    checks["path_qkv_closure_max_abs"] = float(np.max(np.abs(sum(ppath.values()) - path["o"])))
    checks["endpoint_residual_closure_max_abs"] = float(np.max(np.abs(end["h"]+end["o"]-end["r"])))
    checks["path_residual_closure_max_abs"] = float(np.max(np.abs(path["h"]+path["o"]-path["r"])))
    base = forward_only(np.zeros(4))
    checks["path_completeness_max_abs"] = max(float(np.max(np.abs(path[k].sum(-1) - (values[k] - base[k])))) for k in path)
    old = json.loads((OLD / "numeric_data.json").read_text())
    checks["old_path_map_max_abs"] = max(float(np.max(np.abs(np.array(old["learning_images"][oldkey]).reshape(path[k].shape)-path[k]))) for k, oldkey in [("x", "x"), ("h", "h"), ("o", "o"), ("r", "z")])
    checks["old_forward_values_max_abs"] = max(float(np.max(np.abs(np.array(old["values"][oldkey])-values[k]))) for k, oldkey in [("h","h"),("o","o"),("r","z")])
    # CNN: shared [1/2, 1/2] filter, bias -1.5, ReLU, unique max winner.
    pre = values["h"] - 1.5
    relu = np.maximum(pre, 0.)
    winner = int(np.argmax(relu))
    cnn_end = X * P[winner]
    cnn_ig = cnn_end * .5  # alpha > 1.5/3; exact piecewise integration.
    checks["cnn_endpoint_paper_chain_max_abs"] = float(np.max(np.abs((np.eye(2)[winner] @ (np.diag(pre > 0) @ P) @ np.diag(X)) - cnn_end)))
    checks["cnn_path_completeness_abs"] = float(abs(cnn_ig.sum() - relu[winner]))
    # Fixed downstream cotangent for F = (r_1)^2/2.
    g = values["r"][0]
    rid, ratt = g * end["h"][0], g * end["o"][0]
    total = g * end["r"][0]
    step = 1e-5
    fdF = np.array([(.5*forward_only(X+step*np.eye(4)[i])["r"][0]**2-.5*forward_only(X-step*np.eye(4)[i])["r"][0]**2)/(2*step) for i in range(4)]) * X
    checks["downstream_residual_finite_difference_max_abs"] = float(np.max(np.abs(total-fdF)))
    # LN and GELU local Jacobians, including nonzero LN beta.
    u = np.array([-1., .5, 2., -.5])
    ly, lj, frozen = ln(u)
    lfd = np.stack([(ln(u+eps*np.eye(4)[i])[0]-ln(u-eps*np.eye(4)[i])[0])/(2*eps) for i in range(4)], axis=-1)
    checks["layernorm_finite_difference_max_abs"] = float(np.max(np.abs(lj-lfd)))
    gy, gj = gelu(u)
    gfd = (gelu(u+eps)[0]-gelu(u-eps)[0])/(2*eps)
    checks["gelu_finite_difference_max_abs"] = float(np.max(np.abs(gj-gfd)))
    # Nonconstant gradient with IG == endpoint: f(t)=t^3/3-2t^2/3+t/2.
    checks["nonconstant_equal_example_abs"] = abs((1/3 - 2/3 + 1/2) - (1 - 4/3 + 1/2))
    tolerances = {k: (2e-8 if "finite_difference" in k else 1e-10) for k in checks}
    for k, value in checks.items():
        if value > tolerances[k]:
            raise ValueError(f"{k}: {value} > {tolerances[k]}")
    data = {
        "kind": "analytic_schematic_not_trained_model_measurement",
        "x": X.tolist(), "baseline": [0.,0.,0.,0.], "projection": P.tolist(),
        "model": "2 patches, scalar Q=K=V=h, softmax(h h^T), residual h+o",
        "model_scope": "main chain omits LN/MLP/position/bias; LN and GELU are separate local examples",
        "values": {k:v.tolist() for k,v in values.items()},
        "endpoint": {k:v.tolist() for k,v in end.items()},
        "path": {k:v.tolist() for k,v in path.items()},
        "endpoint_qkv": {k:v.tolist() for k,v in pend.items()},
        "path_qkv": {k:v.tolist() for k,v in ppath.items()},
        "cnn": {"pre":pre.tolist(),"relu":relu.tolist(),"winner":winner,"score":float(relu[winner]),"endpoint":cnn_end.tolist(),"path":cnn_ig.tolist(),"bias":-1.5},
        "residual_final": {"score":float(.5*g*g),"cotangent":float(g),"identity":rid.tolist(),"attention":ratt.tolist(),"total":total.tolist()},
        "ln": {"u":u.tolist(),"output":ly.tolist(),"jacobian":lj.tolist(),"full_map":(u*lj[0]).tolist(),"frozen_map":(u*np.diag(frozen)[0]).tolist(),"eps":.2},
        "gelu": {"u":u.tolist(),"output":gy.tolist(),"derivative":gj.tolist()},
        "checks":checks,"tolerances":tolerances,
    }
    return data


def text(ax,x,y,s,size=12,color=INK,ha="left",weight="normal"):
    return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va="center",weight=weight)


def canvas(title,subtitle,height=8.3):
    fig=plt.figure(figsize=(16,height))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,16),ylim=(0,height));ax.axis("off")
    text(ax,.45,height-.43,title,21,weight="bold")
    text(ax,.45,height-.92,subtitle,11,MUTED)
    return fig,ax


def box(ax,x,y,w,h,label,color=BLUE,size=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.08",edgecolor=color,facecolor="#F5F8FB",lw=1.2))
    text(ax,x+w/2,y+h/2,label,size,color,ha="center")


def arrow(ax,x1,y1,x2,y2,dashed=False):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,lw=1.2,color=MUTED,linestyle="--" if dashed else "-"))


def map2(ax,cx,y,arr,label,width=1.45,vmax=3.,total=True):
    a=np.asarray(arr).reshape(2,2)
    ax.imshow(a,extent=(cx-width/2,cx+width/2,y,y+width),cmap="RdBu_r",vmin=-vmax,vmax=vmax,origin="upper",aspect="auto")
    for r in range(2):
        for c in range(2):
            value=a[r,c]
            text(ax,cx-width/2+(c+.5)*width/2,y+width-(r+.5)*width/2,f"{value:.3f}",10,"white" if abs(value)>.60*vmax else INK,ha="center")
    text(ax,cx,y+width+.26,label,12,ha="center",weight="bold")
    if total:text(ax,cx,y-.25,rf"$\Sigma = {a.sum():.4f}$",11,ha="center")


def fig_identity(d):
    fig,ax=canvas("01  |  Same endpoint principle; different attribution rules", "Original NNS-CNN conditions and the full ViT endpoint graph share input-value seeds.")
    cols=[(.45,"NNS-CNN endpoint",GREEN),(5.65,"NNS-ViT Endpoint",BLUE),(10.85,"Earlier Path-integrated",ORANGE)]
    rows=[
        ("Conv / ReLU / MaxPool",r"$J_{s,x}(x)\,\mathrm{diag}(x)$",r"$L_s=x\odot\nabla s(x)$","Current gates and winner","Paper-formula identity under\nconsistent local-Jacobian rules"),
        ("Patch / LN / attention / MLP",r"$J_{s,x}(x)\,\mathrm{diag}(x)$",r"$L_s=x\odot\nabla s(x)$","Full Q/K/V and residual paths","Internal-target Gradient x Input\nOne endpoint; no baseline forward"),
        ("Same ViT; many path states",r"$\int_0^1 J_{s,x}(x_\alpha)\,d\alpha$",r"$L_s=\Delta x\odot\int_0^1\nabla s(x_\alpha)d\alpha$","Seed: displacement from baseline","Internal-target Integrated Gradients\nInput and baseline both specified"),
    ]
    for (x,title,c),r in zip(cols,rows):
        box(ax,x,6.24,4.65,.63,title,c,15)
        text(ax,x+2.32,5.74,r[0],12,ha="center")
        arrow(ax,x+2.32,5.43,x+2.32,5.06)
        box(ax,x+.17,4.26,4.31,.77,r[1],c,15)
        text(ax,x+2.32,3.66,r[2],14 if c!=ORANGE else 12,ha="center")
        text(ax,x+2.32,2.87,r[3],12,c,ha="center")
        text(ax,x+2.32,2.09,r[4],12,ha="center")
    text(ax,.6,1.13,"Fixed-attention Value-path IG explains a conditional target with clean A held fixed; full Path IG differentiates Q/K/V.",11,MUTED)
    text(ax,.6,.49,"The architecture specifies the Jacobian. The seed and evaluation rule specify endpoint versus path attribution.",13,BLUE,weight="bold")
    return fig


def fig_chain(d):
    fig,ax=canvas("02  |  Normal forward above; input-coordinate maps below", "Same analytic input as the earlier figure set: x = [1, 1; 3, 3], zero reference for IG.",9.7)
    cols=[2.55,5.3,8.05,10.8,13.55]
    items=[("x",0,r"$x_1$"),("h",0,r"$h_1$"),("h",1,r"$h_2$"),("o",0,r"$o_1$"),("r",0,r"$r_1=h_1+o_1$")]
    text(ax,.5,7.65,"Forward",12,BLUE,weight="bold")
    text(ax,.5,5.12,"Endpoint",12,BLUE,weight="bold")
    text(ax,.5,2.4,"Path IG",12,ORANGE,weight="bold")
    for cx,(k,i,label) in zip(cols,items):
        box(ax,cx-.78,7.28,1.56,.9,label+"\n"+f"{d['values'][k][i]:.4f}",size=13)
        arrow(ax,cx,7.1,cx,6.56,True)
        map2(ax,cx,4.64,d["endpoint"][k][i],label,1.45,3.)
        map2(ax,cx,1.91,d["path"][k][i],label,1.45,3.)
    text(ax,8.,1.07,r"$h=Px,\quad Q=K=V=h,\quad A=\mathrm{softmax}(hh^T),\quad o=Ah,\quad r=h+o$",14,ha="center")
    text(ax,.5,.45,"Each lower map explains the scalar directly above it. Patch maps agree; attention changes the path Jacobian.",12,MUTED)
    return fig


def fig_cnn(d):
    fig,ax=canvas("03  |  Original CNN endpoint logic: current gates and winner", "Analytic CNN: shared [1/2, 1/2] convolution, bias -1.5, ReLU, then max pooling.")
    stages=[(.55,"Input\n[1, 1; 3, 3]"),(4.5,"Convolution\n[-0.5, 1.5]"),(8.45,"ReLU\n[0, 1.5]"),(12.4,"MaxPool\ns = 1.5")]
    for x,l in stages:box(ax,x,5.96,3.05,1.03,l,GREEN,14)
    for a,b in zip(stages,stages[1:]):arrow(ax,a[0]+3.08,6.47,b[0]-.06,6.47)
    map2(ax,2.3,2.79,d["cnn"]["endpoint"],"Endpoint pooled unit",1.8,1.5)
    map2(ax,6.45,2.79,d["cnn"]["path"],"Same CNN, Path IG",1.8,1.5)
    text(ax,9.1,4.8,r"$s(\alpha x)=\max(3\alpha-1.5,0)$",16)
    text(ax,9.1,4.13,r"$\nabla s(x)=P_{2,:}$",15)
    text(ax,9.1,3.46,r"$L^{end}=x\odot P_{2,:}$",15)
    text(ax,9.1,2.79,r"$IG=x\odot\int_{1/2}^1P_{2,:}\,d\alpha$",15)
    text(ax,.55,1.68,"Endpoint sum = 3; response = 1.5. Bias sets the gate threshold but receives no direct input seed.",13)
    text(ax,.55,.95,"The pooled winner is unique at x. Along the path, the output is zero until the ReLU threshold is crossed.",12,MUTED)
    text(ax,.55,.43,"This is a paper-formula CNN example, not a reproduction of the paper's trained-model images.",11,MUTED)
    return fig


def fig_attention(d):
    fig,ax=canvas("04  |  Full attention = Query routing + Key routing + Value", "Target o1 in the same two-token chain; head dimension = 1, bias = 0; four input coordinates.",9.2)
    text(ax,.55,7.78,r"$dO=J_{softmax}(S)[(dQ)K^T+Q(dK)^T]V+A(dV)$",17)
    text(ax,.55,7.13,"Endpoint components",12,BLUE,weight="bold")
    text(ax,.55,4.30,"Path-integrated components",12,ORANGE,weight="bold")
    ax.plot([12.45,12.45],[1.75,7.1],color="#B7C5CE",lw=1,ls="--")
    cols=[1.9,4.9,7.9,10.9,13.9]
    names=["Q","K","V","full","fixed"]
    labels=["Query","Key","Value","Full attention","Fixed-A endpoint"]
    for cx,k,l in zip(cols,names,labels):
        e=d["endpoint_qkv"][k][0] if k in ("Q","K","V") else d["endpoint"]["o"][0] if k=="full" else d["endpoint_qkv"]["V"][0]
        map2(ax,cx,4.95,e,l,1.5,1.9)
        p=d["path_qkv"][k][0] if k in ("Q","K","V") else d["path"]["o"][0] if k=="full" else d["endpoint_qkv"]["V"][0]
        map2(ax,cx,2.18,p,"Path "+l if k!="fixed" else "Fixed-A Path IG",1.5,1.9)
    text(ax,.55,1.34,"Each of the first two rows closes: Q + K + V = full. The right column instead freezes clean A.",12)
    text(ax,.55,.81,"Here V is linear and the baseline is zero, so fixed-A Endpoint = fixed-A IG = the endpoint Value component.",11,MUTED)
    text(ax,.55,.32,"That last equality is specific to this example. In general, conditional Value-path IG differs from full-path Value conductance.",10.5,MUTED)
    return fig


def fig_residual(d):
    fig,ax=canvas("05  |  Residual input edges: one downstream cotangent", "First explain r1 itself, then the downstream scalar F = r1^2 / 2.",9.)
    h,o,r=d["values"]["h"][0],d["values"]["o"][0],d["values"]["r"][0]
    for cx,label,val in [(3.,"h1",h),(7.5,"o1",o),(12.,"r1",r)]:
        box(ax,cx-1.2,6.77,2.4,.8,f"{label} = {val:.4f}")
    text(ax,5.25,7.17,"+",22,ha="center");text(ax,9.75,7.17,"=",22,ha="center")
    for cx,k in [(3.,"h"),(7.5,"o"),(12.,"r")]:map2(ax,cx,4.4,d["endpoint"][k][0],"Endpoint "+k+"1",1.55,3.)
    text(ax,5.25,5.2,"+",22,ha="center");text(ax,9.75,5.2,"=",22,ha="center")
    rf=d["residual_final"]
    for cx,k,l in [(3.,"identity","Identity edge"),(7.5,"attention","Attention edge"),(12.,"total","Total for F")]:map2(ax,cx,1.87,rf[k],l,1.55,8.)
    text(ax,5.25,2.64,"+",22,ha="center");text(ax,9.75,2.64,"=",22,ha="center")
    text(ax,.55,1.05,r"$g_r=\nabla_rF;\quad E_{id}=x\odot J_h^Tg_r,\quad E_{attn}=x\odot J_o^Tg_r$",14)
    text(ax,.55,.43,"Every input-to-F path crosses this add. Using the total gradient at h as identity would count the attention path twice.",11,MUTED)
    return fig


def fig_targets(d):
    fig,ax=canvas("06  |  Neuron, channel sum and original filter aggregation", "Same h = Px. The target determines the learning image; activation weighting changes the target.")
    h=np.array(d["values"]["h"]);unit=np.array(d["endpoint"]["h"])
    targets=[(2.8,"Single unit",r"$s=h_2=3$",unit[1]),(8.,"Channel sum",r"$s=h_1+h_2=4$",unit.sum(0)),(13.2,"Feature-map energy",r"$s=(h_1^2+h_2^2)/2=5$",h@unit)]
    for cx,title,formula,a in targets:
        box(ax,cx-2.12,5.81,4.24,.69,title,size=15)
        text(ax,cx,5.14,formula,16,ha="center")
        map2(ax,cx,2.67,a,"Endpoint learning image",1.85,4.5)
    text(ax,.6,1.55,r"$I^{filter}=\sum_t h_t L_{h_t}=x\odot\nabla_x\left(\frac{1}{2}\sum_t h_t^2\right)$",18)
    text(ax,.6,.84,"The paper's activation weights and unit maps must refer to the same tensor. Sum and energy are distinct targets.",12,MUTED)
    text(ax,.6,.33,"Energy is planned, not yet registered. In this quadratic example the endpoint sum is 2s = 10, not s = 5.",11,MUTED)
    return fig


def fig_rules(d):
    fig,ax=canvas("07  |  Endpoint and Path IG: state, seed and equality", "Compare the same target and input coordinates before comparing images.",9.4)
    text(ax,.6,7.96,r"$L^{end}=x\odot\nabla s(x)$",20,BLUE)
    text(ax,7.9,7.96,r"$L^{\Delta end}=\Delta x\odot\nabla s(x)$",20,BLUE)
    text(ax,.6,7.04,r"$IG=\Delta x\odot\int_0^1\nabla s(x_\alpha)d\alpha$",19,ORANGE)
    text(ax,.6,6.05,r"$L^{\Delta end}_i=IG_i\ \Longleftrightarrow\ \Delta x_i\left[\partial_i s(x)-\int_0^1\partial_i s(x_\alpha)d\alpha\right]=0$",17)
    box(ax,.6,4.94,14.8,.68,"Constant path Jacobian is sufficient, not necessary. Canonical Endpoint also has a different seed if x0 is nonzero.",size=12)
    # Endpoint derivative and path average both equal 1/6.
    sub=fig.add_axes([.08,.17,.36,.29])
    a=np.linspace(0,1,301);g=a*a-4*a/3+1/2
    sub.plot(a,g,color=ORANGE,lw=2.4,label="path gradient")
    sub.axhline(1/6,color=BLUE,ls="--",label="endpoint = path mean")
    sub.set(xlabel=r"$\alpha$",ylabel=r"$s'(\alpha)$",title="Nonconstant gradient, equal attributions")
    sub.legend(fontsize=9,loc="lower center");sub.spines[["top","right"]].set_visible(False)
    text(ax,8.,4.04,r"$s(t)=t^3/3-2t^2/3+t/2$",16)
    text(ax,8.,3.35,r"$x_0=0,\ x=1:\quad L^{end}=IG=1/6$",15)
    text(ax,8.,2.61,r"$x=(r-\mu)/\sigma$",16)
    text(ax,8.,1.98,r"$x_i\partial_{x_i}s=(r_i-\mu_i)\partial_{r_i}s$",16)
    text(ax,.6,.58,"Normalized-input endpoint depends on the coordinate origin; it does not require a baseline forward or imply completeness.",11.5,MUTED)
    return fig


def fig_ln(d):
    fig,ax=canvas("08  |  ViT local operators: full LayerNorm and GELU derivatives", "Local analytic examples supplement the patch-attention-residual chain; no trained-model measurements.",9.)
    text(ax,.6,7.65,r"$\frac{\partial y_i}{\partial u_j}=\frac{\gamma_i}{\sigma}\left[\delta_{ij}-\frac{1}{D}-\frac{\hat u_i\hat u_j}{D}\right]$",21)
    text(ax,8.4,7.65,r"$GELU'(u)=\Phi(u)+u\phi(u)$",21)
    map2(ax,2.6,4.63,d["ln"]["full_map"],"Full LN: target y1",1.9,1.15)
    map2(ax,6.4,4.63,d["ln"]["frozen_map"],"Frozen mean / variance",1.9,1.15)
    sub=fig.add_axes([.58,.47,.35,.27])
    xx=np.linspace(-3,3,401)
    sub.plot(xx,gelu(xx)[1],color=BLUE,lw=2.3,label="GELU derivative")
    sub.plot(xx,(xx>0).astype(float),color=MUTED,lw=1.6,ls="--",label="ReLU gate")
    sub.set(xlabel="u",ylabel="local multiplier");sub.legend(fontsize=10);sub.spines[["top","right"]].set_visible(False)
    text(ax,.6,3.56,r"$u=[-1,\ 0.5,\ 2,\ -0.5],\quad D=4,\quad \epsilon=0.2$"+"  (maps in local u coordinates)",13)
    text(ax,.6,2.82,"LN uses the biased variance (division by D). Mean and variance couple channels.",13)
    text(ax,.6,2.13,"LN beta, projection bias, CLS and positional parameters affect downstream states; they have no direct image seed.",12)
    text(ax,.6,1.41,"GELU uses its continuous endpoint derivative. Substituting a binary gate changes the model Jacobian.",12)
    text(ax,.6,.6,"A full ViT implementation retains these operators, both residual paths, head concat and output projection in eval mode.",11.5,MUTED)
    return fig


def fig_cuts(d):
    fig,ax=canvas("09  |  Tensor-valued cuts: endpoint versus path conductance", "Fixed scalar target F; inner products sum over every coordinate in component z_k.",8.8)
    box(ax,.55,6.49,6.85,.66,"Endpoint component",BLUE,16)
    box(ax,8.0,6.49,7.45,.66,"Input-resolved path conductance",ORANGE,16)
    text(ax,.65,5.78,r"$E^C_{k,i}=x_i\langle\nabla_{z_k}F(x),\partial_i z_k(x)\rangle$",18)
    text(ax,8.1,5.78,r"$K^C_{k,i}=\Delta x_i\int_0^1\langle\nabla_{z_k}F(x_\alpha),\partial_i z_k(x_\alpha)\rangle d\alpha$",13)
    text(ax,.65,4.94,r"$\sum_kE^C_{k,i}=x_i\partial_iF(x)$",18,BLUE)
    text(ax,8.1,4.94,r"$\sum_kK^C_{k,i}=IG_i(F;x,x_0)$",18,ORANGE)
    for cx,label in [(2.3,"Identity edge"),(6.4,"Head 1 ... Head H"),(12.5,"Same final target F")]:
        box(ax,cx-1.65,3.27,3.3,.76,label,size=13)
    text(ax,4.37,3.65,"+",22,ha="center")
    arrow(ax,8.15,3.65,10.76,3.65)
    text(ax,.65,2.49,"Each input-to-F path crosses exactly one registered edge. Heads include the native projection and downstream graph.",12)
    text(ax,.65,1.82,"Residual/head edge cuts and attention differential paths are two forms of structured decomposition.",12)
    text(ax,.65,1.16,"Each block or cut reconstructs its own registered total. Do not add different cuts, or routing together with Q and K.",12)
    text(ax,.65,.49,"Closure is a chain-rule property. Finite interventions require their own directions, reference activations and measurements.",11.5,MUTED)
    return fig


DRAWERS=[fig_identity,fig_chain,fig_cnn,fig_attention,fig_residual,fig_targets,fig_rules,fig_ln,fig_cuts]
KEYS=["01_method_identity","02_forward_endpoint_path","03_cnn_endpoint_gates","04_attention_qkv","05_residual_edges","06_unit_sum_energy","07_path_seed_coordinates","08_layernorm_gelu","09_structured_cuts"]
TITLES=["三种方法的身份","同一前向量的 Endpoint 与 Path 图","原始 CNN 终点门控","Attention Q/K/V 完整传播","Residual 输入边与下游目标","单元、求和与原论文 energy 聚合","路径、seed 与坐标原点","LayerNorm 与 GELU 的真实 Jacobian","向量成分的 Endpoint cut 与 Path conductance"]
NOTES=[
"NNS-CNN 与 Endpoint Scanner 共享输入值 seed 和当前点 Jacobian；ViT 增加新的局部算子。此前完整 Path IG 改用输入差值和路径平均 Jacobian。原 NNS-CNN 等价限定为门控一致的单神经元生成规则；fixed-attention Value-path IG 解释另一个条件目标。",
"上排是实际解析前向值；中排是对应标量的 canonical Endpoint；下排是同一标量的 zero-baseline IG。每图保留四个原始输入位置。Patch 层相等，attention 层不同；Endpoint 的像素和不要求等于上排响应。所有图使用共同色标 [-3,3]，未逐图归一化。",
"共享卷积核 [0.5,0.5] 加 bias -1.5，经 ReLU 后 max-pool 输出 1.5。终点图像素和为 3；路径在 alpha=0.5 才打开获胜门，因此 IG 和为 1.5。当前 winner 唯一；非激活路径段的零输出不影响积分。这是原论文规则的解析例子，不是训练 CNN 的复现实验。",
"前四列为 Q、K、V 及其和；上排使用终点，下排沿路径积分。最右列固定 clean A，在本例线性 Value、零参考条件下，两种 fixed-A 方法恰好相等，并等于终点 Value 分量；它不等于完整 attention，也不等于下排 full-path Value 分量。一般上游非线性时该特例等式不成立。",
"上排 r1=h1+o1；中排对这个 residual 响应的图逐像素相加。下排改为同一个下游目标 F=r1^2/2，identity/attention 都乘加法器输出 cotangent g_r。每条输入到 F 的路径经过该 add，因此闭合。不同排目标不同；不能把源 h 的总梯度当 identity 再加 branch。",
"单元、channel sum 和 energy 使用同一 h=Px，却定义三个不同标量。原 NNS filter aggregate 的激活加权和对应 energy 的 endpoint 图，前提是权重与单元图使用同一激活张量。二次 energy 的 endpoint 和为 10，即响应 5 的两倍，不具有响应 completeness。图为数学定义演示；当前 scanner 尚未注册 energy kind。",
"列出 canonical、baseline-relative endpoint 与 IG 的精确定义。Jacobian 恒定只是相等的充分条件；例子 s(t)=t^3/3-2t^2/3+t/2 在 [0,1] 上梯度变化，但平均值与终点同为 1/6。右下说明 normalized-input 的坐标原点依赖。",
"独立局部例子，地图坐标是当前 LN 输入 u 而非完整图像像素：完整 LN Jacobian 与冻结均值/方差的近似产生不同的输入坐标图。采用总体方差、epsilon=0.2 便于展示，非生产模型参数；gamma=[1,1.2,0.8,1.1]、beta=[0.2,-0.1,0.3,0]。GELU 导数曲线与二值 ReLU 门不同。主解析链省略 LN/MLP，这张补充其真实算子合同。",
"成分可以是向量或张量，内积遍历其全部内部坐标。Endpoint 在单点使用 seed x，Path 在路径上使用 seed Δx；完整切面条件在相应求值状态成立时分别闭合到 G×I 和 IG。Head 对最终目标的分解须包含 identity bypass、projection 与下游。它是公式示意，未新增多 head 模型实验。",
]


def document(root,records):
    rows=["# Endpoint / Path / NNS-CNN：数学对比图", "", "[图册](gallery.html) · [论文规划](../../docs/"+PLAN.name+") · [数学分析](../../docs/"+MATH.name+")", "", "本图包全部为可复算的解析示意。红为正、蓝为负、白为零；格内数字是未经显示归一化的数值。Figure 2–6 各图内部使用固定对称色标（Residual 的两种目标分排设色标），颜色跨图不比较大小。", ""]
    cards=[]
    for i,(record,title,note) in enumerate(zip(records,TITLES,NOTES),1):
        key=record["key"]; links={r["format"]:r["path"] for r in record["files"]}
        rows += [f"## Image #{i}：{title}","",note,"",f"![Image #{i}]({links['PNG']})",""," · ".join(f"[{fmt}]({p})" for fmt,p in links.items()),""]
        cards.append(f'<article id="image-{i}"><h2>Image #{i}：{html.escape(title)}</h2><p>{html.escape(note)}</p><a href="{links["PNG"]}"><img loading="lazy" src="{links["PNG"]}" alt="{html.escape(title)}"></a><p>'+" · ".join(f'<a href="{p}">{fmt}</a>' for fmt,p in links.items())+"</p></article>")
    rows += ["## 数值与来源","","[numeric_data.json](numeric_data.json) 保存输入、前向、完整 Endpoint/IG 图、Q/K/V、CNN、residual、LN/GELU 例子及数值检查。[validation_report.json](validation_report.json) 是解析验证结果；[artifact_manifest.json](artifact_manifest.json) 绑定脚本、文档、原论文文本、旧图数据和全部图文件。","","同一旧解析链的 Path 数组按原始 JSON 复核；本轮没有加载训练模型或使用 GPU。"]
    (root/"figure_analysis.md").write_text("\n".join(rows),encoding="utf-8")
    (root/"gallery.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Endpoint 数学对比图</title><style>body{max-width:1600px;margin:auto;padding:24px;background:#eef2f6;color:#172b3a;font:16px/1.65 system-ui}article{background:white;padding:24px;margin:24px 0;border-radius:10px}img{width:100%;height:auto}a{color:#126b9b}</style><h1>NNS-ViT Endpoint Scanner：数学对比图</h1><p>解析示意 · 正常前向与输入坐标图对应 · Endpoint / Path IG / NNS-CNN</p>'+''.join(cards)+'</html>',encoding="utf-8")
    (root/"README.md").write_text("# NNS-ViT Endpoint 数学对比图\n\n[图册](gallery.html) · [Image #1–#9](figure_analysis.md) · [数值](numeric_data.json) · [验证](validation_report.json)\n\n9 张图，各有 PNG/PDF/SVG；全部为 CPU 解析示意，不是新增模型实测。继承旧图的输入和布局，补齐 Endpoint 与原始 CNN 对比。\n\n仓库根目录运行（新目录必须不存在）：\n\n§§§bash\nCUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python \\\n  vitnns_rebuild/scripts/render_endpoint_math_comparison.py \\\n  --output-dir vitnns_rebuild/reports/nns_vit_endpoint_math_comparison_reproduction\n§§§\n\n复核既有输出的哈希和数值：\n\n§§§bash\nCUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python \\\n  vitnns_rebuild/scripts/render_endpoint_math_comparison.py \\\n  --verify vitnns_rebuild/reports/nns_vit_endpoint_math_comparison_r1_20260918\n§§§\n".replace('§',chr(96)),encoding="utf-8")


def verify(root):
    manifest=json.loads((root/"artifact_manifest.json").read_text())
    errors=[]
    for item in manifest["artifacts"]:
        path=root/item["path"]
        if not path.exists() or sha(path)!=item["sha256"]:errors.append(item["path"])
    for item in manifest["sources"]:
        path=REPO/item["path"]
        if not path.exists() or sha(path)!=item["sha256"]:errors.append("source: "+item["path"])
    if inventory(OLD)!=manifest["predecessor_inventory"]:errors.append("predecessor changed")
    if compute()!=json.loads((root/"numeric_data.json").read_text()):errors.append("numeric recomputation differs")
    if errors:raise ValueError(errors)
    print(json.dumps({"status":"PASS_RECOMPUTED","figures":len(DRAWERS),"exports":3*len(DRAWERS),"checks":"hashes, sources, predecessor, fresh numeric recomputation"}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,default=DEFAULT);parser.add_argument("--verify",type=Path);args=parser.parse_args()
    if args.verify:verify(args.verify.resolve());return
    root=args.output_dir.resolve()
    if root.exists():raise FileExistsError("Use a fresh output directory: "+str(root))
    before=inventory(OLD);d=compute();root.mkdir(parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"pdf.fonttype":42,"svg.fonttype":"path","svg.hashsalt":"endpoint-math-v1"})
    records=[];layout=[]
    for key,draw in zip(KEYS,DRAWERS):
        fig=draw(d);fig.canvas.draw();renderer=fig.canvas.get_renderer()
        outside=[]
        for ax in fig.axes:
            for artist in ax.texts:
                b=artist.get_window_extent(renderer)
                if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:outside.append(artist.get_text())
        layout.append({"figure":key,"text_outside_canvas":outside})
        if outside:raise ValueError("Text outside canvas: "+repr(outside))
        files=[]
        for ext in ("png","pdf","svg"):
            path=root/f"figures_{ext}"/f"{key}.{ext}";path.parent.mkdir(exist_ok=True)
            fig.savefig(path,dpi=240,facecolor="white")
            if ext == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            files.append({"format":ext.upper(),"path":path.relative_to(root).as_posix()})
        plt.close(fig);records.append({"key":key,"files":files});print(key,flush=True)
    if inventory(OLD)!=before:raise ValueError("Predecessor changed during rendering")
    write(root/"numeric_data.json",d)
    write(root/"layout_audit.json",layout)
    write(root/"validation_report.json",{"status":"PASS_ANALYTIC_NUMERICS","figure_count":len(DRAWERS),"export_count":3*len(DRAWERS),"checks":d["checks"],"tolerances":d["tolerances"],"torch_imported":"torch" in sys.modules,"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],"new_trained_model_forwards":0,"predecessor_unchanged":True,"visual_review":"separate manual_visual_qc.json"})
    document(root,records)
    sources=[Path(__file__),PLAN,MATH,REPO/"src/vitnns/endpoint_vit.py",REPO/"src/vitnns/attribution.py",OLD/"numeric_data.json",OLD/"terminal_receipt.json",REPO.parent/"相关文章/Understanding neural network through neuron level visualization/zh/03-所提方法.md",REPO.parent/"相关文章/Understanding neural network through neuron level visualization/zh/04-实验.md"]
    write(root/"artifact_manifest.json",{"schema":"endpoint_math_comparison_v1","sources":[{"path":os.path.relpath(p,REPO),"sha256":sha(p),"bytes":p.stat().st_size} for p in sources],"predecessor_inventory":before,"artifacts":[{"path":p.relative_to(root).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(root.rglob("*")) if p.is_file()]})
    verify(root)


if __name__=="__main__":main()
