"""Independent original-NNS propagation fixtures for the M3 fidelity bridge.

The explicit path propagates one learning-image row per current activation. It
does not call autograd. The comparison path differentiates the ordinary Torch
forward and forms ``x * grad``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def diagonal_learning_image(value: Tensor) -> Tensor:
    return torch.diag(value.reshape(-1))


def linear(value: Tensor, learning: Tensor, layer: nn.Linear) -> tuple[Tensor, Tensor]:
    return layer(value), layer.weight @ learning


def relu(value: Tensor, learning: Tensor) -> tuple[Tensor, Tensor]:
    # PyTorch's chosen derivative at zero is zero.
    return F.relu(value), (value > 0).reshape(-1, 1).to(learning) * learning


def conv2d(value: Tensor, learning: Tensor, layer: nn.Conv2d) -> tuple[Tensor, Tensor]:
    output = layer(value.unsqueeze(0)).squeeze(0)
    coordinates = learning.shape[1]
    propagated = F.conv2d(
        learning.T.reshape(coordinates, *value.shape), layer.weight, None,
        layer.stride, layer.padding, layer.dilation, layer.groups,
    )
    return output, propagated.movedim(0, -1).reshape(-1, coordinates)


def max_pool2d(value: Tensor, learning: Tensor, kernel_size: int = 2) -> tuple[Tensor, Tensor]:
    output, indices = F.max_pool2d(value.unsqueeze(0), kernel_size, return_indices=True)
    output, indices = output.squeeze(0), indices.squeeze(0)
    spatial = value.shape[-2] * value.shape[-1]
    offsets = torch.arange(value.shape[0], device=value.device).reshape(-1, 1, 1) * spatial
    source_rows = (indices + offsets).reshape(-1)
    return output, learning[source_rows]


def flatten(value: Tensor, learning: Tensor) -> tuple[Tensor, Tensor]:
    return value.reshape(-1), learning


@dataclass
class FidelityForward:
    output: Tensor
    targets: dict[str, Tensor]


class FidelityNetwork(nn.Module):
    """Five small original-NNS-compatible graph families."""

    def __init__(self, kind: str, dtype: torch.dtype, seed: int = 31019):
        super().__init__(); self.kind = kind
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + ["fcn", "cnn", "cnn_maxpool", "residual_mlp", "residual_cnn"].index(kind))
            if kind == "fcn":
                self.l1=nn.Linear(7,9); self.l2=nn.Linear(9,6); self.l3=nn.Linear(6,4)
            elif kind == "cnn":
                self.c1=nn.Conv2d(1,3,3,padding=1); self.c2=nn.Conv2d(3,2,3,padding=1); self.l=nn.Linear(2*6*6,4)
            elif kind == "cnn_maxpool":
                self.c1=nn.Conv2d(1,3,3,padding=1); self.c2=nn.Conv2d(3,2,3,padding=1); self.l=nn.Linear(2*3*3,4)
            elif kind == "residual_mlp":
                self.l1=nn.Linear(8,8); self.l2=nn.Linear(8,8); self.l3=nn.Linear(8,5)
            elif kind == "residual_cnn":
                self.c1=nn.Conv2d(2,2,3,padding=1); self.c2=nn.Conv2d(2,2,3,padding=1); self.l=nn.Linear(2*5*5,4)
            else: raise ValueError("UNKNOWN_FIDELITY_NETWORK")
        self.to(dtype=dtype)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return {
            "fcn": (7,), "cnn": (1,6,6), "cnn_maxpool": (1,6,6),
            "residual_mlp": (8,), "residual_cnn": (2,5,5),
        }[self.kind]

    def forward_with_targets(self, x: Tensor) -> FidelityForward:
        if self.kind == "fcn":
            a=self.l1(x); b=F.relu(a); c=self.l2(b); d=F.relu(c); y=self.l3(d)
            targets={"hidden_pre":a[2],"hidden_relu":b[5],"deep_hidden":d[3],"output0":y[0],"output3":y[3]}
        elif self.kind in {"cnn","cnn_maxpool"}:
            a=self.c1(x.unsqueeze(0)).squeeze(0); b=F.relu(a)
            if self.kind=="cnn_maxpool": b=F.max_pool2d(b.unsqueeze(0),2).squeeze(0)
            c=self.c2(b.unsqueeze(0)).squeeze(0); d=F.relu(c); y=self.l(d.reshape(-1))
            targets={"conv_pre":a.reshape(-1)[7],"conv_relu":b.reshape(-1)[11],"deep_conv":d.reshape(-1)[5],"output0":y[0],"output3":y[3]}
        elif self.kind == "residual_mlp":
            a=self.l1(x); b=F.relu(a); branch=self.l2(b); residual=x+branch; d=F.relu(residual); y=self.l3(d)
            targets={"branch":branch[1],"residual":residual[4],"residual_relu":d[6],"output0":y[0],"output4":y[4]}
        else:
            a=self.c1(x.unsqueeze(0)).squeeze(0); b=F.relu(a); branch=self.c2(b.unsqueeze(0)).squeeze(0); residual=x+branch; d=F.relu(residual); y=self.l(d.reshape(-1))
            targets={"branch":branch.reshape(-1)[3],"residual":residual.reshape(-1)[17],"residual_relu":d.reshape(-1)[31],"output0":y[0],"output3":y[3]}
        return FidelityForward(y,targets)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_targets(x).output

    @torch.no_grad()
    def explicit_learning_images(self, x: Tensor) -> dict[str, Tensor]:
        value=x.detach(); learning=diagonal_learning_image(value); targets={}
        if self.kind == "fcn":
            a,la=linear(value,learning,self.l1); targets["hidden_pre"]=la[2]
            b,lb=relu(a,la); targets["hidden_relu"]=lb[5]
            c,lc=linear(b,lb,self.l2); d,ld=relu(c,lc); targets["deep_hidden"]=ld[3]
            y,ly=linear(d,ld,self.l3); targets.update(output0=ly[0],output3=ly[3])
        elif self.kind in {"cnn","cnn_maxpool"}:
            a,la=conv2d(value,learning,self.c1); targets["conv_pre"]=la[7]
            b,lb=relu(a,la)
            if self.kind=="cnn_maxpool": b,lb=max_pool2d(b,lb)
            targets["conv_relu"]=lb[11]
            c,lc=conv2d(b,lb,self.c2); d,ld=relu(c,lc); targets["deep_conv"]=ld[5]
            d,ld=flatten(d,ld); y,ly=linear(d,ld,self.l); targets.update(output0=ly[0],output3=ly[3])
        elif self.kind == "residual_mlp":
            a,la=linear(value,learning,self.l1); b,lb=relu(a,la); branch,lbranch=linear(b,lb,self.l2)
            targets["branch"]=lbranch[1]; residual,lresidual=value+branch,learning+lbranch; targets["residual"]=lresidual[4]
            d,ld=relu(residual,lresidual); targets["residual_relu"]=ld[6]
            y,ly=linear(d,ld,self.l3); targets.update(output0=ly[0],output4=ly[4])
        else:
            a,la=conv2d(value,learning,self.c1); b,lb=relu(a,la); branch,lbranch=conv2d(b,lb,self.c2)
            targets["branch"]=lbranch[3]; residual,lresidual=value+branch,learning+lbranch; targets["residual"]=lresidual[17]
            d,ld=relu(residual,lresidual); targets["residual_relu"]=ld[31]
            d,ld=flatten(d,ld); y,ly=linear(d,ld,self.l); targets.update(output0=ly[0],output3=ly[3])
        return targets


def autograd_endpoint(model: FidelityNetwork, x: Tensor) -> dict[str, Tensor]:
    clean=x.detach().clone().requires_grad_(True); targets=model.forward_with_targets(clean).targets
    result={}
    for index,(name,score) in enumerate(targets.items()):
        gradient=torch.autograd.grad(score,clean,retain_graph=index+1<len(targets))[0]
        result[name]=(clean*gradient).reshape(-1).detach()
    return result


def tolerance_ratio(observed: Tensor, expected: Tensor, atol: float, rtol: float) -> float:
    return float(((observed-expected).abs()/(atol+rtol*expected.abs())).max())


def compare_learning_images(model: FidelityNetwork, x: Tensor, atol: float, rtol: float) -> list[dict]:
    explicit=model.explicit_learning_images(x); automatic=autograd_endpoint(model,x)
    if set(explicit)!=set(automatic): raise ValueError("TARGET_SET_MISMATCH")
    rows=[]
    for name in explicit:
        error=explicit[name]-automatic[name]
        rows.append({
            "target":name, "relative_l2":float(error.norm()/(automatic[name].norm()+torch.finfo(x.dtype).eps)),
            "max_tolerance_ratio":tolerance_ratio(explicit[name],automatic[name],atol,rtol),
            "max_abs":float(error.abs().max()),
        })
    return rows
