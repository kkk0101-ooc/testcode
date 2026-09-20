"""Supplement-v1 target-generalized M4 scan.

Derived from endpoint_m4.scan_endpoint_maps; adds selected blocks, a fixed
margin scalar, and target-contracted directions. Original kernel unchanged.
"""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from vitnns.endpoint_nns import _frozen_eval
from vitnns.endpoint_m3_protocol import rademacher
from vitnns.endpoint_m4 import (EndpointM4Error, explicit_graph, _finite, _summary,
    _ratio, _cosine, _total_variation, _relative_l2, _vjp, _head_vjps, _parity,
    _attention_saturation)
Tensor = torch.Tensor

def scan_endpoint_maps(
    model: nn.Module, image: Tensor, *, class_id: int, chunk_size: int = 6,
    detach_validation: bool = False, progress=None, blocks=tuple(range(12)), runnerup=None, task_id="supplement",
) -> dict:
    """Compute per-head attention and both residual-add decompositions."""
    if image.shape != (1, 3, 224, 224) or image.dtype != torch.float32:
        raise EndpointM4Error("INPUT_CONTRACT")
    if len(model.blocks) != 12 or not 0 <= int(class_id) < 1000:
        raise EndpointM4Error("MODEL_OR_TARGET_CONTRACT")
    clean = image.detach().clone().requires_grad_(True)
    observed: dict[int, dict[str, Tensor]] = {}
    native_logits = None
    with _frozen_eval(model), torch.no_grad():
        native_logits = model(image.detach()).detach()
    with _frozen_eval(model), explicit_graph(model, observed), torch.enable_grad():
        logits = model(clean)
        score = logits[:, int(class_id)] if runnerup is None else logits[:, int(class_id)] - logits[:, int(runnerup)]
        _finite(logits, score)
        if not torch.allclose(logits.detach(), native_logits, atol=2e-6, rtol=2e-6):
            raise EndpointM4Error("FORWARD_PARITY")
        full_gradient = torch.autograd.grad(score.sum(), clean, retain_graph=True)[0][0].detach()
        full_map = clean.detach()[0] * full_gradient
        direction_vectors = torch.stack([torch.as_tensor(
            rademacher((3,224,224), "ENDPOINT_SUP_V1_CONTRACTED_JVP", task_id, d),
            device=clean.device, dtype=clean.dtype) / math.sqrt(3*224*224) for d in range(8)])
        attention_rows, residual_rows, validation_rows, block_validation = [], [], [], []
        detach_references = {}
        for block in blocks:
            edge = observed[block]
            required = {"query", "key", "value", "attention", "av", "u",
                        "attention_branch", "r1", "mlp_branch", "r2"}
            if not required.issubset(edge):
                raise EndpointM4Error("CAPTURE_INCOMPLETE")
            gq, gk, gv, gav, g1, g2 = torch.autograd.grad(
                score.sum(), tuple(edge[name] for name in ("query", "key", "value", "av", "r1", "r2")),
                retain_graph=True, create_graph=False,
            )
            q_grad = _head_vjps(edge["query"], clean, gq, chunk_size)
            k_grad = _head_vjps(edge["key"], clean, gk, chunk_size)
            v_grad = _head_vjps(edge["value"], clean, gv, chunk_size)
            native_grad = _head_vjps(edge["av"], clean, gav, chunk_size)
            saturation, cls_saturation = _attention_saturation(edge["attention"].detach())
            q_map, k_map, v_map = clean.detach()[0] * q_grad, clean.detach()[0] * k_grad, clean.detach()[0] * v_grad
            routing_map = q_map + k_map
            explicit_full_map = routing_map + v_map
            native_map = clean.detach()[0] * native_grad
            aggregate = {
                "query": q_map.sum(0), "key": k_map.sum(0),
                "routing": routing_map.sum(0), "value": v_map.sum(0),
                "attention": native_map.sum(0),
            }
            for head in range(native_map.shape[0]):
                native_summary = _summary(native_map[head])
                routing_summary = _summary(routing_map[head])
                value_summary = _summary(v_map[head])
                denominator = routing_summary["absolute_mass"] + value_summary["absolute_mass"]
                path_cancel = None if denominator == 0 else 1.0 - native_summary["absolute_mass"] / denominator
                endpoint_error = native_map[head] - v_map[head]
                gradient_error = native_grad[head] - v_grad[head]
                row = {
                    "block": block, "head": head,
                    "q": _summary(q_map[head]), "k": _summary(k_map[head]),
                    "routing": routing_summary, "value": value_summary, "full": native_summary,
                    "routing_share": _ratio(routing_summary["absolute_mass"], denominator),
                    "frozen_attention_endpoint_error": _relative_l2(endpoint_error, native_map[head]),
                    "frozen_attention_absolute_error": float(endpoint_error.double().norm()),
                    "contracted_gradient_error": _relative_l2(gradient_error, native_grad[head]),
                    "contracted_gradient_absolute_error": float(gradient_error.double().norm()),
                    "routing_value_path_cancellation": path_cancel,
                    "routing_value_cosine": _cosine(routing_map[head], v_map[head]),
                    "mean_query_saturation": float(saturation[head]),
                    "cls_query_saturation": float(cls_saturation[head]),
                    "qkv_native_closure": _parity(explicit_full_map[head], native_map[head]),
                    "target_contracted_jvp_discrepancy": _ratio(
                        float((direction_vectors.double()*gradient_error.double()).flatten(1).sum(1).norm()),
                        float((direction_vectors.double()*native_grad[head].double()).flatten(1).sum(1).norm())),
                    "frozen_attention_identity": _parity(endpoint_error, routing_map[head]),
                }
                if not row["qkv_native_closure"]["pass"] or not row["frozen_attention_identity"]["pass"]:
                    raise EndpointM4Error(
                        f"ATTENTION_ENDPOINT_CLOSURE:{block}:{head}:"
                        f"{row['qkv_native_closure']}:{row['frozen_attention_identity']}"
                    )
                attention_rows.append(row)
            if detach_validation:
                detach_references[block] = {
                    "cotangent": gav.detach().cpu(), "value_gradient": v_grad.detach().cpu(),
                }
            identity = _vjp(edge["u"], clean, g1)
            attention_branch = _vjp(edge["attention_branch"], clean, g1)
            carry = _vjp(edge["r1"], clean, g2)
            mlp_branch = _vjp(edge["mlp_branch"], clean, g2)
            for residual_type, identity_gradient, branch_gradient in (
                ("attention", identity, attention_branch), ("mlp", carry, mlp_branch),
            ):
                identity_map = clean.detach()[0] * identity_gradient
                branch_map = clean.detach()[0] * branch_gradient
                output_map = identity_map + branch_map
                identity_summary, branch_summary = _summary(identity_map), _summary(branch_map)
                output_summary = _summary(output_map)
                denominator = identity_summary["absolute_mass"] + branch_summary["absolute_mass"]
                closure = _parity(output_map, full_map)
                omitted_error = _parity(branch_map, full_map)
                if identity_summary["absolute_mass"] > 0 and omitted_error["max_abs"] == 0:
                    raise EndpointM4Error("NEGATIVE_CONTROL_FALSE_PASS")
                if not closure["pass"]:
                    raise EndpointM4Error("RESIDUAL_ENDPOINT_CLOSURE")
                residual_rows.append({
                    "block": block, "residual_type": residual_type,
                    "identity": identity_summary, "branch": branch_summary, "output": output_summary,
                    "identity_share": _ratio(identity_summary["absolute_mass"], denominator),
                    "branch_share": _ratio(branch_summary["absolute_mass"], denominator),
                    "path_cancellation": None if denominator == 0 else 1.0 - output_summary["absolute_mass"] / denominator,
                    "identity_branch_cosine": _cosine(identity_map, branch_map),
                    "output_identity_patch_tv": _total_variation(output_map, identity_map),
                    "closure": closure,
                    "negative_control": {
                        "control": "omit_nonzero_identity_or_carry",
                        "component_nonzero": identity_summary["absolute_mass"] > 0,
                        "observed_failure_max_abs": omitted_error["max_abs"],
                        "passed": identity_summary["absolute_mass"] > 0 and omitted_error["max_abs"] > 0,
                    },
                })
                aggregate["identity" if residual_type == "attention" else "carry"] = identity_map
                aggregate["attention" if residual_type == "attention" else "mlp"] = branch_map
            block_validation.append({
                "block": block,
                "components": {name: _summary(value) for name, value in aggregate.items()},
                "head_sum_attention_closure": _parity(native_map.sum(0), aggregate["attention"]),
                "qk_routing_closure": _parity(aggregate["query"] + aggregate["key"], aggregate["routing"]),
                "routing_value_attention_closure": _parity(aggregate["routing"] + aggregate["value"], aggregate["attention"]),
            })
            if progress:
                progress()
        endpoint_digest = hashlib.sha256(full_map.cpu().contiguous().numpy().tobytes()).hexdigest()
    # Release the large original graph before optional detach checks.
    del observed, logits, score
    if detach_validation:
        for block in blocks:
            detached_input = image.detach().clone().requires_grad_(True)
            detached_observed: dict[int, dict[str, Tensor]] = {}
            with _frozen_eval(model), explicit_graph(model, detached_observed, detach_attention_blocks=(block,)), torch.enable_grad():
                detached_logits = model(detached_input)
                if not torch.allclose(detached_logits.detach(), native_logits, atol=2e-6, rtol=2e-6):
                    raise EndpointM4Error("DETACH_FORWARD_PARITY")
                reference = detach_references[block]
                cotangent = reference["cotangent"].to(detached_input.device)
                observed_grad = _head_vjps(detached_observed[block]["av"], detached_input, cotangent, chunk_size)
                expected_grad = reference["value_gradient"].to(observed_grad.device)
                validation_rows.append({
                    "block": block, "attention_detach_value_gradient": _parity(observed_grad, expected_grad),
                    "forward_parity": _parity(detached_logits.detach(), native_logits),
                })
                if not validation_rows[-1]["attention_detach_value_gradient"]["pass"]:
                    raise EndpointM4Error("ATTENTION_DETACH_PARITY")
            if progress:
                progress()
    return {
        "target_value": float(native_logits[0, int(class_id)] - (0 if runnerup is None else native_logits[0, int(runnerup)])),
        "full_endpoint": _summary(full_map), "full_endpoint_sha256": endpoint_digest,
        "attention": attention_rows, "residual": residual_rows,
        "block_validation": block_validation,
        "attention_detach_validation": validation_rows,
        "forward_parity": _parity(native_logits, native_logits),
    }


def feature_distributions(model, image, channels_by_block, task_id):
    """Patch-energy targets for frozen channels, storing p and all control q's."""
    captured = {}; handles=[]
    for block in range(12):
        handles.append(model.blocks[block].register_forward_hook(
            lambda _m,_a,out,b=block: captured.__setitem__(b,out)))
    clean=image.detach().clone().requires_grad_(True)
    try:
        with _frozen_eval(model), torch.enable_grad():
            model(clean)
            rows=[]
            for block in range(12):
                h=captured[block][0,1:]; channels=channels_by_block[str(block)]
                energy=.5*h[:,channels].square().sum(0)
                grads=torch.autograd.grad(energy,clean,grad_outputs=torch.eye(len(channels),device=clean.device),
                    is_grads_batched=True,retain_graph=True)[0][:,0]
                maps=clean.detach()[0]*grads
                _finite(maps)
                masses=maps.abs().double().reshape(len(channels),3,14,16,14,16).sum((1,3,5)).reshape(len(channels),196)
                for j,c in enumerate(channels):
                    wrong=channels[(j+1)%len(channels)]
                    seed=int.from_bytes(hashlib.sha256(f'ENDPOINT_SUP_V1_FEATURE|{task_id}|{block}|{c}'.encode()).digest()[:8],'big')
                    rng=np.random.default_rng(seed)
                    random_channel=int(rng.integers(h.shape[-1]-1)); random_channel+=random_channel>=c
                    q=h[:,c].detach().double().square(); mass=masses[j]
                    def distribution(v):
                        v=v.detach().double(); total=float(v.sum())
                        return None if total==0 else (v/total).cpu().tolist()
                    q_correct=distribution(q)
                    rows.append(dict(block=block,channel=c,energy=float(energy[j].detach()),
                        endpoint_absolute_mass=float(mass.sum()),p=distribution(mass),q=q_correct,
                        q_wrong_channel=distribution(h[:,wrong].square()),wrong_channel=wrong,
                        q_cross_layer=distribution(captured[(block+1)%12][0,1:,c].square()),cross_layer=(block+1)%12,
                        q_permuted=None if q_correct is None else np.asarray(q_correct)[rng.permutation(196)].tolist(),
                        q_random_channel=distribution(h[:,random_channel].square()),random_channel=random_channel))
            return rows
    finally:
        for handle in handles:handle.remove()
