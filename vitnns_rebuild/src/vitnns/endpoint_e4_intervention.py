"""ENDPOINT-E4 local-direction and finite activation-replacement kernels.

The implementation operates on the registered unfused timm ViT edges.  A
replacement forward always starts from the clean input, replaces exactly one
registered component (or the registered routing/value joint pair), and then
recomputes the native downstream graph.  Reference activations come from a
separate forward of the same image under the frozen input reference.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .endpoint_nns import _frozen_eval


ETA_VALUES = (0.01, 0.003, 0.001, 0.0003)
ANALYTIC_JVP_ATOL = 1e-5
ANALYTIC_JVP_RTOL = 1e-4
CONDITION_SCALED_JVP_ATOL = 1e-5
CONDITION_SCALED_JVP_RTOL = 1e-5
MAIN_FAMILIES = ("heads", "routing_value", "residual_branches")
EDGE_NAMES = ("heads", "routing", "value", "query", "key", "attention", "mlp")


class EndpointE4Error(ValueError):
    """The E4 mathematical or execution contract was violated."""


@dataclass
class EdgeCapture:
    logits: Tensor
    score: Tensor
    edges: dict[int, dict[str, Tensor]]


def component_rows(num_heads: int, *, include_secondary: bool = False) -> list[dict[str, Any]]:
    if num_heads <= 0:
        raise EndpointE4Error("HEAD_COUNT")
    rows: list[dict[str, Any]] = []
    for block in range(12):
        for head in range(num_heads):
            rows.append({
                "component_id": f"b{block:02d}.head_{head:02d}", "family": "heads",
                "block": block, "operation": "head", "head": head,
                "num_heads": num_heads,
                "e3_component_keys": [f"head_{head}"],
            })
        rows.extend((
            {"component_id": f"b{block:02d}.routing", "family": "routing_value",
             "block": block, "operation": "routing", "head": None,
             "e3_component_keys": ["routing"]},
            {"component_id": f"b{block:02d}.value", "family": "routing_value",
             "block": block, "operation": "value", "head": None,
             "e3_component_keys": ["value"]},
            {"component_id": f"b{block:02d}.attention_branch", "family": "residual_branches",
             "block": block, "operation": "attention", "head": None,
             "e3_component_keys": ["attention"]},
            {"component_id": f"b{block:02d}.mlp_branch", "family": "residual_branches",
             "block": block, "operation": "mlp", "head": None,
             "e3_component_keys": ["mlp"]},
        ))
        if include_secondary:
            rows.extend((
                {"component_id": f"b{block:02d}.query", "family": "query_key_secondary",
                 "block": block, "operation": "query", "head": None,
                 "e3_component_keys": ["query"]},
                {"component_id": f"b{block:02d}.key", "family": "query_key_secondary",
                 "block": block, "operation": "key", "head": None,
                 "e3_component_keys": ["key"]},
                {"component_id": f"b{block:02d}.joint_routing_value",
                 "family": "joint_routing_value_secondary", "block": block,
                 "operation": "joint_routing_value", "head": None,
                 "e3_component_keys": ["routing", "value"]},
            ))
    return rows


def _check_model(model: nn.Module) -> None:
    from timm.models.vision_transformer import Attention, Block
    if len(model.blocks) != 12 or any(type(block) is not Block for block in model.blocks):
        raise EndpointE4Error("MODEL_BLOCK_CONTRACT")
    if any(type(block.attn) is not Attention or getattr(block.attn, "fused_attn", True) for block in model.blocks):
        raise EndpointE4Error("UNFUSED_ATTENTION_REQUIRED")
    if getattr(model, "grad_checkpointing", False):
        raise EndpointE4Error("GRAD_CHECKPOINTING_DISABLED")


@contextmanager
def _explicit_attention_runtime(
    model: nn.Module, captures: dict[int, dict[str, Tensor]] | None = None,
    modifications: Sequence[Mapping[str, Any]] = (),
) -> Iterator[None]:
    """Expose registered attention edges and apply batch-indexed replacements."""
    originals = []
    by_block: dict[int, list[Mapping[str, Any]]] = {}
    for modification in modifications:
        by_block.setdefault(int(modification["block"]), []).append(modification)
    try:
        for block_index, block in enumerate(model.blocks):
            attention = block.attn
            original = attention.forward
            had_instance = "forward" in attention.__dict__
            originals.append((attention, original, had_instance))

            def explicit_forward(module: nn.Module, value: Tensor, block_index=block_index) -> Tensor:
                batch, tokens, channels = value.shape
                qkv_projection = module.qkv(value)
                qkv = qkv_projection.reshape(
                    batch, tokens, 3, module.num_heads, module.head_dim,
                ).permute(2, 0, 3, 1, 4)
                query_projection, key_projection, value_projection = qkv.unbind(0)
                rows = by_block.get(block_index, ())
                if rows:
                    query_projection = query_projection.clone()
                    key_projection = key_projection.clone()
                    value_projection = value_projection.clone()
                    for row in rows:
                        index, operation = int(row["batch_index"]), row["operation"]
                        replacement = row["replacement"]
                        if operation == "query":
                            query_projection[index] = replacement[0].to(query_projection)
                        elif operation == "key":
                            key_projection[index] = replacement[0].to(key_projection)
                        elif operation in {"value", "joint_routing_value"}:
                            value_projection[index] = replacement["value"][0].to(value_projection) if isinstance(replacement, dict) else replacement[0].to(value_projection)
                query = module.q_norm(query_projection)
                key = module.k_norm(key_projection)
                scores = (query * module.scale) @ key.transpose(-2, -1)
                probabilities = module.attn_drop(scores.softmax(dim=-1))
                if rows:
                    probabilities = probabilities.clone()
                    for row in rows:
                        if row["operation"] in {"routing", "joint_routing_value"}:
                            replacement = row["replacement"]
                            probabilities[int(row["batch_index"])] = (
                                replacement["routing"][0] if isinstance(replacement, dict) else replacement[0]
                            ).to(probabilities)
                av = probabilities @ value_projection
                projection_input = av.transpose(1, 2).reshape(batch, tokens, channels)
                if rows:
                    projection_input = projection_input.clone()
                    for row in rows:
                        if row["operation"] == "head":
                            index, head = int(row["batch_index"]), int(row["head"])
                            lo, hi = head * module.head_dim, (head + 1) * module.head_dim
                            projection_input[index, :, lo:hi] = row["replacement"][0].to(projection_input)
                if captures is not None:
                    if block_index in captures:
                        raise EndpointE4Error("ATTENTION_CAPTURE_DUPLICATE")
                    captures[block_index] = {
                        "heads": projection_input, "routing": probabilities,
                        "value": value_projection, "query": query_projection,
                        "key": key_projection,
                    }
                return module.proj_drop(module.proj(projection_input))

            attention.forward = MethodType(explicit_forward, attention)
        yield
    finally:
        for attention, original, had_instance in reversed(originals):
            if had_instance:
                attention.forward = original
            else:
                del attention.forward


@contextmanager
def _branch_runtime(
    model: nn.Module, captures: dict[int, dict[str, Tensor]] | None = None,
    modifications: Sequence[Mapping[str, Any]] = (),
) -> Iterator[None]:
    handles = []
    by_edge: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in modifications:
        if row["operation"] in {"attention", "mlp"}:
            by_edge.setdefault((int(row["block"]), row["operation"]), []).append(row)

    def hook(block_index: int, operation: str):
        def callback(_module, _args, output):
            if captures is not None:
                captures.setdefault(block_index, {})[operation] = output
            rows = by_edge.get((block_index, operation), ())
            if not rows:
                return None
            changed = output.clone()
            for row in rows:
                changed[int(row["batch_index"])] = row["replacement"][0].to(changed)
            return changed
        return callback

    try:
        for block_index, block in enumerate(model.blocks):
            handles.append(block.drop_path1.register_forward_hook(hook(block_index, "attention")))
            handles.append(block.drop_path2.register_forward_hook(hook(block_index, "mlp")))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _forward_registered_edges(model: nn.Module, inputs: Tensor, *, class_id: int) -> EdgeCapture:
    edges: dict[int, dict[str, Tensor]] = {}
    with _explicit_attention_runtime(model, edges), _branch_runtime(model, edges):
        logits = model(inputs)
    score = logits[:, int(class_id)]
    if set(edges) != set(range(12)) or any(set(row) != set(EDGE_NAMES) for row in edges.values()):
        raise EndpointE4Error("EDGE_CAPTURE_INCOMPLETE")
    return EdgeCapture(logits=logits, score=score, edges=edges)


def capture_registered_edges(
    model: nn.Module, inputs: Tensor, *, class_id: int, enable_grad: bool,
) -> EdgeCapture:
    _check_model(model)
    if inputs.shape != (1, 3, 224, 224) or inputs.dtype is not torch.float32:
        raise EndpointE4Error("INPUT_CONTRACT")
    if not 0 <= int(class_id) < 1000 or not bool(torch.isfinite(inputs).all()):
        raise EndpointE4Error("INPUT_OR_TARGET")
    value = inputs.detach().clone().requires_grad_(enable_grad)
    context = torch.enable_grad() if enable_grad else torch.no_grad()
    with _frozen_eval(model), context:
        capture = _forward_registered_edges(model, value, class_id=class_id)
    if not bool(torch.isfinite(capture.logits).all()) or any(
        not bool(torch.isfinite(tensor).all()) for row in capture.edges.values() for tensor in row.values()
    ):
        raise EndpointE4Error("NONFINITE_CAPTURE")
    return capture


def edge_gradients(capture: EdgeCapture) -> dict[int, dict[str, Tensor]]:
    ordered = [(block, edge) for block in range(12) for edge in EDGE_NAMES]
    tensors = tuple(capture.edges[block][edge] for block, edge in ordered)
    gradients = torch.autograd.grad(capture.score.sum(), tensors, retain_graph=False)
    result: dict[int, dict[str, Tensor]] = {block: {} for block in range(12)}
    for (block, edge), gradient in zip(ordered, gradients):
        if not bool(torch.isfinite(gradient).all()):
            raise EndpointE4Error("NONFINITE_EDGE_GRADIENT")
        result[block][edge] = gradient.detach()
    return result


def directional_edges(model: nn.Module, inputs: Tensor, *, class_id: int) -> dict[int, dict[str, Tensor]]:
    """Compute J_z,x x for every registered edge in one formal JVP."""
    order = [(block, edge) for block in range(12) for edge in EDGE_NAMES]

    def callback(value: Tensor):
        capture = _forward_registered_edges(model, value, class_id=class_id)
        return tuple(capture.edges[block][edge] for block, edge in order)

    with _frozen_eval(model), torch.enable_grad():
        _, tangent = torch.autograd.functional.jvp(
            callback, inputs.detach(), inputs.detach(), create_graph=False, strict=True,
        )
    result: dict[int, dict[str, Tensor]] = {block: {} for block in range(12)}
    for (block, edge), value in zip(order, tangent):
        if not bool(torch.isfinite(value).all()):
            raise EndpointE4Error("NONFINITE_EDGE_JVP")
        result[block][edge] = value.detach()
    return result


def component_tensor(edges: Mapping[str, Tensor], row: Mapping[str, Any]) -> Tensor | dict[str, Tensor]:
    operation = row["operation"]
    if operation == "head":
        heads = edges["heads"]; width = heads.shape[-1] // int(row["num_heads"])
        lo, hi = int(row["head"]) * width, (int(row["head"]) + 1) * width
        return heads[:, :, lo:hi]
    if operation == "joint_routing_value":
        return {"routing": edges["routing"], "value": edges["value"]}
    return edges[operation]


def dot64(left: Tensor | dict[str, Tensor], right: Tensor | dict[str, Tensor]) -> float:
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise EndpointE4Error("JOINT_TENSOR_KEYS")
        return float(sum((left[key].detach().double() * right[key].detach().double()).sum() for key in sorted(left)))
    if isinstance(right, dict):
        raise EndpointE4Error("TENSOR_KIND_MISMATCH")
    return float((left.detach().double() * right.detach().double()).sum())


def absolute_dot_mass64(
    left: Tensor | dict[str, Tensor], right: Tensor | dict[str, Tensor],
) -> float:
    """Return sum(abs(left * right)) in float64 for a dot-product error scale."""
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise EndpointE4Error("JOINT_TENSOR_KEYS")
        return float(sum(
            (left[key].detach().double() * right[key].detach().double()).abs().sum()
            for key in sorted(left)
        ))
    if isinstance(right, dict):
        raise EndpointE4Error("TENSOR_KIND_MISMATCH")
    return float((left.detach().double() * right.detach().double()).abs().sum())


def norm64(value: Tensor | dict[str, Tensor]) -> float:
    if isinstance(value, dict):
        return float(sum(tensor.detach().double().square().sum() for tensor in value.values()).sqrt())
    return float(value.detach().double().norm())


def subtract(left: Tensor | dict[str, Tensor], right: Tensor | dict[str, Tensor]) -> Tensor | dict[str, Tensor]:
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise EndpointE4Error("JOINT_TENSOR_KEYS")
        return {key: left[key].detach() - right[key].detach() for key in left}
    if isinstance(right, dict):
        raise EndpointE4Error("TENSOR_KIND_MISMATCH")
    return left.detach() - right.detach()


def shifted(clean: Tensor | dict[str, Tensor], direction: Tensor | dict[str, Tensor], eta: float) -> Tensor | dict[str, Tensor]:
    if isinstance(clean, dict):
        if not isinstance(direction, dict):
            raise EndpointE4Error("JOINT_TENSOR_KEYS")
        return {key: clean[key].detach() - eta * direction[key].detach() for key in clean}
    if isinstance(direction, dict):
        raise EndpointE4Error("TENSOR_KIND_MISMATCH")
    return clean.detach() - eta * direction.detach()


def zero_like(value: Tensor | dict[str, Tensor]) -> Tensor | dict[str, Tensor]:
    if isinstance(value, dict):
        return {key: torch.zeros_like(tensor) for key, tensor in value.items()}
    return torch.zeros_like(value)


def intervention_scores(
    model: nn.Module, inputs: Tensor, *, class_id: int, candidates: Sequence[Mapping[str, Any]],
    replacements: Mapping[str, Tensor | dict[str, Tensor]],
) -> tuple[float, list[float], float]:
    """Run one batched clean control followed by component replacements."""
    if not candidates:
        raise EndpointE4Error("EMPTY_INTERVENTION_BATCH")
    batch = inputs.detach().repeat(len(candidates) + 1, 1, 1, 1)
    modifications = []
    for index, row in enumerate(candidates, 1):
        component_id = row["component_id"]
        if component_id not in replacements:
            raise EndpointE4Error("MISSING_REPLACEMENT:" + component_id)
        modifications.append({**row, "batch_index": index, "replacement": replacements[component_id]})
    with _frozen_eval(model), torch.no_grad(), _explicit_attention_runtime(model, modifications=modifications), _branch_runtime(model, modifications=modifications):
        logits = model(batch)
    scores = logits[:, int(class_id)]
    if not bool(torch.isfinite(scores).all()):
        raise EndpointE4Error("NONFINITE_INTERVENTION_SCORE")
    baseline = float(scores[0]); values = [float(value) for value in scores[1:]]
    return baseline, values, float(logits.abs().max())


def average_ranks(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise EndpointE4Error("RANK_INPUT")
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=np.float64)
    start = 0
    while start < len(data):
        stop = start + 1
        while stop < len(data) and data[order[stop]] == data[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float], *, minimum: int = 8) -> dict[str, Any]:
    if len(left) != len(right) or len(left) < minimum:
        return {"value": None, "reason": "NA_INSUFFICIENT_COMPONENTS", "n": len(left)}
    x, y = average_ranks(left), average_ranks(right)
    x -= x.mean(); y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        return {"value": None, "reason": "NA_CONSTANT_VECTOR", "n": len(left)}
    return {"value": float(np.dot(x, y) / denominator), "reason": None, "n": len(left)}


def _detach_edges(edges: Mapping[int, Mapping[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
    return {block: {key: value.detach() for key, value in row.items()} for block, row in edges.items()}


def _component_map(decomposition: Any, row: Mapping[str, Any]) -> Tensor:
    values = [decomposition.components[key] for key in row["e3_component_keys"]]
    return sum(values[1:], values[0]) if len(values) > 1 else values[0]


def _component_summaries(
    model: nn.Module, inputs: Tensor, *, class_id: int, rows: Sequence[Mapping[str, Any]],
    progress=None,
) -> dict[str, dict[str, float]]:
    from .endpoint_vit import EndpointTarget, decompose_endpoint_vit
    by_block: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(int(row["block"]), []).append(row)
    summaries: dict[str, dict[str, float]] = {}
    target = EndpointTarget("ground_truth_logit", "", kind="ground_truth_logit", class_id=class_id)
    for block in range(12):
        decomposition = decompose_endpoint_vit(
            model, inputs, block=block, target=target, atol=1e-5, rtol=1e-4,
        )
        for row in by_block[block]:
            value = _component_map(decomposition, row).detach().double()
            summaries[row["component_id"]] = {
                "a": float(value.sum()), "m": float(value.abs().sum()),
                "map_l2": float(value.norm()),
            }
        if progress is not None:
            progress()
    return summaries


def _run_in_chunks(
    model: nn.Module, inputs: Tensor, *, class_id: int, rows: Sequence[Mapping[str, Any]],
    replacements: Mapping[str, Tensor | dict[str, Tensor]], chunk_size: int, progress=None,
) -> tuple[float, dict[str, float], float]:
    observed_clean: float | None = None
    outputs: dict[str, float] = {}
    maximum_logit = 0.0
    for start in range(0, len(rows), chunk_size):
        part = rows[start:start + chunk_size]
        clean, values, current_max = intervention_scores(
            model, inputs, class_id=class_id, candidates=part, replacements=replacements,
        )
        if observed_clean is None:
            observed_clean = clean
        elif abs(clean - observed_clean) > 1e-5 + 1e-4 * abs(observed_clean):
            raise EndpointE4Error("CLEAN_SCORE_BATCH_DRIFT")
        outputs.update((row["component_id"], value) for row, value in zip(part, values))
        maximum_logit = max(maximum_logit, current_max)
        if progress is not None:
            progress()
    if observed_clean is None or set(outputs) != {row["component_id"] for row in rows}:
        raise EndpointE4Error("INTERVENTION_COVERAGE")
    return observed_clean, outputs, maximum_logit


def _random_summary(task_seed: int, family: str, reference: str, effect: Sequence[float]) -> dict[str, Any]:
    rows = []
    for repeat in range(20):
        payload = f"NNS_VIT_ENDPOINT_E4_RANDOM_V1|{task_seed}|{reference}|{family}|{repeat}".encode()
        seed = int.from_bytes(__import__("hashlib").sha256(payload).digest()[:8], "big")
        predictor = np.random.Generator(np.random.PCG64(seed)).random(len(effect))
        rows.append({"repeat": repeat, "seed": seed, **spearman(predictor, effect)})
    valid = [row["value"] for row in rows if row["value"] is not None]
    return {"repeat_count": 20, "mean_spearman": None if not valid else float(np.mean(valid)), "repeats": rows}


def scan_e4_task(
    model: nn.Module, inputs: Tensor, references: Mapping[str, Tensor], *, class_id: int,
    task_seed: int, include_local: bool, include_secondary: bool, chunk_size: int = 8,
    local_contract_mode: str = "finite_eta_v1", progress=None,
) -> dict[str, Any]:
    """Compute the complete E4 record for one image-model task."""
    _check_model(model)
    if set(references) != {"normalized_zero", "blur_rgb"} or chunk_size <= 0:
        raise EndpointE4Error("REFERENCE_OR_CHUNK_CONTRACT")
    if local_contract_mode not in {
        "finite_eta_v1", "analytic_jvp_identity_r2", "condition_scaled_jvp_r3",
    }:
        raise EndpointE4Error("LOCAL_CONTRACT_MODE")
    num_heads = int(model.blocks[0].attn.num_heads)
    if any(int(block.attn.num_heads) != num_heads for block in model.blocks):
        raise EndpointE4Error("HEAD_COUNT_DRIFT")
    main_rows = component_rows(num_heads)
    all_rows = component_rows(num_heads, include_secondary=include_secondary)
    summaries = _component_summaries(
        model, inputs, class_id=class_id, rows=all_rows, progress=progress,
    )

    clean_capture = capture_registered_edges(model, inputs, class_id=class_id, enable_grad=True)
    clean_score = float(clean_capture.score.detach())
    gradients = edge_gradients(clean_capture)
    clean_edges = _detach_edges(clean_capture.edges)
    clean_predicted_class = int(clean_capture.logits.detach().argmax())
    reference_edges = {
        name: _detach_edges(capture_registered_edges(
            model, value, class_id=class_id, enable_grad=False,
        ).edges)
        for name, value in references.items()
    }
    del clean_capture

    for row in all_rows:
        block = int(row["block"]); component_id = row["component_id"]
        z_clean = component_tensor(clean_edges[block], row)
        g_clean = component_tensor(gradients[block], row)
        summaries[component_id].update({
            "output_norm": norm64(z_clean), "gradient_norm": norm64(g_clean),
            "gradient_activation": dot64(g_clean, z_clean),
        })

    local = {"performed": include_local, "components": [], "all_pass": None}
    if include_local:
        directions = directional_edges(model, inputs, class_id=class_id)
        noop_replacements: dict[str, Tensor | dict[str, Tensor]] = {}
        for row in all_rows:
            block = int(row["block"]); component_id = row["component_id"]
            clean_z = component_tensor(clean_edges[block], row)
            noop_replacements[component_id] = clean_z
        noop_clean, noop_scores, _ = _run_in_chunks(
            model, inputs, class_id=class_id, rows=all_rows, replacements=noop_replacements,
            chunk_size=chunk_size, progress=progress,
        )
        noop_max_abs = max(abs(noop_clean - value) for value in noop_scores.values())
        if noop_max_abs > 1e-4 + 1e-4 * abs(noop_clean):
            raise EndpointE4Error("NOOP_REPLACEMENT_PARITY")
        local_clean = noop_clean
        local_scores: dict[str, float] = {}
        if local_contract_mode == "finite_eta_v1":
            expanded = []
            replacement_map = {}
            for row in all_rows:
                block = int(row["block"]); clean_z = component_tensor(clean_edges[block], row)
                direction = component_tensor(directions[block], row)
                for eta in ETA_VALUES:
                    expanded_id = f"{row['component_id']}@{eta}"
                    expanded.append({**row, "component_id": expanded_id})
                    replacement_map[expanded_id] = shifted(clean_z, direction, eta)
            local_clean, local_scores, _ = _run_in_chunks(
                model, inputs, class_id=class_id, rows=expanded, replacements=replacement_map,
                chunk_size=chunk_size, progress=progress,
            )
            if abs(local_clean - clean_score) > 1e-5 + 1e-4 * abs(clean_score):
                raise EndpointE4Error("LOCAL_CLEAN_SCORE_DRIFT")
        for row in all_rows:
            component_id = row["component_id"]; a_value = summaries[component_id]["a"]
            block = int(row["block"])
            directional_prediction = dot64(
                component_tensor(gradients[block], row), component_tensor(directions[block], row),
            )
            direction_error = abs(directional_prediction - a_value)
            if local_contract_mode == "finite_eta_v1":
                curve = []
                for eta in ETA_VALUES:
                    estimate = (local_clean - local_scores[f"{component_id}@{eta}"]) / eta
                    curve.append({"eta": eta, "estimate": estimate, "absolute_error": abs(estimate - a_value)})
                threshold = 1e-4 + 0.01 * abs(a_value)
                row_pass = (all(np.isfinite(item["estimate"]) for item in curve) and
                            min(item["absolute_error"] for item in curve) <= threshold and
                            direction_error <= threshold)
                detail = {"curve": curve, "minimum_absolute_error": min(item["absolute_error"] for item in curve)}
            elif local_contract_mode == "analytic_jvp_identity_r2":
                threshold = ANALYTIC_JVP_ATOL + ANALYTIC_JVP_RTOL * abs(a_value)
                row_pass = np.isfinite(directional_prediction) and direction_error <= threshold
                detail = {"arithmetic": "float64_dot_of_float32_autodiff_tensors",
                    "atol": ANALYTIC_JVP_ATOL, "rtol": ANALYTIC_JVP_RTOL}
            else:
                absolute_dot_mass = absolute_dot_mass64(
                    component_tensor(gradients[block], row),
                    component_tensor(directions[block], row),
                )
                threshold = CONDITION_SCALED_JVP_ATOL + CONDITION_SCALED_JVP_RTOL * absolute_dot_mass
                row_pass = (np.isfinite(directional_prediction) and
                            np.isfinite(absolute_dot_mass) and direction_error <= threshold)
                detail = {
                    "arithmetic": "float64_dot_of_float32_autodiff_tensors",
                    "absolute_dot_mass": absolute_dot_mass,
                    "atol": CONDITION_SCALED_JVP_ATOL,
                    "rtol_absolute_dot_mass": CONDITION_SCALED_JVP_RTOL,
                }
            local["components"].append({
                "component_id": component_id, "a": a_value,
                "directional_prediction": directional_prediction,
                "directional_prediction_absolute_error": direction_error,
                "threshold": threshold, "pass": bool(row_pass), **detail,
            })
        local["method"] = local_contract_mode
        local["noop_max_abs"] = noop_max_abs
        local["all_pass"] = all(row["pass"] for row in local["components"])
        if not local["all_pass"]:
            raise EndpointE4Error("LOCAL_DIRECTION_CONTRACT_FAILED")

    finite: dict[str, Any] = {}
    for reference_id in ("normalized_zero", "blur_rgb"):
        replacements = {
            row["component_id"]: component_tensor(reference_edges[reference_id][int(row["block"])], row)
            for row in all_rows
        }
        observed_clean, replaced, maximum_logit = _run_in_chunks(
            model, inputs, class_id=class_id, rows=all_rows, replacements=replacements,
            chunk_size=chunk_size, progress=progress,
        )
        if abs(observed_clean - clean_score) > 1e-5 + 1e-4 * abs(clean_score):
            raise EndpointE4Error("FINITE_CLEAN_SCORE_DRIFT")
        component_outputs = []
        for row in all_rows:
            block = int(row["block"]); component_id = row["component_id"]
            z_clean = component_tensor(clean_edges[block], row)
            z_reference = component_tensor(reference_edges[reference_id][block], row)
            gradient = component_tensor(gradients[block], row)
            effect = clean_score - replaced[component_id]
            component_outputs.append({
                **row, **summaries[component_id], "reference_id": reference_id,
                "reference_output_norm": norm64(z_reference),
                "reference_displacement_norm": norm64(subtract(z_clean, z_reference)),
                "p_ref": dot64(gradient, subtract(z_clean, z_reference)),
                "replacement_score": replaced[component_id], "effect": effect,
                "absolute_effect": abs(effect),
                "direction_agreement_a": None if abs(effect) <= 1e-6 else bool(np.sign(summaries[component_id]["a"]) == np.sign(effect)),
            })
        metrics = {}
        for family in MAIN_FAMILIES:
            family_rows = [row for row in component_outputs if row["family"] == family]
            absolute_effect = [row["absolute_effect"] for row in family_rows]
            metrics[family] = {
                "component_count": len(family_rows),
                "m_vs_absolute_effect": spearman([row["m"] for row in family_rows], absolute_effect),
                "a_vs_effect": spearman([row["a"] for row in family_rows], [row["effect"] for row in family_rows]),
                "p_ref_vs_effect": spearman([row["p_ref"] for row in family_rows], [row["effect"] for row in family_rows]),
                "abs_p_ref_vs_absolute_effect": spearman([abs(row["p_ref"]) for row in family_rows], absolute_effect),
                "output_norm_vs_absolute_effect": spearman([row["output_norm"] for row in family_rows], absolute_effect),
                "gradient_norm_vs_absolute_effect": spearman([row["gradient_norm"] for row in family_rows], absolute_effect),
                "abs_gradient_activation_vs_absolute_effect": spearman([abs(row["gradient_activation"]) for row in family_rows], absolute_effect),
                "random_20": _random_summary(task_seed, family, reference_id, absolute_effect),
            }
        finite[reference_id] = {
            "components": component_outputs, "per_image_metrics": metrics,
            "clean_score": observed_clean, "maximum_absolute_logit": maximum_logit,
        }

    zeroing: dict[str, Any] = {"performed": include_secondary, "components": []}
    if include_secondary:
        zero_rows = [row for row in main_rows if row["operation"] in {"head", "attention", "mlp"}]
        zeros = {
            row["component_id"]: zero_like(component_tensor(clean_edges[int(row["block"])], row))
            for row in zero_rows
        }
        zero_clean, zero_scores, _ = _run_in_chunks(
            model, inputs, class_id=class_id, rows=zero_rows, replacements=zeros,
            chunk_size=chunk_size, progress=progress,
        )
        zeroing["clean_score"] = zero_clean
        zeroing["components"] = [{
            "component_id": row["component_id"], "operation": row["operation"],
            "block": row["block"], "head": row["head"],
            "zero_score": zero_scores[row["component_id"]],
            "zero_effect": zero_clean - zero_scores[row["component_id"]],
        } for row in zero_rows]

    return {
        "clean_score": clean_score, "clean_predicted_class": clean_predicted_class,
        "num_heads": num_heads, "main_component_count": len(main_rows),
        "all_component_count": len(all_rows), "include_local": include_local,
        "include_secondary": include_secondary, "local_contract": local,
        "finite_utility": finite, "zeroing_secondary": zeroing,
    }


def validate_run_spec(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "stage", "authorization_token", "predecessors", "cohort",
        "models", "target", "local_contract", "finite_utility", "method_scope",
        "runtime", "outputs", "failure",
    }
    if set(value) != required or value["schema_version"] != 1 or value["stage"] != "ENDPOINT-E4":
        raise EndpointE4Error("RUN_SPEC_SCHEMA")
    if value["authorization_token"] != "ENDPOINT_E4_EXPLICIT_USER_REQUEST_20260918":
        raise EndpointE4Error("AUTHORIZATION")
    if value["cohort"] != {
        "split": "evaluation", "count": 400, "models_share_rows": True,
        "source": "ENDPOINT-E2_authoritative_materialization", "replacement_allowed": False,
    }:
        raise EndpointE4Error("COHORT")
    if value["models"] != ["vit_b16", "deit_s16"]:
        raise EndpointE4Error("MODELS")
    if value["target"] != {"kind": "ground_truth_logit", "class_source": "frozen_EP0_class_id"}:
        raise EndpointE4Error("TARGET")
    local = value["local_contract"]
    if local != {
        "evaluation_orders": list(range(16)), "eta": list(ETA_VALUES),
        "direction": "J_zx_times_x_at_clean_endpoint",
        "acceptance": "minimum_abs_error_le_1e-4_plus_0.01_abs_a_all_finite",
        "components": "all_main_plus_query_key_joint_all12blocks",
        "claim_scope": "CORRECTNESS_ONLY_NOT_C4_UTILITY",
    }:
        raise EndpointE4Error("LOCAL_CONTRACT")
    utility = value["finite_utility"]
    if utility != {
        "references": ["normalized_zero", "blur_rgb"],
        "effect": "F_clean_minus_F_single_component_reference_replacement",
        "main_components": "heads_routing_value_residual_branches_all12blocks_all400",
        "secondary_components": "query_key_joint_first64_evaluation_both_references",
        "zeroing": "heads_attention_branch_mlp_branch_first64_evaluation",
        "primary_predictor": "endpoint_absolute_mass_m_vs_absolute_effect",
        "signed_secondary": "endpoint_signed_sum_a_vs_signed_effect",
        "mandatory_descriptive_comparators": ["p_ref", "output_norm", "gradient_norm", "gradient_activation", "random_20"],
        "statistics": "per_image_spearman_only_global_bootstrap_holm_and_claim_decisions_deferred_to_ENDPOINT_E6",
    }:
        raise EndpointE4Error("FINITE_UTILITY")
    if value["method_scope"] != {
        "local_direction_allowed": True, "activation_replacement_allowed": True,
        "c4a_per_image_metrics_allowed": True, "c4b_spatial_intervention_allowed": False,
        "path_ig_allowed": False, "input_fidelity_allowed": False,
        "input_stability_allowed": False, "global_holm_or_final_claim_decision_allowed": False,
    }:
        raise EndpointE4Error("METHOD_SCOPE")
    runtime = value["runtime"]
    expected_runtime = {
        "gpu_hours_cap": 136, "wall_hours_cap": 132, "incremental_retained_gib_cap": 56,
        "max_gpus": 8, "min_gpus": 1, "gpu_type": "NVIDIA H20",
        "min_free_mib": 32768, "max_utilization_percent": 10,
        "gpu_memory_cap_gib": 24, "cpu_rss_cap_gib": 32, "minimum_free_disk_gib": 50,
        "task_timeout_seconds": 1800, "claim_lease_seconds": 180, "heartbeat_seconds": 30,
        "max_attempts": 2, "dynamic_global_queue": True, "static_modulo_sharding": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "production_dtype": "float32",
        "tf32": False, "autocast": False, "deterministic_algorithms": True,
    }
    if runtime != expected_runtime:
        raise EndpointE4Error("RUNTIME")
    if value["failure"] != {
        "missing_duplicate_nonfinite_source_drift_local_contract_or_cap": "FAIL_CLOSED_PRESERVE_ROOT",
        "resource_cap": "PAUSED_RESOURCE_CAP",
        "next_on_valid_execution": "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST",
    }:
        raise EndpointE4Error("FAILURE")


def validate_recovery_run_spec(value: Mapping[str, Any]) -> None:
    """Validate the prospective R2 recovery contract without accepting drift."""
    required = {
        "schema_version", "stage", "authorization_token", "recovery",
        "predecessors", "cohort", "models", "target", "local_contract",
        "finite_utility", "method_scope", "runtime", "outputs", "failure",
    }
    if set(value) != required or value["schema_version"] != 2 or value["stage"] != "ENDPOINT-E4-R2-RECOVERY":
        raise EndpointE4Error("RECOVERY_RUN_SPEC_SCHEMA")
    if value["authorization_token"] != "ENDPOINT_E4_RECOVERY_EXPLICIT_USER_REQUEST_20260918":
        raise EndpointE4Error("RECOVERY_AUTHORIZATION")
    if value["recovery"] != {
        "source_root": "reports/nns_vit_endpoint_e4_intervention_r1_20260918",
        "source_status": "ENDPOINT_E4_FAIL_LOCAL_DIRECTION_CONTRACT",
        "source_immutable": True, "inherit_partial_results": False,
        "outcome_blind_contract_freeze": True,
    }:
        raise EndpointE4Error("RECOVERY_BOUNDARY")
    if set(value["predecessors"]) != {
        "cohort_registry_sha256", "endpoint_e3_r1_artifact_manifest_sha256",
        "endpoint_e3_r2_terminal_sha256", "endpoint_e3_r2_validation_sha256",
        "endpoint_e4_r1_artifact_manifest_sha256", "endpoint_e4_r1_coverage_sha256",
        "endpoint_e4_r1_terminal_sha256", "endpoint_e4_r1_validation_sha256",
        "intervention_registry_sha256", "materialization_receipt_sha256",
        "model_registry_sha256", "statistical_analysis_plan_sha256",
        "target_registry_sha256", "unified_plan_sha256",
    } or not all(isinstance(item, str) and len(item) == 64 for item in value["predecessors"].values()):
        raise EndpointE4Error("RECOVERY_PREDECESSORS")
    if value["cohort"] != {
        "split": "evaluation", "count": 400, "models_share_rows": True,
        "source": "ENDPOINT-E2_authoritative_materialization", "replacement_allowed": False,
    } or value["models"] != ["vit_b16", "deit_s16"]:
        raise EndpointE4Error("RECOVERY_COHORT_OR_MODELS")
    if value["target"] != {"kind": "ground_truth_logit", "class_source": "frozen_EP0_class_id"}:
        raise EndpointE4Error("RECOVERY_TARGET")
    if value["local_contract"] != {
        "evaluation_orders": list(range(16)), "method": "analytic_jvp_identity_r2",
        "direction": "J_zx_times_x_at_clean_endpoint",
        "arithmetic": "float64_dot_of_float32_autodiff_tensors",
        "acceptance": "absolute_error_le_1e-5_plus_1e-4_abs_a_all_finite",
        "components": "all_main_plus_query_key_joint_all12blocks",
        "noop_replacement_parity_required": True,
        "finite_eta_curve": "REMOVED_FROM_GATE_HISTORICAL_R1_FAILURE_PRESERVED",
        "claim_scope": "CORRECTNESS_ONLY_NOT_C4_UTILITY",
    }:
        raise EndpointE4Error("RECOVERY_LOCAL_CONTRACT")
    if value["finite_utility"] != {
        "references": ["normalized_zero", "blur_rgb"],
        "effect": "F_clean_minus_F_single_component_reference_replacement",
        "main_components": "heads_routing_value_residual_branches_all12blocks_all400",
        "secondary_components": "query_key_joint_first64_evaluation_both_references",
        "zeroing": "heads_attention_branch_mlp_branch_first64_evaluation",
        "primary_predictor": "endpoint_absolute_mass_m_vs_absolute_effect",
        "signed_secondary": "endpoint_signed_sum_a_vs_signed_effect",
        "mandatory_descriptive_comparators": ["p_ref", "output_norm", "gradient_norm", "gradient_activation", "random_20"],
        "statistics": "per_image_spearman_only_global_bootstrap_holm_and_claim_decisions_deferred_to_ENDPOINT_E6",
    }:
        raise EndpointE4Error("RECOVERY_FINITE_UTILITY")
    if value["method_scope"] != {
        "activation_replacement_allowed": True, "c4a_per_image_metrics_allowed": True,
        "c4b_spatial_intervention_allowed": False,
        "global_holm_or_final_claim_decision_allowed": False,
        "input_fidelity_allowed": False, "input_stability_allowed": False,
        "local_direction_allowed": True, "path_ig_allowed": False,
    }:
        raise EndpointE4Error("RECOVERY_METHOD_SCOPE")
    expected_runtime = {
        "gpu_hours_cap": 136, "wall_hours_cap": 132, "incremental_retained_gib_cap": 56,
        "max_gpus": 8, "min_gpus": 1, "gpu_type": "NVIDIA H20",
        "min_free_mib": 32768, "max_utilization_percent": 10,
        "gpu_memory_cap_gib": 24, "cpu_rss_cap_gib": 32, "minimum_free_disk_gib": 50,
        "task_timeout_seconds": 1800, "claim_lease_seconds": 180, "heartbeat_seconds": 30,
        "max_attempts": 2, "dynamic_global_queue": True, "static_modulo_sharding": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "production_dtype": "float32",
        "tf32": False, "autocast": False, "deterministic_algorithms": True,
    }
    if value["runtime"] != expected_runtime:
        raise EndpointE4Error("RECOVERY_RUNTIME")
    if value["outputs"] != {
        "formal_root": "reports/nns_vit_endpoint_e4_intervention_r2_20260918",
        "source_materialization": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
    }:
        raise EndpointE4Error("RECOVERY_OUTPUTS")
    if value["failure"] != {
        "missing_duplicate_nonfinite_source_drift_local_contract_or_cap": "FAIL_CLOSED_PRESERVE_ROOT",
        "resource_cap": "PAUSED_RESOURCE_CAP",
        "next_on_valid_execution": "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST",
    }:
        raise EndpointE4Error("RECOVERY_FAILURE_POLICY")


def validate_recovery_r3_run_spec(value: Mapping[str, Any]) -> None:
    """Validate the closed R3 salvage-and-rerun contract."""
    required = {
        "schema_version", "stage", "authorization_token", "recovery",
        "predecessors", "cohort", "models", "target", "local_contract",
        "finite_utility", "method_scope", "runtime", "outputs", "failure",
    }
    if set(value) != required or value["schema_version"] != 3 or value["stage"] != "ENDPOINT-E4-R3-RECOVERY":
        raise EndpointE4Error("R3_RUN_SPEC_SCHEMA")
    if value["authorization_token"] != "ENDPOINT_E4_R3_RECOVERY_EXPLICIT_USER_REQUEST_20260918":
        raise EndpointE4Error("R3_AUTHORIZATION")
    if value["recovery"] != {
        "source_root": "reports/nns_vit_endpoint_e4_intervention_r2_20260918",
        "source_status": "ENDPOINT_E4_R2_FAIL_ANALYTIC_JVP_CONTRACT",
        "source_immutable": True, "inherited_evaluation_orders": list(range(16, 400)),
        "rerun_evaluation_orders": list(range(16)), "inherited_task_count": 768,
        "new_task_count": 32, "source_revalidation_required": True,
        "exact_union_required": True, "r1_results_allowed": False,
    }:
        raise EndpointE4Error("R3_RECOVERY_BOUNDARY")
    if set(value["predecessors"]) != {
        "cohort_registry_sha256", "endpoint_e3_r2_terminal_sha256",
        "endpoint_e4_r2_artifact_manifest_sha256", "endpoint_e4_r2_coverage_sha256",
        "endpoint_e4_r2_diagnostic_sha256", "endpoint_e4_r2_terminal_sha256",
        "endpoint_e4_r2_validation_sha256", "intervention_registry_sha256",
        "materialization_receipt_sha256", "model_registry_sha256",
        "statistical_analysis_plan_sha256", "target_registry_sha256",
        "unified_plan_sha256",
    } or not all(isinstance(item, str) and len(item) == 64 for item in value["predecessors"].values()):
        raise EndpointE4Error("R3_PREDECESSORS")
    if value["cohort"] != {
        "split": "evaluation", "count": 400, "models_share_rows": True,
        "source": "ENDPOINT-E2_authoritative_materialization", "replacement_allowed": False,
    } or value["models"] != ["vit_b16", "deit_s16"]:
        raise EndpointE4Error("R3_COHORT_OR_MODELS")
    if value["target"] != {"kind": "ground_truth_logit", "class_source": "frozen_EP0_class_id"}:
        raise EndpointE4Error("R3_TARGET")
    if value["local_contract"] != {
        "evaluation_orders": list(range(16)), "method": "condition_scaled_jvp_r3",
        "direction": "J_zx_times_x_at_clean_endpoint",
        "arithmetic": "float64_dot_of_float32_autodiff_tensors",
        "condition_scale": "sum_abs_gradient_times_direction_products",
        "acceptance": "absolute_error_le_1e-5_plus_1e-5_absolute_dot_mass_all_finite",
        "components": "all_main_plus_query_key_joint_all12blocks",
        "noop_replacement_parity_required": True,
        "finite_eta_curve": "REMOVED_FROM_GATE_HISTORICAL_R1_FAILURE_PRESERVED",
        "claim_scope": "CORRECTNESS_ONLY_NOT_C4_UTILITY",
    }:
        raise EndpointE4Error("R3_LOCAL_CONTRACT")
    if value["finite_utility"] != {
        "references": ["normalized_zero", "blur_rgb"],
        "effect": "F_clean_minus_F_single_component_reference_replacement",
        "main_components": "heads_routing_value_residual_branches_all12blocks_all400",
        "secondary_components": "query_key_joint_first64_evaluation_both_references",
        "zeroing": "heads_attention_branch_mlp_branch_first64_evaluation",
        "primary_predictor": "endpoint_absolute_mass_m_vs_absolute_effect",
        "signed_secondary": "endpoint_signed_sum_a_vs_signed_effect",
        "mandatory_descriptive_comparators": ["p_ref", "output_norm", "gradient_norm", "gradient_activation", "random_20"],
        "statistics": "per_image_spearman_only_global_bootstrap_holm_and_claim_decisions_deferred_to_ENDPOINT_E6",
    }:
        raise EndpointE4Error("R3_FINITE_UTILITY")
    if value["method_scope"] != {
        "activation_replacement_allowed": True, "c4a_per_image_metrics_allowed": True,
        "c4b_spatial_intervention_allowed": False,
        "global_holm_or_final_claim_decision_allowed": False,
        "input_fidelity_allowed": False, "input_stability_allowed": False,
        "local_direction_allowed": True, "path_ig_allowed": False,
    }:
        raise EndpointE4Error("R3_METHOD_SCOPE")
    expected_runtime = {
        "gpu_hours_cap": 136, "wall_hours_cap": 132, "incremental_retained_gib_cap": 56,
        "max_gpus": 8, "min_gpus": 1, "gpu_type": "NVIDIA H20",
        "min_free_mib": 32768, "max_utilization_percent": 10,
        "gpu_memory_cap_gib": 24, "cpu_rss_cap_gib": 32, "minimum_free_disk_gib": 50,
        "task_timeout_seconds": 1800, "claim_lease_seconds": 180, "heartbeat_seconds": 30,
        "max_attempts": 2, "dynamic_global_queue": True, "static_modulo_sharding": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "production_dtype": "float32",
        "tf32": False, "autocast": False, "deterministic_algorithms": True,
    }
    if value["runtime"] != expected_runtime:
        raise EndpointE4Error("R3_RUNTIME")
    if value["outputs"] != {
        "formal_root": "reports/nns_vit_endpoint_e4_intervention_r3_20260918",
        "source_materialization": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
    }:
        raise EndpointE4Error("R3_OUTPUTS")
    if value["failure"] != {
        "missing_duplicate_nonfinite_source_drift_local_contract_or_cap": "FAIL_CLOSED_PRESERVE_ROOT",
        "resource_cap": "PAUSED_RESOURCE_CAP",
        "next_on_valid_execution": "ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST",
    }:
        raise EndpointE4Error("R3_FAILURE_POLICY")
