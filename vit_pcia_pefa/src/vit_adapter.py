"""Second-order PEFA fields on registered timm ViT use ports.

The implementation keeps every result in the normalized RGB coordinate system.
It never materializes an input Hessian: source-coordinate contractions are
evaluated with JVP/VJP and can be streamed over an exact partition of pixels.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from vitnns.endpoint_nns import _frozen_eval
from vitnns.endpoint_vit import capture_unfused_attention_edges


ATOL = 1e-5
RTOL = 1e-4
NEAR_ZERO = 1e-6


class VitAdapterError(ValueError):
    """A frozen ViT port or numerical contract was violated."""


@dataclass(frozen=True)
class PortLayout:
    component_ids: tuple[str, ...]
    cuts: Mapping[str, tuple[str, ...]]


def port_layout(num_heads: int) -> PortLayout:
    if type(num_heads) is not int or num_heads <= 0:
        raise VitAdapterError("INVALID_HEAD_COUNT")
    heads = tuple(f"head_{index}" for index in range(num_heads))
    q = tuple(f"query_head_{index}" for index in range(num_heads))
    k = tuple(f"key_head_{index}" for index in range(num_heads))
    v = tuple(f"value_head_{index}" for index in range(num_heads))
    components = (
        "identity", "attention", "projection", *heads, *q, *k, *v,
        "carry", "mlp", "mlp_fc1", "mlp_gelu", "mlp_fc2",
    )
    cuts = {
        "RESIDUAL1_FULL": ("identity", "attention"),
        "HEADS_WITH_BYPASS_FULL": ("identity", *heads),
        "QKV_WITH_BYPASS_FULL": ("identity", *q, *k, *v),
        "ROUTING_VALUE_WITH_BYPASS_FULL": ("identity", *q, *k, *v),
        "PROJECTION_WITH_BYPASS_FULL": ("identity", "projection"),
        "RESIDUAL2_FULL": ("carry", "mlp"),
        "MLP_FC1_WITH_CARRY_FULL": ("carry", "mlp_fc1"),
        "MLP_GELU_WITH_CARRY_FULL": ("carry", "mlp_gelu"),
        "MLP_FC2_WITH_CARRY_FULL": ("carry", "mlp_fc2"),
    }
    return PortLayout(components, cuts)


def _save_once(observed: dict[str, Tensor], key: str, value: Tensor) -> None:
    if key in observed:
        raise VitAdapterError("PORT_INVOCATION_AMBIGUOUS:" + key)
    if not isinstance(value, Tensor):
        raise VitAdapterError("PORT_VALUE_NOT_TENSOR:" + key)
    observed[key] = value


def _register_capture(stack: ExitStack, module: nn.Module, location: str,
                      observed: dict[str, Tensor], key: str) -> None:
    if location == "input":
        handle = module.register_forward_pre_hook(
            lambda _module, args, key=key: _save_once(observed, key, args[0])
        )
    elif location == "output":
        handle = module.register_forward_hook(
            lambda _module, _args, output, key=key: _save_once(observed, key, output)
        )
    else:
        raise VitAdapterError("UNKNOWN_CAPTURE_LOCATION")
    stack.callback(handle.remove)


def port_fields(model: nn.Module, root: Tensor, *, block_index: int,
                class_id: int, second_order: bool = True) -> tuple[Tensor, tuple[str, ...]]:
    """Return every registered port field plus the independent full gradient.

    Each port field is ``J(z_e, root)^T dF/dz_e``.  ``create_graph=True`` is
    intentionally used for both cotangent and VJP so JVPs include the complete
    second-order dependency.
    """
    from timm.models.vision_transformer import Attention, Block

    if root.ndim != 4 or root.shape[0] != 1 or not root.is_floating_point():
        raise VitAdapterError("ROOT_SHAPE_OR_DTYPE")
    if not 0 <= block_index < len(model.blocks):
        raise VitAdapterError("BLOCK_INDEX")
    if not 0 <= class_id < 1000:
        raise VitAdapterError("CLASS_ID")
    block = model.blocks[block_index]
    if type(block) is not Block or type(block.attn) is not Attention:
        raise VitAdapterError("UNREGISTERED_BLOCK_TYPE")
    if getattr(block.attn, "fused_attn", True):
        raise VitAdapterError("STRUCTURAL_BACKEND_REQUIRES_UNFUSED_ATTENTION")
    if getattr(model, "grad_checkpointing", False):
        raise VitAdapterError("GRAD_CHECKPOINTING_NOT_ALLOWED")

    observed: dict[str, Tensor] = {}
    with ExitStack() as stack:
        for module, location, key in (
            (block, "input", "block_input"),
            (block.drop_path1, "output", "attention_branch"),
            (block.norm2, "input", "r1"),
            (block.norm2, "output", "mlp_input"),
            (block.mlp.fc1, "output", "fc1_output"),
            (block.mlp.act, "output", "gelu_output"),
            (block.mlp.fc2, "input", "fc2_input"),
            (block.drop_path2, "output", "mlp_branch"),
            (block, "output", "r2"),
        ):
            _register_capture(stack, module, location, observed, key)
        with _frozen_eval(model), capture_unfused_attention_edges(block.attn, observed), torch.enable_grad():
            logits = model(root)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise VitAdapterError("MODEL_OUTPUT_SHAPE")
            score = logits[:, class_id].sum()

            cotangent_keys = (
                "r1", "r2", "projection_input", "query_after_norm",
                "key_after_norm", "value_projection", "mlp_input",
                "fc1_output", "fc2_input",
            )
            cotangents = torch.autograd.grad(
                score, tuple(observed[key] for key in cotangent_keys),
                create_graph=second_order, retain_graph=True, allow_unused=False,
            )
            cotangent = dict(zip(cotangent_keys, cotangents))
            values: list[Tensor] = []
            names: list[str] = []

            def append(name: str, value_key: str, seed: Tensor) -> None:
                field = torch.autograd.grad(
                    observed[value_key], root, grad_outputs=seed,
                    create_graph=second_order, retain_graph=True, allow_unused=False,
                )[0]
                names.append(name)
                values.append(field)

            def append_head_slices(prefix: str, value_key: str, seed_source: Tensor,
                                   *, head_axis: int, head_count: int, batch_size: int = 3) -> None:
                """Evaluate disjoint head VJPs in small auditable batches."""
                for begin in range(0, head_count, batch_size):
                    selected = tuple(range(begin, min(begin + batch_size, head_count)))
                    seeds = torch.zeros(
                        (len(selected), *seed_source.shape), dtype=seed_source.dtype,
                        device=seed_source.device,
                    )
                    for local, head in enumerate(selected):
                        target = [slice(None)] * seeds.ndim
                        target[0] = local
                        target[head_axis + 1] = head
                        source = [slice(None)] * seed_source.ndim
                        source[head_axis] = head
                        seeds[tuple(target)] = seed_source[tuple(source)]
                    batched = torch.autograd.grad(
                        observed[value_key], root, grad_outputs=seeds,
                        is_grads_batched=True, create_graph=second_order,
                        retain_graph=True, allow_unused=False,
                    )[0]
                    for local, head in enumerate(selected):
                        names.append(f"{prefix}_{head}")
                        values.append(batched[local])

            append("identity", "block_input", cotangent["r1"])
            append("attention", "attention_branch", cotangent["r1"])
            append("projection", "projection_input", cotangent["projection_input"])

            projection = observed["projection_input"]
            projection_cotangent = cotangent["projection_input"]
            heads = block.attn.num_heads
            if projection.shape[-1] % heads:
                raise VitAdapterError("HEAD_SLICE_NOT_EXHAUSTIVE")
            width = projection.shape[-1] // heads
            reshaped_projection = projection.reshape(*projection.shape[:-1], heads, width)
            reshaped_cotangent = projection_cotangent.reshape_as(reshaped_projection)
            # Register the reshaped view as the producer for batched head slices.
            observed["projection_heads"] = reshaped_projection
            append_head_slices(
                "head", "projection_heads", reshaped_cotangent,
                head_axis=2, head_count=heads,
            )

            for family, value_key in (
                ("query", "query_after_norm"),
                ("key", "key_after_norm"),
                ("value", "value_projection"),
            ):
                source = observed[value_key]
                seed_source = cotangent[value_key]
                if source.ndim != 4 or source.shape[1] != heads:
                    raise VitAdapterError("QKV_HEAD_LAYOUT")
                append_head_slices(
                    family + "_head", value_key, seed_source,
                    head_axis=1, head_count=heads,
                )

            append("carry", "r1", cotangent["r2"])
            append("mlp", "mlp_branch", cotangent["r2"])
            append("mlp_fc1", "mlp_input", cotangent["mlp_input"])
            append("mlp_gelu", "fc1_output", cotangent["fc1_output"])
            append("mlp_fc2", "fc2_input", cotangent["fc2_input"])
            full = torch.autograd.grad(
                score, root, create_graph=second_order, retain_graph=True, allow_unused=False,
            )[0]
            names.append("__full__")
            values.append(full)

    layout = port_layout(block.attn.num_heads)
    if tuple(names[:-1]) != layout.component_ids:
        raise VitAdapterError("PORT_ORDER_DRIFT")
    result = torch.stack(values)
    if not bool(torch.isfinite(result).all()):
        raise VitAdapterError("NONFINITE_PORT_FIELD")
    return result, tuple(names)


def q_fields(model: nn.Module, root: Tensor, *, block_index: int,
             class_id: int) -> tuple[Tensor, tuple[str, ...]]:
    fields, names = port_fields(model, root, block_index=block_index, class_id=class_id)
    return root.unsqueeze(0) * fields, names


def parity(observed: Tensor, reference: Tensor, *, atol: float = ATOL,
           rtol: float = RTOL, near_zero: float = NEAR_ZERO) -> dict[str, Any]:
    if observed.shape != reference.shape:
        raise VitAdapterError("PARITY_SHAPE")
    if not bool(torch.isfinite(observed).all()) or not bool(torch.isfinite(reference).all()):
        raise VitAdapterError("PARITY_NONFINITE")
    left = observed.detach().double()
    right = reference.detach().double()
    error = left - right
    maximum = float(error.abs().max()) if error.numel() else 0.0
    reference_norm = float(right.norm())
    relative = float(error.norm() / right.norm().clamp_min(1e-30))
    elementwise = bool((error.abs() <= atol + rtol * right.abs()).all())
    if reference_norm <= near_zero:
        branch = "absolute_near_zero"
        branch_pass = maximum <= atol
    else:
        branch = "relative_nonzero"
        branch_pass = relative <= rtol
    return {
        "pass": bool(elementwise and branch_pass), "branch": branch,
        "max_abs": maximum, "relative_l2": relative,
        "reference_norm": reference_norm, "elementwise_pass": bool(elementwise),
    }


def patch_quadrant_directions(image: Tensor, *, patch_size: int = 16) -> tuple[Tensor, ...]:
    if image.shape != (1, 3, 224, 224) or patch_size != 16:
        raise VitAdapterError("PATCH_GROUP_INPUT")
    groups = []
    for row_group, col_group in ((0, 0), (0, 1), (1, 0), (1, 1)):
        mask = torch.zeros_like(image)
        row = slice(row_group * 7 * patch_size, (row_group + 1) * 7 * patch_size)
        col = slice(col_group * 7 * patch_size, (col_group + 1) * 7 * patch_size)
        mask[:, :, row, col] = 1
        groups.append(image * mask)
    reconstructed = sum(groups, torch.zeros_like(image))
    if not torch.equal(reconstructed, image):
        raise VitAdapterError("PATCH_GROUPS_NOT_EXACT_PARTITION")
    return tuple(groups)


def _weighted_add(accumulator: Tensor | None, value: Tensor, weight: float) -> Tensor:
    term = value.detach().double() * float(weight)
    return term if accumulator is None else accumulator.add(term)


def stream_path_integrals(
    model: nn.Module, image: Tensor, *, block_index: int, class_id: int,
    nodes: Sequence[float], weights: Sequence[float],
    directions: Sequence[Tensor], progress: Callable[[], None] | None = None,
    collect_group_diagnostics: bool = False,
) -> dict[str, Any]:
    """Compute A, contracted R/T, and IG without a Hessian tensor."""
    if len(nodes) != len(weights) or not nodes:
        raise VitAdapterError("QUADRATURE_LENGTH")
    if len(directions) != 4 or not torch.equal(sum(directions, torch.zeros_like(image)), image):
        raise VitAdapterError("DIRECTION_PARTITION")
    accumulators: dict[str, Tensor | None] = {
        "A": None, "R": None, "T": None, "IG": None,
        "R_block_diagonal": None,
    }
    names = (*port_layout(model.blocks[block_index].attn.num_heads).component_ids, "__full__")
    started = torch.cuda.Event(enable_timing=True) if image.is_cuda else None
    finished = torch.cuda.Event(enable_timing=True) if image.is_cuda else None
    if started is not None:
        started.record()
    import time
    wall_started = time.perf_counter()
    field_evaluations = 0
    for node, weight in zip(nodes, weights):
        root = (image * float(node)).detach().requires_grad_(True)
        directional_d: Tensor | None = None
        directional_q: Tensor | None = None
        primal: Tensor | None = None
        group_r: list[Tensor] = []
        for direction in directions:
            current, jvp_d = torch.autograd.functional.jvp(
                lambda value: port_fields(
                    model, value, block_index=block_index, class_id=class_id,
                )[0],
                root, direction, create_graph=False, strict=True,
            )
            field_evaluations += 1
            if primal is None:
                primal = current
            elif not torch.equal(primal, current):
                raise VitAdapterError("JVP_PRIMAL_DRIFT")
            directional_d = jvp_d if directional_d is None else directional_d + jvp_d
            group_t = direction.unsqueeze(0) * current + root.unsqueeze(0) * jvp_d
            directional_q = group_t if directional_q is None else directional_q + group_t
            group_r.append(root.unsqueeze(0) * jvp_d)
        assert primal is not None and directional_d is not None and directional_q is not None
        direct = image.unsqueeze(0) * primal
        curvature = root.unsqueeze(0) * directional_d
        accumulators["A"] = _weighted_add(accumulators["A"], direct, weight)
        accumulators["R"] = _weighted_add(accumulators["R"], curvature, weight)
        accumulators["T"] = _weighted_add(accumulators["T"], directional_q, weight)
        accumulators["IG"] = _weighted_add(accumulators["IG"], direct[-1], weight)
        block_diagonal = (
            sum(
                group * direction.ne(0).unsqueeze(0)
                for group, direction in zip(group_r, directions)
            ) if collect_group_diagnostics else curvature
        )
        accumulators["R_block_diagonal"] = _weighted_add(
            accumulators["R_block_diagonal"], block_diagonal, weight,
        )
        if progress is not None:
            progress()
    if finished is not None:
        finished.record(); torch.cuda.synchronize()
        device_seconds = float(started.elapsed_time(finished) / 1000.0)
    else:
        device_seconds = time.perf_counter() - wall_started
    assert all(value is not None for value in accumulators.values())
    return {
        **{key: value for key, value in accumulators.items() if value is not None},
        "names": names, "node_count": len(nodes),
        "patch_group_count": len(directions),
        "group_diagnostics_collected": collect_group_diagnostics,
        "port_field_evaluations": field_evaluations,
        "elapsed_seconds": time.perf_counter() - wall_started,
        "device_elapsed_seconds": device_seconds,
    }


def summarize_validation(
    model: nn.Module, image: Tensor, *, model_id: str, block_index: int,
    class_id: int, integration: Mapping[str, Any], omega_directions: Sequence[Tensor],
) -> dict[str, Any]:
    fields, names = port_fields(
        model, image.detach().requires_grad_(True), block_index=block_index,
        class_id=class_id, second_order=False,
    )
    if tuple(names) != tuple(integration["names"]):
        raise VitAdapterError("INTEGRATION_PORT_ORDER")
    layout = port_layout(model.blocks[block_index].attn.num_heads)
    indexes = {name: index for index, name in enumerate(names)}
    port_endpoint = image.unsqueeze(0).double() * fields[:-1].detach().double()
    full_endpoint = image.double() * fields[-1].detach().double()
    A = integration["A"][:-1]
    R = integration["R"][:-1]
    T = integration["T"][:-1]
    IG = integration["IG"]
    per_port_pefa = parity(T, port_endpoint)
    product_rule = parity(T, A + R)
    cut_rows = []
    for cut_id, keys in layout.cuts.items():
        selected = [indexes[key] for key in keys]
        endpoint_d = parity(fields[selected].sum(0), fields[-1])
        endpoint_q = parity(port_endpoint[selected].sum(0), full_endpoint)
        pcia_ig = parity(A[selected].sum(0), IG)
        pefa_endpoint = parity(T[selected].sum(0), full_endpoint)
        endpoint_ig = parity(R[selected].sum(0), full_endpoint - IG)
        omega_probes = []
        for direction_index, direction in enumerate(omega_directions):
            def cut_field(root: Tensor) -> Tensor:
                current, current_names = port_fields(
                    model, root, block_index=block_index, class_id=class_id, second_order=True,
                )
                current_indexes = {name: index for index, name in enumerate(current_names)}
                return current[[current_indexes[key] for key in keys]].sum(0)
            root = image.detach().requires_grad_(True)
            _out, jvp = torch.autograd.functional.jvp(
                cut_field, root, direction, create_graph=False, strict=True,
            )
            _out, vjp = torch.autograd.functional.vjp(
                cut_field, root, v=direction, create_graph=False, strict=True,
            )
            omega = parity(vjp, jvp)
            omega_probes.append({"direction_index": direction_index, **omega})
        identities = {
            "endpoint_D_closure": endpoint_d,
            "endpoint_Q_closure": endpoint_q,
            "pcia_to_ig": pcia_ig,
            "pefa_to_endpoint": pefa_endpoint,
            "T_equals_A_plus_R": product_rule,
            "endpoint_minus_ig_equals_R": endpoint_ig,
            "omega_zero_sum_directional_probes": omega_probes,
        }
        passed = all(
            value["pass"] for key, value in identities.items()
            if key != "omega_zero_sum_directional_probes"
        ) and all(value["pass"] for value in omega_probes)
        cut_rows.append({
            "cut_id": cut_id, "component_ids": list(keys),
            "identity_families": 5, "identities": identities, "pass": passed,
        })
    return {
        "model_id": model_id, "block_index": block_index,
        "class_id": class_id, "port_count": len(layout.component_ids),
        "cut_count": len(layout.cuts), "per_port_pefa_closure": per_port_pefa,
        "per_port_product_rule": product_rule, "cuts": cut_rows,
        "pass": per_port_pefa["pass"] and product_rule["pass"] and all(row["pass"] for row in cut_rows),
    }


def first_order_all_blocks(model: nn.Module, image: Tensor, *, class_id: int) -> list[dict[str, Any]]:
    rows = []
    for block_index, block in enumerate(model.blocks):
        fields, names = port_fields(
            model, image.detach().requires_grad_(True), block_index=block_index,
            class_id=class_id, second_order=False,
        )
        indexes = {name: index for index, name in enumerate(names)}
        cuts = []
        for cut_id, keys in port_layout(block.attn.num_heads).cuts.items():
            result = parity(fields[[indexes[key] for key in keys]].sum(0), fields[-1])
            cuts.append({"cut_id": cut_id, **result})
        rows.append({"block_index": block_index, "cuts": cuts,
                     "pass": all(row["pass"] for row in cuts)})
    return rows


def negative_controls(model: nn.Module, image: Tensor, *, block_index: int,
                      class_id: int, integration: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields, names = port_fields(
        model, image.detach().requires_grad_(True), block_index=block_index,
        class_id=class_id, second_order=False,
    )
    indexes = {name: index for index, name in enumerate(names)}
    layout = port_layout(model.blocks[block_index].attn.num_heads)
    full = fields[-1]
    A, R, T, IG = (integration[key][:-1] if key != "IG" else integration[key]
                    for key in ("A", "R", "T", "IG"))
    heads = layout.cuts["HEADS_WITH_BYPASS_FULL"][1:]
    qkv = layout.cuts["QKV_WITH_BYPASS_FULL"]
    cases = [
        ("omit_residual_identity", fields[indexes["attention"]], full, "FAIL_CUT_NOT_LEGAL"),
        ("duplicate_residual_identity", fields[[indexes["identity"], indexes["attention"], indexes["identity"]]].sum(0), full, "FAIL_CUT_NOT_LEGAL"),
        ("omit_attention_head", fields[[indexes[key] for key in ("identity", *heads[:-1])]].sum(0), full, "FAIL_CUT_NOT_LEGAL"),
        ("omit_qkv_value_head", fields[[indexes[key] for key in qkv[:-1]]].sum(0), full, "FAIL_CUT_NOT_LEGAL"),
        ("duplicate_serial_mlp_cuts", fields[[indexes["carry"], indexes["mlp_fc1"], indexes["mlp_gelu"]]].sum(0), full, "FAIL_CUT_NOT_LEGAL"),
        ("detach_second_order_D", A, T, "FAIL_SECOND_ORDER_GRAPH_TRUNCATED"),
        ("hessian_block_diagonal_only", A + integration["R_block_diagonal"][:-1],
         T, "FAIL_SECOND_ORDER_GRAPH_TRUNCATED"),
        ("omit_port_in_PCIA", A[[indexes[key] for key in qkv[:-1]]].sum(0), IG, "FAIL_QUADRATURE_CLOSURE"),
        ("omit_port_in_PEFA", T[[indexes[key] for key in qkv[:-1]]].sum(0), image.double() * fields[-1].double(), "FAIL_QUADRATURE_CLOSURE"),
    ]
    rows = []
    for control_id, observed, reference, expected in cases:
        result = parity(observed, reference)
        rows.append({
            "control_id": control_id, "expected_failure": expected,
            "observed_failure": expected if not result["pass"] else "FALSE_PASS",
            "pass": not result["pass"], "parity": result,
        })
    return rows


def structural_negative_controls(model: nn.Module, image: Tensor, *, block_index: int,
                                 class_id: int) -> list[dict[str, Any]]:
    """Negative controls that do not depend on a path-integral outcome."""
    fields, names = port_fields(
        model, image.detach().requires_grad_(True), block_index=block_index,
        class_id=class_id, second_order=False,
    )
    indexes = {name: index for index, name in enumerate(names)}
    heads = port_layout(model.blocks[block_index].attn.num_heads).cuts["HEADS_WITH_BYPASS_FULL"][1:]
    full = fields[-1]
    cases = [
        ("omit_residual_identity", fields[indexes["attention"]], "FAIL_CUT_NOT_LEGAL"),
        ("duplicate_residual_identity", fields[[indexes["identity"], indexes["attention"], indexes["identity"]]].sum(0), "FAIL_CUT_NOT_LEGAL"),
        ("omit_attention_head", fields[[indexes[key] for key in ("identity", *heads[:-1])]].sum(0), "FAIL_CUT_NOT_LEGAL"),
        ("swap_residual_input_slots", fields[indexes["identity"]] - fields[indexes["attention"]], "FAIL_CUT_NOT_LEGAL"),
        ("duplicate_serial_mlp_cuts", fields[[indexes["carry"], indexes["mlp_fc1"], indexes["mlp_gelu"]]].sum(0), "FAIL_CUT_NOT_LEGAL"),
    ]
    rows = []
    for control_id, observed, expected in cases:
        result = parity(observed, full)
        rows.append({
            "control_id": control_id, "expected_failure": expected,
            "observed_failure": expected if not result["pass"] else "FALSE_PASS",
            "pass": not result["pass"], "parity": result,
        })
    direction = image.detach() / image.detach().norm().clamp_min(1e-30)
    def detached_cut(root: Tensor) -> Tensor:
        current, current_names = port_fields(
            model, root, block_index=block_index, class_id=class_id,
        )
        current_indexes = {name: index for index, name in enumerate(current_names)}
        keys = port_layout(model.blocks[block_index].attn.num_heads).cuts["RESIDUAL1_FULL"]
        return current[[current_indexes[key] for key in keys]].sum(0).detach()
    try:
        torch.autograd.functional.jvp(
            detached_cut, image.detach().requires_grad_(True), direction, strict=True,
        )
    except RuntimeError as error:
        rows.append({
            "control_id": "detach_second_order_omega",
            "expected_failure": "FAIL_SECOND_ORDER_GRAPH_TRUNCATED",
            "observed_failure": "FAIL_SECOND_ORDER_GRAPH_TRUNCATED",
            "framework_error": str(error), "pass": True,
        })
    else:
        rows.append({
            "control_id": "detach_second_order_omega",
            "expected_failure": "FAIL_SECOND_ORDER_GRAPH_TRUNCATED",
            "observed_failure": "FALSE_PASS", "pass": False,
        })
    return rows


def tensor_summary(value: Tensor) -> dict[str, Any]:
    flat = value.detach().double().flatten().cpu().tolist()
    return {
        "shape": list(value.shape), "finite": all(math.isfinite(item) for item in flat),
        "signed_sum": math.fsum(flat), "absolute_mass": math.fsum(abs(item) for item in flat),
        "max_abs": max((abs(item) for item in flat), default=0.0),
    }
