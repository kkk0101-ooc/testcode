"""Supplementary M1 missing-edge acquisition; primary E3/E4 remain immutable."""
from contextlib import contextmanager
from types import MethodType

import torch

from .endpoint_nns import _frozen_eval
from .endpoint_e4_intervention import (
    EDGE_NAMES, _explicit_attention_runtime, _branch_runtime, component_rows,
    component_tensor, dot64, norm64, subtract,
)


def missing_rows(num_heads, include_qk=True):
    rows = []
    for block in range(12):
        for name, key in (("identity1", "identity"), ("identity2", "carry")):
            rows.append(dict(component_id=f"b{block:02d}.{name}", block=block,
                             operation=name, head=None, e3_component_keys=[key]))
    if include_qk:
        rows += [r for r in component_rows(num_heads, include_secondary=True)
                 if r["operation"] in {"query", "key"}]
    return rows


@contextmanager
def identity_runtime(model, captures=None, modifications=()):
    """Expose two distinct skip edges. Replacing a skip leaves its branch clean."""
    originals = []
    try:
        for index, block in enumerate(model.blocks):
            originals.append((block, block.forward, "forward" in block.__dict__))

            def forward(module, value, index=index):
                skip1 = value.clone()
                skip2_rows = []
                for row in modifications:
                    if row["block"] == index:
                        if row["operation"] == "identity1":
                            skip1[row["batch_index"]] = row["replacement"][0].to(skip1)
                        elif row["operation"] == "identity2":
                            skip2_rows.append(row)
                branch1 = module.drop_path1(module.ls1(module.attn(module.norm1(value))))
                residual1 = skip1 + branch1
                skip2 = residual1.clone()
                for row in skip2_rows:
                    skip2[row["batch_index"]] = row["replacement"][0].to(skip2)
                output = skip2 + module.drop_path2(module.ls2(module.mlp(module.norm2(residual1))))
                if captures is not None:
                    captures[index].update(identity1=skip1, identity2=skip2)
                return output

            block.forward = MethodType(forward, block)
        yield
    finally:
        for block, original, had in reversed(originals):
            if had:
                block.forward = original
            else:
                del block.forward


def capture(model, inputs, class_id, grad=False):
    value = inputs.detach().clone().requires_grad_(grad)
    edges = {}
    context = torch.enable_grad() if grad else torch.no_grad()
    with _frozen_eval(model), context, identity_runtime(model, edges), \
            _explicit_attention_runtime(model, edges), _branch_runtime(model, edges):
        logits = model(value)
        score = logits[:, class_id]
        gradients = None
        if grad:
            keys = [(b, name) for b in range(12) for name in (*EDGE_NAMES, "identity1", "identity2")]
            values = torch.autograd.grad(score.sum(), tuple(edges[b][n] for b, n in keys))
            gradients = {b: {} for b in range(12)}
            for (b, n), g in zip(keys, values):
                gradients[b][n] = g.detach()
    return float(score.detach()), {b: {k: v.detach() for k, v in e.items()} for b, e in edges.items()}, gradients


def replace(model, image, class_id, rows, replacements, chunk=16):
    outputs = {}
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        changes = [dict(r, batch_index=i + 1, replacement=replacements[r["component_id"]])
                   for i, r in enumerate(part)]
        batch = image.detach().repeat(len(part) + 1, 1, 1, 1)
        with _frozen_eval(model), torch.no_grad(), identity_runtime(model, modifications=changes), \
                _explicit_attention_runtime(model, modifications=changes), \
                _branch_runtime(model, modifications=changes):
            scores = model(batch)[:, class_id]
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("NONFINITE_REPLACEMENT")
        for r, s in zip(part, scores[1:]):
            outputs[r["component_id"]] = (float(scores[0]), float(s))
    return outputs


def acquire(model, image, references, task, e3, e4):
    clean_score, edges, gradients = capture(model, image, task["class_id"], grad=True)
    refs = {name: capture(model, value, task["class_id"])[1] for name, value in references.items()}
    tol = 1e-4 + 1e-4 * abs(clean_score)
    if abs(clean_score - e4["clean_score"]) > tol:
        raise ValueError("CLEAN_SOURCE_PARITY")
    rows = missing_rows(model.blocks[0].attn.num_heads, task["split_order"] >= 64)
    controls = [r for r in component_rows(model.blocks[0].attn.num_heads)
                if r["block"] == 0 and (r["head"] in (0, None))]
    all_rows = rows + controls
    noop = replace(model, image, task["class_id"], rows,
                   {r["component_id"]: component_tensor(edges[r["block"]], r) for r in rows})
    noop_error = max(abs(c - s) for c, s in noop.values())
    if noop_error > tol:
        raise ValueError("NOOP_PARITY")
    result = []
    parity_max = 0.0
    for ref in references:
        replaced = replace(model, image, task["class_id"], all_rows,
                           {r["component_id"]: component_tensor(refs[ref][r["block"]], r) for r in all_rows})
        source = {r["component_id"]: r for r in e4["finite_utility"][ref]["components"]}
        for r in all_rows:
            block = r["block"]; cid = r["component_id"]
            z = component_tensor(edges[block], r); g = component_tensor(gradients[block], r)
            dz = subtract(z, component_tensor(refs[ref][block], r))
            observed_clean, score = replaced[cid]
            if abs(observed_clean - clean_score) > tol:
                raise ValueError("BATCH_CLEAN_PARITY")
            computed = dict(output_norm=norm64(z), gradient_norm=norm64(g),
                            gradient_activation=dot64(g, z), reference_displacement_norm=norm64(dz),
                            p_ref=dot64(g, dz), replacement_score=score, effect=clean_score - score)
            if cid in source:
                for key, value in computed.items():
                    error = abs(value - source[cid][key])
                    parity_max = max(parity_max, error)
                    if error > 1e-4 + 1e-4 * abs(source[cid][key]):
                        raise ValueError(f"SOURCE_COMPARATOR_PARITY:{cid}:{key}")
                continue
            summary = e3["blocks"][block]["components"][r["e3_component_keys"][0]]
            row = dict(r, reference_id=ref, **computed, a=summary["signed_sum"],
                       m=summary["absolute_mass"], absolute_effect=abs(computed["effect"]),
                       endpoint_source="E3_sealed_component_summary")
            if not all(torch.isfinite(torch.tensor(v)).item() for v in computed.values()):
                raise ValueError("NONFINITE_PREDICTOR")
            result.append(row)
    return dict(rows=result, clean_score=clean_score, noop_max_error=noop_error,
                control_max_error=parity_max, control_count=len(controls) * 2,
                new_condition_count=len(rows) * 2)
