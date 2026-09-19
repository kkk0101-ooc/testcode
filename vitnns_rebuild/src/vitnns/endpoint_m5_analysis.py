"""Standard-library helpers for the Endpoint M5 reuse-only analysis."""
from __future__ import annotations

import hashlib
import math
import random
import statistics


def average_ranks(values):
    values = [float(value) for value in values]
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2
        for offset in range(start, stop):
            result[order[offset]] = rank
        start = stop
    return result


def spearman(left, right):
    if len(left) != len(right) or len(left) < 4:
        return None
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    left_centered = [value - left_mean for value in left_rank]
    right_centered = [value - right_mean for value in right_rank]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return math.fsum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def quantile(sorted_values, probability):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def deterministic_seed(key):
    payload = ("ENDPOINT_M5_20260919_V1|" + key).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_median(values, key, replicates=10000):
    """Sample the exact empirical-bootstrap median distribution via order statistics."""
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if not count:
        return None
    rng = random.Random(deterministic_seed(key))
    draws = []
    if count % 2:
        shape = (count + 1) // 2
        for _ in range(replicates):
            uniform = rng.betavariate(shape, shape)
            draws.append(ordered[min(int(count * uniform), count - 1)])
    else:
        half = count // 2
        for _ in range(replicates):
            lower_u = rng.betavariate(half, half + 1)
            upper_u = lower_u + (1 - lower_u) * rng.betavariate(1, half)
            lower = ordered[min(int(count * lower_u), count - 1)]
            upper = ordered[min(int(count * upper_u), count - 1)]
            draws.append((lower + upper) / 2)
    draws.sort()
    return [quantile(draws, 0.025), quantile(draws, 0.975)]


def summarize(values, key):
    values = [float(value) for value in values]
    return {
        "n_images": len(values),
        "median": float(statistics.median(values)),
        "ci95": bootstrap_median(values, key),
    }


def head_block_metrics(zero_rows, blur_rows):
    zero = {row["component_id"]: row for row in zero_rows if row["operation"] == "head"}
    blur = {row["component_id"]: row for row in blur_rows if row["operation"] == "head"}
    if set(zero) != set(blur):
        raise ValueError("HEAD_COMPONENT_MISMATCH")
    output = {
        "taylor_reference_concordance": [],
        "finite_effect_reference_concordance": [],
        "endpoint_behavior_normalized_zero": [],
        "endpoint_behavior_blur_rgb": [],
        "taylor_behavior_normalized_zero": [],
        "taylor_behavior_blur_rgb": [],
    }
    exact = {"taylor": 0, "finite_effect": 0}
    for block in range(12):
        identities = sorted(key for key, row in zero.items() if row["block"] == block)
        zrows = [zero[key] for key in identities]
        brows = [blur[key] for key in identities]
        endpoint = [row["m"] for row in zrows]
        taylor_zero = [abs(row["p_ref"]) for row in zrows]
        taylor_blur = [abs(row["p_ref"]) for row in brows]
        effect_zero = [row["absolute_effect"] for row in zrows]
        effect_blur = [row["absolute_effect"] for row in brows]
        output["taylor_reference_concordance"].append(spearman(taylor_zero, taylor_blur))
        output["finite_effect_reference_concordance"].append(spearman(effect_zero, effect_blur))
        output["endpoint_behavior_normalized_zero"].append(spearman(endpoint, effect_zero))
        output["endpoint_behavior_blur_rgb"].append(spearman(endpoint, effect_blur))
        output["taylor_behavior_normalized_zero"].append(spearman(taylor_zero, effect_zero))
        output["taylor_behavior_blur_rgb"].append(spearman(taylor_blur, effect_blur))
        exact["taylor"] += average_ranks(taylor_zero) == average_ranks(taylor_blur)
        exact["finite_effect"] += average_ranks(effect_zero) == average_ranks(effect_blur)
    if any(any(value is None for value in values) for values in output.values()):
        raise ValueError("UNDEFINED_HEAD_SPEARMAN")
    return {key: statistics.fmean(values) for key, values in output.items()}, exact


METHOD_MATRIX = [
    {"method": "Endpoint", "primary_question": "Observed-input local dependence of a registered scalar target", "internal_scalar_target": "yes", "input_space_map": "yes", "external_intervention_reference": "no", "local_jacobian_structural_closure": "yes", "finite_replacement_prediction": "empirical and limited"},
    {"method": "Activation-space Taylor", "primary_question": "First-order prediction of a specified activation replacement", "internal_scalar_target": "yes", "input_space_map": "constructible regional score", "external_intervention_reference": "yes", "local_jacobian_structural_closure": "not its primary contract", "finite_replacement_prediction": "strongly matched"},
    {"method": "Integrated Gradients", "primary_question": "Baseline-to-input path attribution", "internal_scalar_target": "yes", "input_space_map": "yes", "external_intervention_reference": "yes", "local_jacobian_structural_closure": "extensible", "finite_replacement_prediction": "baseline-difference attribution"},
    {"method": "Attention rollout", "primary_question": "Across-layer attention flow", "internal_scalar_target": "usually no", "input_space_map": "yes", "external_intervention_reference": "no", "local_jacobian_structural_closure": "no", "finite_replacement_prediction": "not its primary contract"},
    {"method": "Grad-CAM", "primary_question": "Class-related coarse activation localization", "internal_scalar_target": "partial", "input_space_map": "yes", "external_intervention_reference": "no", "local_jacobian_structural_closure": "no", "finite_replacement_prediction": "not its primary contract"},
    {"method": "Chefer/AttnLRP", "primary_question": "Transformer relevance propagation", "internal_scalar_target": "partial", "input_space_map": "yes", "external_intervention_reference": "usually no", "local_jacobian_structural_closure": "implementation-dependent", "finite_replacement_prediction": "not unified"},
    {"method": "Attention weights", "primary_question": "Token-to-token attention allocation", "internal_scalar_target": "attention object", "input_space_map": "patch/token", "external_intervention_reference": "no", "local_jacobian_structural_closure": "no", "finite_replacement_prediction": "not by weights alone"}
]
