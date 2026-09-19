import math

from vitnns.endpoint_m4_statistics import (
    bootstrap_median, exact_two_sided_sign, holm, image_values, spearman,
)


def synthetic_result():
    attention = []
    residual = []
    for block in range(12):
        for head in range(6):
            base = block + head / 10
            attention.append({
                "block": block, "head": head, "routing_share": base,
                "frozen_attention_endpoint_error": 2 * base,
                "frozen_attention_absolute_error": base, "contracted_gradient_error": base,
                "contracted_gradient_absolute_error": base,
                "routing_value_path_cancellation": base, "routing_value_cosine": base,
                "mean_query_saturation": float(head), "cls_query_saturation": base,
                "frozen_attention_jvp_error": base, "frozen_attention_jvp_absolute_error": base,
            })
        for kind in ("attention", "mlp"):
            summary = {"map_cancellation": block / 20}
            residual.append({
                "block": block, "residual_type": kind, "identity_share": block / 20,
                "branch_share": 1 - block / 20, "path_cancellation": block / 30,
                "identity_branch_cosine": block / 40, "output_identity_patch_tv": block / 50,
                "identity": summary, "branch": summary, "output": summary,
            })
    return {"attention": attention, "residual": residual}


def test_image_first_aggregation_and_relations():
    values = image_values(synthetic_result())
    assert values["contrasts"]["attention_routing_share_late_minus_early"] == 8
    assert values["contrasts"]["saturation_routing"] == 1
    assert math.isclose(values["contrasts"]["attention_identity_share_late_minus_early"], 0.4)


def test_bootstrap_sign_holm_and_na_are_deterministic():
    assert bootstrap_median([1, 2, 3], "x", 100) == bootstrap_median([1, 2, 3], "x", 100)
    assert spearman([1, 2, 3], [1, 2, 3]) is None
    assert math.isclose(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1)
    sign = exact_two_sided_sign([1, 2, -1, 0]); assert sign["positive"] == 2 and sign["negative"] == 1
    rows = holm([{"test_id": "a", "p": 0.01}, {"test_id": "b", "p": 0.04}])
    assert rows[0]["holm_p"] == 0.02 and rows[1]["holm_p"] == 0.04
