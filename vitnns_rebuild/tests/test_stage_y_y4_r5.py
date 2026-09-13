import numpy as np

from scripts import validate_stage_y_y4_r5 as v
from vitnns import stage_y_y4_r5 as y

def synthetic_arrays():
    arrays = {}
    for block in y.BLOCKS:
        prefix = f"b{block:02d}"
        arrays[prefix + "_first_attention_cell"] = np.ones((56, 56), dtype=np.float32)
        arrays[prefix + "_first_attention_matrix"] = np.eye(196, dtype=np.float32)
        arrays[prefix + "_first_attention_cls"] = np.ones(196, dtype=np.float32)
        arrays[prefix + "_first_heads_cell"] = np.ones((12, 56, 56), dtype=np.float32)
        arrays[prefix + "_first_identity_cell"] = np.ones((56, 56), dtype=np.float32)
        arrays[prefix + "_second_residual_cell"] = np.ones((56, 56), dtype=np.float32)
        arrays[prefix + "_second_mlp_cell"] = np.ones((56, 56), dtype=np.float32)
        arrays[prefix + "_input_cell"] = 2 * np.ones((56, 56), dtype=np.float32)
    arrays["b11_first_attention_matrix"] = np.zeros((196, 196), dtype=np.float32)
    return arrays

def test_complete_primitive_field_contract():
    rows, heads = v.primitive(synthetic_arrays(), 0, 0, "source", "clean", "clean", None, "correct")
    assert len(rows) == 12 and len(heads) == 144
    assert all(set(v.PRIMITIVE_COLUMNS).issubset(row) for row in rows)
    assert all(row["spatial_cancellation"] == row["attention_cancellation"] == 0 for row in rows)
    assert rows[-1]["off_diagonal_absolute_mass_fraction"] is None

def test_trajectory_metric_roster_is_numeric_only():
    assert "spatial_cancellation" in v.TRAJ_METRICS
    assert "classification_stratum" not in v.TRAJ_METRICS

def test_r5_identity_and_fresh_namespace():
    assert y.STAGE_ID == "Y4-R5"
    assert y.ROOT.name == "formal"
    assert "y4_r5" in y.ROOT.as_posix()
    assert y.BLOCKED_REF.endswith("y4-r5-mechanism-analysis-and-decision-blocked")
