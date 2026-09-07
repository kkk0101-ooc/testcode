from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vitnns.stage_f_f3_visual import (
    ARCHIVE_REF, ATTACKS, COMPLETION_PATH, REPORT_PATH, ROOT_PATH,
    StageFVisualError, _blue_transform, _load_config, _sources,
    build_false_pass_audit, validate_completion, validate_outputs,
    verify_startup,
)

REPO = Path(__file__).resolve().parents[1]


def test_blue_display_contract_is_exact():
    config = _load_config(REPO)
    transform = config["transform"]
    assert transform["source"] == "ABSOLUTE_MAGNITUDE_OF_RAW_SIGNED_MAP"
    assert transform["normalization"] == "PER_PANEL_MAX_ABSOLUTE_VALUE"
    assert transform["gamma"] == 2.0
    assert transform["resize"] == [224, 224]
    assert transform["clipping"] is None
    assert transform["smoothing"] is None
    assert transform["max_pool"] is None
    assert transform["magnitude_comparison_allowed"] is False
    assert config["prohibitions"]["gradcam_or_figure_7"] is True


def test_blue_transform_is_absolute_panel_normalized_gamma_two():
    config = _load_config(REPO)
    source = np.zeros((56, 56), dtype=np.float32)
    source[0, 0] = -2.0
    source[10, 10] = 1.0
    displayed, record = _blue_transform(source, config)
    assert displayed.dtype == np.float32 and displayed.shape == (224, 224)
    assert record["raw_max_abs"] == 2.0
    assert record["gamma"] == 2.0
    assert record["normalization"] == "PER_PANEL_MAX_ABSOLUTE_VALUE"
    assert record["magnitude_comparison_allowed"] is False
    assert record["clipping"] is None and record["smoothing"] is None and record["max_pool"] is None


def test_zero_map_transform_is_finite_deep_blue_input():
    displayed, record = _blue_transform(np.zeros((56, 56), dtype=np.float32), _load_config(REPO))
    assert np.all(displayed == 0)
    assert np.isfinite(displayed).all()
    assert record["raw_max_abs"] == 0.0


def test_transform_rejects_wrong_dtype_shape_or_nonfinite():
    config = _load_config(REPO)
    with pytest.raises(StageFVisualError):
        _blue_transform(np.zeros((56, 56), dtype=np.float64), config)
    with pytest.raises(StageFVisualError):
        _blue_transform(np.zeros((14, 14), dtype=np.float32), config)
    bad = np.zeros((56, 56), dtype=np.float32); bad[0, 0] = np.nan
    with pytest.raises(StageFVisualError):
        _blue_transform(bad, config)


def test_sealed_f3_sources_preserve_selection_and_cross_layer_coverage():
    sources = _sources(REPO)
    assert len(sources["qualitative"]) == 9
    assert len(sources["cross"]) == 88
    assert sources["cross_selection"]["figure_6"]["sample_id"] == "ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"
    assert sources["cross_selection"]["slot_count"] == 12
    assert sources["cross_selection"]["unique_render_sample_count"] == 11
    assert sources["rules"]["render_order_unique_sample_ids"] == [
        "ILSVRC2012_val/ILSVRC2012_val_00033768.JPEG",
        "ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG",
        "ILSVRC2012_val/ILSVRC2012_val_00043103.JPEG",
        "ILSVRC2012_val/ILSVRC2012_val_00004138.JPEG",
    ]


def test_startup_binds_f3_archive_and_does_not_start_f4():
    gate = verify_startup(REPO, require_clean=False, require_remote=False)
    assert gate["status"] == "PASS"
    assert gate["f3_archive_commit"] == "1589fc3895657ee6a8f01d843ee26b67603998cb"
    assert gate["f3_r7_modified"] is False
    assert gate["f4_started"] is False
    assert gate["runtime"]["new_attribution_or_fidelity"] is False


def test_false_pass_audit_rejects_every_attack():
    audit = build_false_pass_audit(REPO)
    assert [item["attack"] for item in audit["attacks"]] == list(ATTACKS)
    assert all(item["result"] == "REJECTED" for item in audit["attacks"])
    assert audit["pass_possible_without_all_conditions"] is False


def test_visual_outputs_validate_when_present():
    if not (REPO / ROOT_PATH / "render_inventory.json").exists():
        assert not (REPO / REPORT_PATH / "validation_report.json").exists()
        return
    result = validate_outputs(REPO)
    assert result["status"] == "PASS"
    assert result["render_count"] == 27
    assert result["exported_file_count"] == 81
    assert result["gradcam_or_figure7_generated"] is False


def test_visual_completion_is_atomic_when_present():
    if not (REPO / COMPLETION_PATH).exists():
        return
    result = validate_completion(REPO)
    assert result["status"] == "PASS"
    assert result["archive_ref"] == ARCHIVE_REF
    assert result["next_authorization"] == "F4_ONLY"
    assert result["f4_started"] is False
