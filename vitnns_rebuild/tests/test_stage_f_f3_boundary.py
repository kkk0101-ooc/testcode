from __future__ import annotations

import json
from pathlib import Path

import pytest

from vitnns.stage_f_f3 import (
    F3_REPORT_PATH, F3_ROOT_PATH, StageFF3Error, assert_no_f4_outputs,
    validate_completion, validate_outputs,
)

REPO = Path(__file__).resolve().parents[1]


def test_no_f4_or_final_combined_output_exists():
    assert assert_no_f4_outputs(REPO)["status"] == "PASS"


def test_formal_f3_outputs_validate_when_present():
    root = REPO / F3_ROOT_PATH
    if not (root / "render_inventory.json").exists():
        assert not (REPO / F3_REPORT_PATH / "validation_report.json").exists()
        return
    report = validate_outputs(REPO)
    assert report["status"] == "PASS"
    assert report["primary_map_count"] == 1000
    assert report["secondary_terminal_count"] == 3000
    assert report["secondary_valid_map_count"] == 2998
    assert report["secondary_failed_count"] == 2
    assert report["figure_6_sample_id"] == "ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"
    assert report["s4_slot_count"] == 12


def test_terminal_completion_is_atomic_and_f4_only_when_present():
    terminal = REPO / F3_ROOT_PATH / "completion/f3.json"
    if not terminal.exists():
        return
    assert not (REPO / F3_ROOT_PATH / "completion/f3.pending.json").exists()
    result = validate_completion(REPO)
    assert result["status"] == "PASS"
    assert result["next_authorization"] == "F4_ONLY"
    assert result["f4_started"] is False
    assert result["final_combined_manuscript_figures_rendered"] is False
