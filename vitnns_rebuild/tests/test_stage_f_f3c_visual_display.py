from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from vitnns import stage_f_f3c4 as f3c4
from vitnns import stage_f_f3c_visual as visual


REPO = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_native() -> dict[str, np.ndarray]:
    nns = np.zeros((3, 224, 224), dtype=np.float32)
    nns[0, 0, 0] = 3.0
    nns[1, 0, 0] = -1.0
    nns[2, 1, 1] = -4.0
    full_ig = -nns
    return {
        visual.NNS: nns,
        visual.GRAD_CAM: np.ones((14, 14), dtype=np.float32),
        visual.FULL_IG: full_ig,
        visual.ROLLOUT: np.full((14, 14), 2.0, dtype=np.float32),
        visual.RAW_ATTN: np.full((14, 14), 3.0, dtype=np.float32),
    }


def _clean_visual_process(expression: str) -> dict[str, object]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = str(REPO / "src")
    command = (
        "import json; from pathlib import Path; "
        "from vitnns import stage_f_f3c_visual as visual; "
        f"print(json.dumps({expression}, sort_keys=True))"
    )
    process = subprocess.run(
        [sys.executable, "-c", command], cwd=REPO, env=environment,
        text=True, capture_output=True, check=True,
    )
    return json.loads(process.stdout)


def test_signed_projection_and_split_preserve_sign_without_abs_packaging():
    native = _synthetic_native()[visual.NNS]
    signed = visual.signed_rgb_projection(native)
    positive, negative = visual.split_signed_support(signed)
    assert signed[0, 0] == 2.0 and signed[1, 1] == -4.0
    assert positive[0, 0] == 2.0 and positive[1, 1] == 0.0
    assert negative[0, 0] == 0.0 and negative[1, 1] == 4.0
    assert np.array_equal(positive - negative, signed)
    assert not np.shares_memory(positive, native)


def test_build_display_maps_keeps_positive_negative_and_expected_shapes():
    maps = visual.build_display_maps(_synthetic_native())
    assert tuple(maps) == (
        "nns_native_positive", "nns_pool4_bilinear_positive", "grad_cam_native",
        "full_ig_native_positive", "full_ig_pool4_bilinear_positive",
        "rollout_native", "raw_cls_attention_native", "nns_positive",
        "nns_negative", "full_ig_positive", "full_ig_negative",
    )
    assert maps["nns_native_positive"].shape == (224, 224)
    assert maps["nns_pool4_bilinear_positive"].shape == (224, 224)
    assert maps["grad_cam_native"].shape == (14, 14)
    assert maps["nns_positive"][0, 0] == 2.0
    assert maps["nns_negative"][1, 1] == 4.0
    assert maps["full_ig_positive"][1, 1] == 4.0
    assert maps["full_ig_negative"][0, 0] == 2.0


def test_max_pool_4x4_uses_nonoverlapping_maxima_and_resize_is_bilinear():
    value = np.arange(224 * 224, dtype=np.float64).reshape(224, 224)
    pooled = visual.max_pool_4x4(value)
    assert pooled.shape == (56, 56)
    assert pooled[0, 0] == value[3, 3]
    assert pooled[10, 20] == value[43, 83]
    constant = np.full((56, 56), 7.5, dtype=np.float64)
    resized = visual.bilinear_resize_224(constant)
    assert resized.shape == (224, 224)
    assert np.allclose(resized, 7.5)


def test_percentile_scale_clips_and_maximum_scale_does_not_clip():
    value = np.arange(100, dtype=np.float64).reshape(10, 10)
    primary, primary_meta = visual.display_scale(value, mode="percentile99")
    control, control_meta = visual.display_scale(value, mode="maximum")
    assert primary_meta["vmax_raw"] == pytest.approx(np.percentile(value, 99))
    assert primary_meta["upper_clipping"] is True
    assert primary_meta["clipped_pixel_count"] == 1
    assert primary.max() == 1.0
    assert control_meta["vmax_raw"] == 99.0
    assert control_meta["upper_clipping"] is False
    assert control_meta["clipped_pixel_count"] == 0
    assert np.array_equal(control, value / 99.0)


def test_zero_map_is_explicit_and_stays_zero_for_both_scales():
    for mode in ("percentile99", "maximum"):
        scaled, metadata = visual.display_scale(np.zeros((14, 14)), mode=mode)
        assert np.count_nonzero(scaled) == 0
        assert metadata["all_zero"] is True
        assert metadata["vmax_raw"] == 0.0
        assert metadata["clipped_pixel_count"] == 0


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: visual.signed_rgb_projection(np.zeros((224, 224), dtype=np.float32)), "3x224x224"),
        (lambda: visual.signed_rgb_projection(np.zeros((3, 224, 224), dtype=np.float64)), "float32"),
        (lambda: visual.max_pool_4x4(-np.ones((224, 224))), "nonnegative"),
        (lambda: visual.max_pool_4x4(np.zeros((223, 224))), "shape"),
        (lambda: visual.bilinear_resize_224(np.zeros((55, 56))), "shape"),
        (lambda: visual.display_scale(np.array([[np.nan]]), mode="maximum"), "finite"),
        (lambda: visual.display_scale(np.ones((2, 2)), mode="unknown"), "unsupported"),
        (lambda: visual.build_display_maps({}), "roster"),
    ],
)
def test_illegal_inputs_fail_closed(call, match):
    with pytest.raises(visual.StageFF3CVisualError, match=match):
        call()


def test_display_contract_records_independent_scales_gamma_and_noncomparability():
    params = visual.display_parameters()
    assert params["primary_scale"] == {
        "scope": "each_panel_independently", "upper_bound": "99th_percentile",
        "upper_clipping": True, "gamma": 1.0,
    }
    assert params["unclipped_control_scale"]["upper_clipping"] is False
    assert params["unclipped_control_scale"]["gamma"] == 1.0
    assert params["pooling"]["kernel"] == [4, 4]
    assert params["pooling"]["display_resize"] == "bilinear_56_to_224"
    assert params["color_intensity_cross_method_comparable"] is False
    assert params["exact_replication_of_paper_figure_14_claimed"] is False


def test_one_real_source_is_unchanged_by_display_transforms():
    row = f3c4._formal_rows(REPO)[0]
    terminal = f3c4._terminal_root(REPO, row["selection_order"], visual.NNS)
    path = terminal / "native.npy"
    before = (_sha(path), path.stat().st_size)
    manifest = f3c4._read(terminal / "manifest.json")
    native = f3c4._load_array(terminal, manifest, "native.npy", (3, 224, 224))
    signed = visual.signed_rgb_projection(native)
    positive, negative = visual.split_signed_support(signed)
    visual.bilinear_resize_224(visual.max_pool_4x4(positive))
    assert np.array_equal(positive - negative, signed)
    assert (_sha(path), path.stat().st_size) == before
    assert native.flags.writeable is False


def test_sealed_upstream_snapshot_is_recomputed_and_stable():
    before = _clean_visual_process("visual._upstream_snapshot(Path('.').resolve())")
    after = _clean_visual_process("visual._upstream_snapshot(Path('.').resolve())")
    assert before == after
    assert before["f3c3_file_count"] == 4118
    assert before["f3c4_r1_file_count"] == 134
    assert before["f3c4_r3_completion_validation"]["status"] == "PASS"
    assert before["archive_refs"]["f3c3"]["commit"] == f3c4.STARTUP_HEAD


def test_module_has_no_model_runtime_import_or_f4_surface():
    source = (REPO / visual.SOURCE_PATH).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import timm" not in source
    assert "load_model" not in source
    assert "stage_f_f4" not in source


def test_terminal_visual_package_revalidates_all_outputs_and_sources():
    result = _clean_visual_process("visual.validate_final(Path('.').resolve())")
    assert result == {
        "status": "PASS", "formal_samples": 50, "png_count": 30, "pdf_count": 3,
        "manual_visual_qc": "30/30 PASS", "sealed_sources_unchanged": True,
        "f4_started": False,
    }
