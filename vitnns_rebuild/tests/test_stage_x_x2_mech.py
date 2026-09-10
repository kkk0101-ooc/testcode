from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_x_x2_mech as mech
from vitnns import stage_x_x2_mech_analysis as analysis
from vitnns import stage_x_x2_mech_worker as worker

REPO = Path(__file__).resolve().parents[1]


def synthetic_map() -> np.ndarray:
    row, col = np.indices((224, 224), dtype=np.float64)
    return (np.sin(row / 11.0) + np.cos(col / 17.0) + (row - col) / 1000.0).astype(np.float32)


def test_config_schema_digest_and_closed_contract() -> None:
    config = mech.load_config(REPO); schema = mech.strict_read(REPO / mech.CONFIG_SCHEMA)
    Draft202012Validator.check_schema(schema)
    assert not list(Draft202012Validator(schema).iter_errors(config))
    mutated = copy.deepcopy(config); mutated["unknown"] = True
    assert list(Draft202012Validator(schema).iter_errors(mutated))
    assert config["cohort"]["method_transform_instances"] == 4776


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    for content in ('{"a":1,"a":2}', '{"a":NaN}'):
        path = tmp_path / "bad.json"; path.write_text(content)
        with pytest.raises(mech.X2MechP0Error): mech.strict_read(path)


def test_aggregation_uses_signed_and_absolute_sums() -> None:
    value = np.arange(64, dtype=np.float32).reshape(8, 8) - 20
    mask = np.ones((8, 8), dtype=bool)
    result = analysis.aggregate_blocks(value, mask, 4)
    assert result["signed_sum"].shape == (2, 2)
    assert result["signed_sum"][0, 0] == pytest.approx(value[:4, :4].sum(dtype=np.float64))
    assert result["absolute_mass_sum"][0, 0] == pytest.approx(np.abs(value[:4, :4]).sum(dtype=np.float64))


def test_partial_valid_blocks_are_excluded() -> None:
    value = synthetic_map(); mask = np.ones(value.shape, dtype=bool); mask[2, 2] = False
    result = analysis.aggregate_blocks(value, mask, 4)
    assert not result["valid_blocks"][0, 0]
    assert np.isnan(result["signed_sum"][0, 0])
    assert result["valid_blocks"].sum() == 56 * 56 - 1


def test_top_fraction_is_ceil_units_not_mass_and_deterministic() -> None:
    value = np.arange(100, dtype=np.float64).reshape(10, 10); valid = np.ones((10, 10), dtype=bool)
    first = analysis.top_fraction_indices(value, valid); second = analysis.top_fraction_indices(value, valid)
    assert np.array_equal(first, second) and len(first) == 10
    valid[0, :5] = False
    assert len(analysis.top_fraction_indices(value, valid)) == 10  # ceil(9.5)


def test_transport_flip_and_translation_controls() -> None:
    value = synthetic_map()
    flip, flip_mask = analysis.forward_inverse_control(value, "horizontal_flip")
    assert np.array_equal(flip, value) and flip_mask.all()
    translated, mask = analysis.forward_inverse_control(value, "translate_x_4")
    assert np.array_equal(translated[mask], value[mask])
    assert mask.sum() == 224 * 220


def test_sse_identity_and_partial_count() -> None:
    clean = synthetic_map(); changed = clean.copy(); changed[:8, :8] += np.arange(64).reshape(8, 8) / 10
    mask = np.ones(clean.shape, dtype=bool); mask[9, 9] = False
    result = analysis.sse_decomposition(clean, changed, mask, 4)
    assert result["sse_total"] == pytest.approx(result["sse_between"] + result["sse_within"], abs=1e-10, rel=1e-10)
    assert result["excluded_partial_block_count"] == 1
    assert np.isfinite(result["block_level_signed_rank_stability"])
    assert np.isfinite(result["within_block_demeaned_residual_stability"])
    assert result["block_level_signed_rank_stability"] < 1.0


def test_multiscale_stability_and_structure_are_finite() -> None:
    clean = synthetic_map(); changed = clean + np.roll(clean, 1, axis=1) * np.float32(0.01)
    mask = np.ones(clean.shape, dtype=bool)
    for block in (1, 4, 16):
        result = analysis.stability_metrics(clean, changed, mask, block)
        assert set(result) == set(mech.load_config(REPO)["stability"]["metrics"]) | {"valid_unit_count", "excluded_partial_unit_count"}
        assert all(np.isfinite(value) for value in result.values())
    structure = analysis.clean_structure_metrics(clean)
    assert set(structure) == set(mech.load_config(REPO)["clean_structure_metrics"])
    assert all(np.isfinite(value) for value in structure.values())


def test_synchronized_cluster_bootstrap_is_deterministic() -> None:
    rows = [{"model_id": model, "sample_id": sample, "x": index, "y": index + offset}
            for offset, model in enumerate(("m1", "m2")) for index, sample in enumerate(("a", "b", "c", "d"))]
    one = analysis.association_summary(rows, "x", "y", draws=50)
    two = analysis.association_summary(rows, "x", "y", draws=50)
    assert one == two and one["effect"] == pytest.approx(1.0)
    assert one["confirmatory_p_value"] is None


def test_seed_and_shard_are_deterministic() -> None:
    assert analysis.seed64("namespace", "a") == analysis.seed64("namespace", "a")
    assert analysis.seed64("namespace", "a") != analysis.seed64("namespace", "b")
    digest = "01" * 32
    assert worker.deterministic_shard(digest) == worker.deterministic_shard(digest)


def test_map_contract_and_tamper_rejection(tmp_path: Path) -> None:
    value = synthetic_map(); record = worker.save_map_new(tmp_path / "map.npy", value)
    assert record["dtype"] == "float32" and record["shape"] == [224, 224]
    with pytest.raises(FileExistsError): worker.save_map_new(tmp_path / "map.npy", value)
    with pytest.raises(worker.X2MechWorkerError): worker.save_map_new(tmp_path / "map.png", value)
    with pytest.raises(worker.X2MechWorkerError): worker.validate_map(value.astype(np.float64))


def test_worker_rejects_failed_or_tampered_route() -> None:
    route = {"unit_digest": "01" * 32, "source_status": "FAILED_ROUTE_LOCAL_NUMERICAL", "shard": 1, "source_maps": []}
    with pytest.raises(worker.X2MechWorkerError, match="failed"): worker.validate_route_plan(route, 1)
    route["source_status"] = "VALID"; route["shard"] = 0
    with pytest.raises(worker.X2MechWorkerError, match="shard"): worker.validate_route_plan(route, 0)


def test_p0_modules_have_no_forbidden_top_level_imports() -> None:
    forbidden = {"torch", "torchvision", "timm", "PIL", "cv2"}
    for relative in ("src/vitnns/stage_x_x2_mech.py", "src/vitnns/stage_x_x2_mech_worker.py", "src/vitnns/stage_x_x2_mech_analysis.py"):
        tree = ast.parse((REPO / relative).read_text())
        imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import): imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
        assert not imports.intersection(forbidden)


def test_clean_interpreter_import_does_not_import_model_stack() -> None:
    code = (
        "import json,sys; import vitnns.stage_x_x2_mech; "
        "import vitnns.stage_x_x2_mech_worker; import vitnns.stage_x_x2_mech_analysis; "
        "print(json.dumps({k:any(x==k or x.startswith(k+'.') for x in sys.modules) for k in ['torch','timm','torchvision','PIL']}))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, text=True, capture_output=True, check=True, env={**dict(os.environ), "PYTHONPATH": str(REPO / "src")})
    assert json.loads(result.stdout) == {"torch": False, "timm": False, "torchvision": False, "PIL": False}


def test_current_p0_test_process_has_no_model_stack() -> None:
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    assert not any(name == "timm" or name.startswith("timm.") for name in sys.modules)


def test_p0_control_has_no_real_array_or_outcome_reader() -> None:
    source = (REPO / "src/vitnns/stage_x_x2_mech.py").read_text()
    assert "np.load(" not in source and "numpy.load(" not in source
    assert "csv.DictReader" not in source and "pandas" not in source
    assert "run_m1_shard(" not in source


def test_p0_source_validation_does_not_load_arrays_or_outcome_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("array load reached")))
    config = mech.load_config(REPO)
    bindings, checks = mech._validate_x2(REPO)
    routes = mech._source_routes(REPO)
    budget = mech._budget(REPO, config)
    assert len(bindings) == 9 and len(checks) == 4
    assert len(routes) == 200 and sum(len(row["source_maps"]) for row in routes) == 398
    assert all(row["status"] == "PASS" for row in budget)


def test_future_worker_requires_explicit_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False
    def forbidden() -> dict[str, object]:
        nonlocal called; called = True; return {}
    monkeypatch.setattr(worker, "_lazy_runtime", forbidden)
    with pytest.raises(worker.X2MechWorkerError, match="authorization"):
        worker.run_m1_shard(REPO, 0, "cuda:0", "WRONG")
    assert not called


def test_completion_if_present_is_terminal_and_access_clean() -> None:
    if not (REPO / mech.COMPLETION).exists(): return
    result = mech.validate_completion(REPO)
    assert result["status"] == "PASS"
    assert result["next_authorization"] == mech.NEXT_AUTHORIZATION
