from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from vitnns import stage_y_y2 as y2

REPO = Path(__file__).resolve().parents[1]


def _validator():
    spec = importlib.util.spec_from_file_location(
        "validate_stage_y_y2_p0_r1_test",
        REPO / "scripts/validate_stage_y_y2_p0_r1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _formal_cohort() -> dict:
    records = []
    for order in range(100):
        records.append({
            "dataset_release_id": "ILSVRC2012_VAL_OFFICIAL",
            "class_id": order,
            "official_wnid": f"n{order:08d}",
            "source_object_id": f"ILSVRC2012_VAL_OFFICIAL:ILSVRC2012_val/ILSVRC2012_val_{order + 1:08d}.JPEG",
            "release_native_object_id": f"ILSVRC2012_val/ILSVRC2012_val_{order + 1:08d}.JPEG",
            "tar_member_name": f"ILSVRC2012_val_{order + 1:08d}.JPEG",
            "file_bytes": 1,
            "file_sha256": f"{order:064x}",
            "class_digest": f"{order + 100:064x}",
            "object_digest": f"{order + 200:064x}",
            "selection_order": order,
            "selection_digest": f"{order + 300:064x}",
        })
    return y2.signed("stage_y_y2_cohort_registry", "PASS", {"records": records})


def test_r1_validator_is_cpu_only_and_never_probes_gpu() -> None:
    source = (REPO / "scripts/validate_stage_y_y2_p0_r1.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "nvidia-smi" not in source
    imported = [alias.name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
    assert not any(name in {"torch", "torchvision", "timm"} for name in imported)
    assert not any(name.endswith("stage_y_y2_worker") for name in imported)


def test_r1_binds_immutable_p0_and_pre_access_state() -> None:
    validator = _validator()
    binding = validator.validate_p0_binding(False)
    audit = validator.pre_access_audit()
    assert binding["historical_root_immutable"] is True
    assert binding["commit"] == validator.P0_COMMIT
    assert audit["formal_outcome_access"] is False
    assert audit["cuda_or_gpu_accessed"] is False
    assert audit["formal_y2_execution_started"] is False


def test_r1_source_recomputes_every_elastic_schedule() -> None:
    validator = _validator()
    schedule, projections = validator.validate_schedule()
    assert list(projections) == [str(count) for count in range(1, 9)]
    assert projections["4"]["y2"]["shard_counts"] == [25, 25, 25, 25]
    assert projections["4"]["y3"]["shard_counts"] == [40, 40, 40, 40]
    assert projections["4"]["y2"]["critical_path_worker_wall_hours"] == 16.0
    assert projections["4"]["y2"]["overall_stage_wall_hours"] == 19.0
    assert schedule["y3_sharding"]["launch_authorized_in_y2_p0_r1"] is False


def test_manifest_schema_accepts_source_recomputed_four_shard_documents(tmp_path: Path) -> None:
    projection = y2.schedule_from_source_ledger(
        y2.read_json(REPO / "evidence/stage_y/y1_r1_resource_telemetry_waiver_20260911/formal/conservative_budget_receipt.json"),
        4,
    )["y2"]
    manifests = y2.build_shard_manifests(_formal_cohort(), tmp_path, 4, projection)
    schema = json.loads((REPO / "config/stage_y/y2_shard_manifest.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    assert [row["payload"]["unit_count"] for row in manifests] == [25, 25, 25, 25]
    for document in manifests:
        assert not list(validator.iter_errors(document))


def test_false_pass_audit_rejects_every_attack_independently() -> None:
    validator = _validator()
    result = validator.false_pass_audit()
    assert result["status"] == "PASS"
    assert result["attack_count"] == result["rejected_count"] == 17
    assert all(row["pass"] and row["observed"] == "REJECT" for row in result["checks"])


def test_r1_never_materializes_formal_y2_roots() -> None:
    assert not (REPO / y2.ROOT).exists()
    assert not (REPO / y2.REPORT_ROOT).exists()
