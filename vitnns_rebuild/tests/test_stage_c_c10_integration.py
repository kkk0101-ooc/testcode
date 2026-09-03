from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from vitnns import stage_c_c10 as c10
from vitnns import stage_c_c10_source as source
from vitnns import stage_c_c10_table as table
from vitnns import stage_c_cli
from vitnns.recovery_c10_launch import RecoveryC10LaunchRecord
from vitnns.stage_c_config import load_stage_c_config
from vitnns.stage_c_plan import (
    build_execution_plan, load_run_spec, load_stage_c_registry,
    load_stage_c_selection,
)


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64
GIT_COMMIT = "8c65e38cdd36bf4fe6a5ea6cf6d81f913f7b9de0"
GIT_TREE = "b4a575d8ffbb4608493a96ebf9dec2016717bfb0"
CODE_ATTESTATION = {
    "commit": GIT_COMMIT,
    "dirty": False,
    "tree": GIT_TREE,
    "archive_ref": c10.C10_PRODUCER_ARCHIVE_REF,
    "source_surface": c10.C10_CODE_SURFACE,
    "source_tree": GIT_TREE,
}
RUN_SPEC = ROOT / "config/stage_c_c9_run_spec.json"


def _recovery_authority(output_root, c9_root):
    return RecoveryC10LaunchRecord(
        path=ROOT / "config/recovery_c10_launch_record_20260903.json",
        document={}, c9_record=SimpleNamespace(output_root=Path(c9_root).resolve()),
        output_root=Path(output_root).resolve(), producer_commit=GIT_COMMIT,
        launch_commit="f" * 40, file_sha256="e" * 64, repo_root=ROOT,
    )


def _plan_unit(*, digest: str = SHA) -> dict:
    key = {
        "sample_id": "holdout-00",
        "method_id": "synthetic_class_method",
        "config_id": "global",
        "block_index": None,
        "head_index": None,
        "target_ref": "clean_predicted_class_raw_logit",
        "canonical_digest": digest,
    }
    return {
        "key": key,
        "sample": {
            "sample_id": key["sample_id"],
            "role": "holdout",
            "audit": False,
        },
        "comparison_line": "class_logit",
        "fidelity_eligible": False,
        "structural_reference_only": False,
    }


def _success_record(unit: dict) -> dict:
    key = unit["key"]
    return {
        "plan_index": 0,
        "unit_key": dict(key),
        "state": "succeeded",
        "trusted_by_batch_validator": True,
        "attempt_path": (
            f"units/{key['canonical_digest']}/attempts/attempt-000001"
        ),
        "hashes": {
            "status_sha256": "1" * 64,
            "selected_status_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "provenance_sha256": "3" * 64,
            "result_sha256": "4" * 64,
            "unit_identity_sha256": key["canonical_digest"],
        },
        "scalars": {
            "sample_id": key["sample_id"],
            "method_id": key["method_id"],
            "prediction": {"clean_top1_correct": False},
            "repeatability": [],
            "stability": [
                {
                    "perturbation_id": perturbation,
                    "natural_prediction_descriptive": {
                        "prediction_flipped": index % 2 == 0
                    },
                }
                for index, perturbation in enumerate(table.PERTURBATION_IDS)
            ],
        },
        "resources": {
            "wall_time_seconds": {
                "status": "NA",
                "value": None,
                "reason": "constant_left",
            },
            "artifact_bytes": 123,
            "calls": {"forward_calls": 7, "backward_calls": 3},
            "cuda": {
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
            },
        },
        "audit": False,
        "failure": None,
        "validation_errors": [],
    }


def _template_keys(unit: dict) -> Counter:
    return Counter(
        (
            spec["family"],
            table.metric_id_from_selectors(spec["metric"], **spec["selectors"]),
            spec["path"],
        )
        for spec in table._expected_specs(table._plan_fields(unit))
    )


def _row_keys(rows: list[dict]) -> Counter:
    return Counter(
        (row["metric_family"], row["metric_id"], row["source_field_path"])
        for row in rows
    )


@pytest.fixture(scope="module")
def frozen_plan():
    config = load_stage_c_config()
    registry = load_stage_c_registry(config, verify_live_assets=False)
    selection = load_stage_c_selection(config, required=True)
    return build_execution_plan(
        config, registry, load_run_spec(RUN_SPEC), selection=selection
    )


def test_frozen_plan_locks_the_301224_row_space(frozen_plan):
    method_counts = Counter(unit.key.method_id for unit in frozen_plan.units)
    signed_counts = {
        c10.NNS_METHOD_ID: 80,
        c10.MATCHED_METHOD_ID: 80,
        "full_model_class_logit_ig": 20,
    }
    l1_count = len(frozen_plan.units) + sum(signed_counts.values()) + (80 * 16)

    assert len(frozen_plan.units) == 320
    assert method_counts[c10.NNS_METHOD_ID] == 80
    assert table.expected_base_row_count(frozen_plan.units) == 297_844
    assert l1_count == 1_780
    assert c10._expected_scientific_derived_row_count(frozen_plan.units) == 1_600
    assert 297_844 + l1_count + 1_600 == 301_224

    signed_methods = set(signed_counts)
    records = [
        {
            "plan_index": index,
            "unit_key": unit.key.to_dict(),
            "scalars": {
                "sign_semantics": (
                    "signed" if unit.key.method_id in signed_methods
                    else "nonnegative"
                )
            },
        }
        for index, unit in enumerate(frozen_plan.units)
    ]
    coordinates = list(c10._expected_l1_row_coordinates(records))
    negative = [
        item for item in coordinates
        if item["method_id"] == c10.NNS_METHOD_ID
        and item["metric_family"] == "clean_spatial_structure_similarity"
        and "view=negative_magnitude" in item["metric_id"]
    ]
    assert len(coordinates) == 1_780
    assert len(negative) == 560
    assert len({tuple(item.values()) for item in negative}) == 560
    assert Counter(
        item["metric_id"].split("|", 1)[0] for item in negative
    ) == {
        "pearson": 80,
        "spearman": 80,
        "jensen_shannon_divergence": 80,
        "hotspot_jaccard_top_10_percent": 80,
        "hotspot_jaccard_top_20_percent": 80,
        "kl": 160,
    }


def test_long_table_emits_the_exact_applicability_template_without_drops():
    unit = _plan_unit()
    record = _success_record(unit)

    rows = table.build_c10_long_table([record], [unit])

    assert _row_keys(rows) == _template_keys(unit)
    assert len(rows) == sum(_template_keys(unit).values()) == 11
    assert Counter(row["status"] for row in rows) == {"OK": 10, "NA": 1}
    wall_time = next(row for row in rows if row["metric_id"] == "wall_time_seconds")
    assert wall_time["value"] is None
    assert wall_time["na_reason"] == "constant_left"
    assert wall_time["invalidating"] is False
    assert all(row["misclassified"] is True for row in rows)
    assert {row["source_attempt_path"] for row in rows} == {
        record["attempt_path"]
    }
    assert not Path(record["attempt_path"]).is_absolute()


def test_missing_unit_emit_mode_retains_the_exact_null_template():
    unit = _plan_unit()

    with pytest.raises(table.StageC10TableIntegrityError, match="every C9 unit"):
        table.build_c10_long_table([], [unit])

    rows = table.build_c10_long_table(
        [], [unit], failed_unit_policy="emit"
    )
    assert _row_keys(rows) == _template_keys(unit)
    assert len(rows) == 11
    assert all(row["status"] == "UNIT_INCOMPLETE" for row in rows)
    assert all(row["value"] is None for row in rows)
    assert all(row["na_reason"] == "unit_missing" for row in rows)
    assert all(row["invalidating"] is True for row in rows)


def test_csv_manifest_hashes_the_actual_emitted_schema_file_bytes(tmp_path):
    rows = table.build_c10_long_table(
        [_success_record(_plan_unit())], [_plan_unit()]
    )
    csv_bytes = table.rows_to_csv_bytes(rows)
    schema_bytes = c10._json_bytes(table.schema_document())
    schema_path = tmp_path / "per_sample.schema.json"
    schema_path.write_bytes(schema_bytes)

    manifest = table.build_row_manifest(rows, csv_bytes=csv_bytes)

    assert manifest["row_count"] == len(rows)
    assert manifest["csv_sha256"] == hashlib.sha256(csv_bytes).hexdigest()
    assert manifest["schema_sha256"] == hashlib.sha256(
        schema_path.read_bytes()
    ).hexdigest()
    assert table.validate_csv_bytes(csv_bytes) == manifest
    assert table.rows_to_csv_bytes(table.read_csv_bytes(csv_bytes)) == csv_bytes


def _tiny_payloads() -> tuple[dict[str, bytes], dict]:
    csv_bytes = table.rows_to_csv_bytes([])
    report = c10._document(
        "stage_c_c10_acceptance_report",
        status="BLOCKED",
        primary_decision="GO",
        downstream_authorized=False,
        stage_d_started=False,
    )
    documents = {
        "per_sample.schema.json": table.schema_document(),
        "per_sample_manifest.json": table.build_row_manifest(
            [], csv_bytes=csv_bytes
        ),
        "summaries.json": c10._document("stage_c_c10_summaries"),
        "bootstrap.json": c10._document("stage_c_c10_bootstrap"),
        "holm.json": c10._document(
            "stage_c_c10_holm_report",
            status="HOLM_BLOCKED_UNDERSPECIFIED",
        ),
        "reconciliation.json": c10._document(
            "stage_c_c10_unit_reconciliation", no_rows_dropped=True
        ),
        "decision.json": c10._document(
            "stage_c_c10_decision",
            decision="GO",
            downstream_authorized=False,
            stage_d_started=False,
        ),
        "stage_c_report.json": report,
        "provenance.json": c10._document(
            "stage_c_c10_provenance",
            code=CODE_ATTESTATION,
        ),
    }
    payloads = {
        name: (
            csv_bytes if name == "per_sample.csv" else c10._json_bytes(documents[name])
        )
        for name in c10.OUTPUT_FILES
    }
    return payloads, report


def _write_tiny_output(root: Path) -> tuple[dict[str, bytes], dict]:
    payloads, report = _tiny_payloads()
    root.mkdir()
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    manifest = c10._output_manifest(root, c10.OUTPUT_FILES)
    (root / c10.MANIFEST_FILENAME).write_bytes(c10._json_bytes(manifest))
    status = c10._document(
        "stage_c_c10_status",
        complete=True,
        terminal=True,
        status=report["status"],
        stage_c_acceptance_status=report["status"],
        primary_decision=report["primary_decision"],
        manifest_sha256=hashlib.sha256(
            (root / c10.MANIFEST_FILENAME).read_bytes()
        ).hexdigest(),
        payload_count=len(c10.OUTPUT_FILES),
        downstream_authorized=False,
        stage_d_started=False,
    )
    (root / c10.STATUS_FILENAME).write_bytes(c10._json_bytes(status))
    return payloads, report


def test_tiny_sealed_output_validates_by_source_recomputation(
    tmp_path, monkeypatch
):
    root = tmp_path / "c10"
    payloads, report = _write_tiny_output(root)
    monkeypatch.setattr(
        c10, "_assemble_c10", lambda *_args, **_kwargs: (payloads, report)
    )
    monkeypatch.setattr(
        c10, "_resolve_c10_producer_attestation", lambda *_args: CODE_ATTESTATION
    )

    authority = _recovery_authority(root, "unused-c9")
    assert c10.validate_c10_output_root(
        "unused-c9", root, object(), object(), recovery_authority=authority
    ) == report


@pytest.mark.parametrize("mutation", ["payload", "manifest", "status", "extra", "missing"])
def test_output_validation_rejects_tamper_and_file_set_drift(
    tmp_path, monkeypatch, mutation
):
    root = tmp_path / "c10"
    payloads, report = _write_tiny_output(root)
    monkeypatch.setattr(
        c10, "_assemble_c10", lambda *_args, **_kwargs: (payloads, report)
    )
    monkeypatch.setattr(
        c10, "_resolve_c10_producer_attestation", lambda *_args: CODE_ATTESTATION
    )

    if mutation == "payload":
        (root / "summaries.json").write_bytes(
            c10._json_bytes(c10._document("tampered_summaries"))
        )
    elif mutation == "manifest":
        value = json.loads((root / c10.MANIFEST_FILENAME).read_text())
        value["artifact_count"] += 1
        (root / c10.MANIFEST_FILENAME).write_bytes(c10._json_bytes(value))
    elif mutation == "status":
        value = json.loads((root / c10.STATUS_FILENAME).read_text())
        value["complete"] = False
        (root / c10.STATUS_FILENAME).write_bytes(c10._json_bytes(value))
    elif mutation == "extra":
        (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    else:
        (root / "summaries.json").unlink()

    with pytest.raises(c10.StageC10IntegrityError):
        c10.validate_c10_output_root(
            "unused-c9", root, object(), object(),
            recovery_authority=_recovery_authority(root, "unused-c9"),
        )


def test_build_requires_clean_git_and_does_not_create_a_root(
    tmp_path, monkeypatch
):
    output = tmp_path / "new-c10"
    monkeypatch.setattr(
        c10, "capture_git_state", lambda _root: {"commit": "a" * 40, "dirty": True}
    )
    monkeypatch.setattr(
        c10, "_assemble_c10", lambda *_a, **_k: pytest.fail("assembly attempted")
    )

    with pytest.raises(c10.StageC10BlockedError, match="clean Git worktree"):
        c10.build_c10_output_root("c9", output, object(), object())
    assert not output.exists()


def test_build_requires_a_fresh_root_before_assembly(tmp_path, monkeypatch):
    output = tmp_path / "existing-c10"
    output.mkdir()
    clean = {"commit": GIT_COMMIT, "dirty": False}
    monkeypatch.setattr(c10, "capture_git_state", lambda _root: clean)
    monkeypatch.setattr(
        c10, "_assemble_c10", lambda *_a, **_k: pytest.fail("assembly attempted")
    )

    with pytest.raises(c10.StageC10BlockedError, match="must not already exist"):
        c10.build_c10_output_root(
            "c9", output, object(), object(),
            recovery_authority=_recovery_authority(output, "c9"),
            observed_git_state=clean
        )


def test_clean_build_seals_a_new_root_and_immediately_revalidates(
    tmp_path, monkeypatch
):
    output = tmp_path / "new-c10"
    clean = {"commit": GIT_COMMIT, "dirty": False}
    payloads, report = _tiny_payloads()
    monkeypatch.setattr(c10, "capture_git_state", lambda _root: clean)
    monkeypatch.setattr(
        c10, "_resolve_c10_producer_attestation", lambda *_args: CODE_ATTESTATION
    )
    monkeypatch.setattr(
        c10, "_assemble_c10", lambda *_args, **_kwargs: (payloads, report)
    )

    result = c10.build_c10_output_root(
        "c9", output, object(), object(),
        recovery_authority=_recovery_authority(output, "c9"),
        observed_git_state=clean
    )

    assert result == report
    assert {path.name for path in output.iterdir()} == {
        *c10.OUTPUT_FILES, c10.MANIFEST_FILENAME, c10.STATUS_FILENAME
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_type", "not_stage_c10_status"),
        ("stage_c_acceptance_status", "PASS"),
        ("payload_count", 999),
        ("downstream_authorized", True),
        ("stage_d_started", True),
    ],
)
def test_output_validation_rejects_resigned_status_safety_tamper(
    tmp_path, monkeypatch, field, value
):
    root = tmp_path / "c10"
    payloads, report = _write_tiny_output(root)
    monkeypatch.setattr(
        c10, "_assemble_c10", lambda *_args, **_kwargs: (payloads, report)
    )
    monkeypatch.setattr(
        c10, "_resolve_c10_producer_attestation", lambda *_args: CODE_ATTESTATION
    )
    status_path = root / c10.STATUS_FILENAME
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status[field] = value
    status["canonical_digest"] = c10.canonical_sha256({
        key: item for key, item in status.items() if key != "canonical_digest"
    })
    status_path.write_bytes(c10._json_bytes(status))

    with pytest.raises(c10.StageC10IntegrityError, match="status/manifest"):
        c10.validate_c10_output_root(
            "unused-c9", root, object(), object(),
            recovery_authority=_recovery_authority(root, "unused-c9"),
        )


def test_output_validation_rejects_self_asserted_producer_commit(
    tmp_path, monkeypatch
):
    root = tmp_path / "c10"
    payloads, report = _write_tiny_output(root)
    monkeypatch.setattr(
        c10, "_assemble_c10", lambda *_args, **_kwargs: (payloads, report)
    )
    trusted = {**CODE_ATTESTATION, "commit": "f" * 40}
    monkeypatch.setattr(
        c10, "_resolve_c10_producer_attestation", lambda *_args: trusted
    )

    with pytest.raises(c10.StageC10IntegrityError, match="independent archive"):
        c10.validate_c10_output_root(
            "unused-c9", root, object(), object(),
            recovery_authority=_recovery_authority(root, "unused-c9"),
        )


def test_build_rejects_git_state_change_before_creating_root(
    tmp_path, monkeypatch
):
    output = tmp_path / "new-c10"
    clean = {"commit": GIT_COMMIT, "dirty": False}
    changed = {"commit": GIT_COMMIT, "dirty": True}
    states = iter((clean, changed))
    payloads, report = _tiny_payloads()
    monkeypatch.setattr(c10, "capture_git_state", lambda _root: next(states))
    monkeypatch.setattr(
        c10, "_resolve_c10_producer_attestation", lambda *_args: CODE_ATTESTATION
    )
    monkeypatch.setattr(
        c10, "_assemble_c10", lambda *_args, **_kwargs: (payloads, report)
    )

    with pytest.raises(c10.StageC10IntegrityError, match="during assembly"):
        c10.build_c10_output_root(
            "c9", output, object(), object(),
            recovery_authority=_recovery_authority(output, "c9"),
            observed_git_state=clean
        )
    assert not output.exists()


@pytest.mark.parametrize("command", ["c10-build", "c10-validate"])
@pytest.mark.parametrize(
    "forbidden",
    [
        ("--device", "cpu"),
        ("--selection", "selection.json"),
        ("--registry", "registry.json"),
        ("--data-root", "images"),
        ("--sample-id", "holdout-00"),
        ("--shard-index", "0"),
    ],
)
def test_c10_cli_rejects_execution_and_selection_overrides(command, forbidden):
    argv = [
        command,
        "--config", "stage_c.json",
        "--run-spec", "stage_c_c9_run_spec.json",
        "--recovery-c10-launch-record", "recovery-c10.json",
        "--c9-root", "sealed-c9",
        "--output-root", "c10",
        *forbidden,
    ]

    with pytest.raises(stage_c_cli.ArgumentError, match="unrecognized arguments"):
        stage_c_cli.build_parser().parse_args(argv)


def test_c10_cli_surface_contains_only_frozen_inputs_and_roots():
    args = stage_c_cli.build_parser().parse_args([
        "c10-validate",
        "--config", "stage_c.json",
        "--run-spec", "stage_c_c9_run_spec.json",
        "--recovery-c10-launch-record", "recovery-c10.json",
        "--c9-root", "sealed-c9",
        "--output-root", "c10",
    ])

    assert vars(args) == {
        "command": "c10-validate",
        "config": Path("stage_c.json"),
        "run_spec": Path("stage_c_c9_run_spec.json"),
        "recovery_c10_launch_record": Path("recovery-c10.json"),
        "c9_root": Path("sealed-c9"),
        "output_root": Path("c10"),
    }


def test_holm_blocker_preserves_primary_decision_and_forbids_stage_d(
    monkeypatch, tmp_path
):
    holdout = [{"role": "holdout"} for _ in range(15)]
    summaries = {
        "winners": {
            "block_00_head_09": {
                "metrics": {
                    "delta_fid_fine4": {"holdout": {}},
                    **{
                        f"residual_signed_spearman_{name}": {"holdout": {}}
                        for name in c10.STABILITY_PERTURBATIONS
                    },
                }
            }
        }
    }
    monkeypatch.setattr(c10, "_scope_sample_ids", lambda _plan: {
        "calibration": [f"cal-{index}" for index in range(5)],
        "holdout": [f"hold-{index}" for index in range(15)],
        "all_20": [f"all-{index}" for index in range(20)],
    })
    monkeypatch.setattr(c10, "_summary_document", lambda _samples, _ids: (summaries, {"records": []}))
    monkeypatch.setattr(c10, "_summarize_long_table", lambda _rows, _ids: ([], []))
    monkeypatch.setattr(c10, "evaluate_effect", lambda _summary: {"status": "PASS_EFFECT"})
    monkeypatch.setattr(
        c10, "evaluate_spatial",
        lambda _rows, _summaries: {
            "status": "PASS_SPATIAL",
            "joint_spatial_status": "PASS_JOINT_SPATIAL",
            "stability": {
                name: {"status": "PASS_STABILITY"}
                for name in c10.STABILITY_PERTURBATIONS
            },
        },
    )
    monkeypatch.setattr(c10, "_evaluate_validity", lambda *_a: {"status": "PASS_VALIDITY"})
    monkeypatch.setattr(c10, "_evaluate_cost", lambda _report: {"status": "PASS_COST"})
    monkeypatch.setattr(c10, "_unit_reconciliation", lambda *_a: {"no_rows_dropped": True})
    monkeypatch.setattr(c10, "sha256_file", lambda _path: "f" * 64)
    monkeypatch.setattr(
        c10,
        "audit_holm_contract",
        lambda *_a: {
            "status": "HOLM_BLOCKED_UNDERSPECIFIED",
            "holm_performed": False,
            "raw_p_values": [],
            "adjusted_p_values": [],
            "affects_primary_decision": False,
            "acceptance_blocking": True,
        },
    )
    snapshot = SimpleNamespace(
        report={}, records=(), bindings={
            "report_file_sha256": "1" * 64,
            "root_inventory_digest": "2" * 64,
            "producer_commit": GIT_COMMIT,
            "authority_execution_commit": "3" * 40,
            "recovery_authority_path": "config/recovery-c9.json",
            "recovery_authority_file_sha256": "4" * 64,
            "recovery_authority_canonical_digest": "5" * 64,
        }, hash_audit={}
    )
    config = SimpleNamespace(
        protocol=SimpleNamespace(path="docs/阶段C_protocol.md"),
        to_dict=lambda: {},
    )
    plan = SimpleNamespace(canonical_digest=SHA)

    documents = c10._build_documents(
        snapshot, config, plan, [], {c10.PRIMARY: holdout},
        code={"commit": "a" * 40},
        c9_root=tmp_path / "c9",
        output_root=tmp_path / "c10",
    )

    assert documents["holm.json"]["status"] == "HOLM_BLOCKED_UNDERSPECIFIED"
    assert documents["decision.json"]["decision"] == "GO"
    assert documents["decision.json"]["stage_c_acceptance_status"] == "BLOCKED"
    assert documents["stage_c_report.json"]["primary_decision"] == "GO"
    assert documents["stage_c_report.json"]["status"] == "BLOCKED"
    for name in ("decision.json", "stage_c_report.json", "provenance.json"):
        assert documents[name]["stage_d_started"] is False
    assert documents["decision.json"]["downstream_authorized"] is False


def test_safe_read_bytes_reads_one_regular_single_link_file(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"sealed-evidence")

    assert source._safe_read_bytes(path, "fixture") == b"sealed-evidence"


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink"])
def test_safe_read_bytes_rejects_symlinks_and_hardlinks(tmp_path, unsafe_kind):
    original = tmp_path / "original.bin"
    original.write_bytes(b"sealed-evidence")
    candidate = tmp_path / "candidate.bin"
    if unsafe_kind == "symlink":
        candidate.symlink_to(original)
    else:
        os.link(original, candidate)

    with pytest.raises(c10.StageC10IntegrityError, match="unsafe"):
        source._safe_read_bytes(candidate, "fixture")


def test_safe_read_bytes_rejects_fd_identity_change(tmp_path, monkeypatch):
    path = tmp_path / "source.bin"
    path.write_bytes(b"sealed-evidence")
    real_fstat = os.fstat
    calls = 0

    def changed_fstat(descriptor):
        nonlocal calls
        calls += 1
        info = real_fstat(descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_mode=info.st_mode, st_nlink=info.st_nlink,
                st_dev=info.st_dev, st_ino=info.st_ino, st_size=info.st_size,
                st_mtime_ns=info.st_mtime_ns + 1,
            )
        return info

    monkeypatch.setattr(source.os, "fstat", changed_fstat)
    with pytest.raises(c10.StageC10IntegrityError, match="unstable"):
        source._safe_read_bytes(path, "fixture")


def test_safe_read_bytes_uses_the_open_fd_if_the_path_is_swapped(
    tmp_path, monkeypatch
):
    path = tmp_path / "source.bin"
    old_path = tmp_path / "opened-inode.bin"
    original = b"sealed-evidence"
    path.write_bytes(original)
    real_read = os.read
    swapped = False

    def swap_after_first_read(descriptor, size):
        nonlocal swapped
        chunk = real_read(descriptor, size)
        if chunk and not swapped:
            swapped = True
            path.rename(old_path)
            path.write_bytes(b"replacement")
        return chunk

    monkeypatch.setattr(source.os, "read", swap_after_first_read)

    assert source._safe_read_bytes(path, "fixture") == original
    assert path.read_bytes() == b"replacement"


def test_c10_output_reader_reads_one_stable_regular_file(tmp_path):
    path = tmp_path / "payload.json"
    path.write_bytes(b"{}\n")

    assert c10._safe_read_regular_bytes(path, "payload") == b"{}\n"


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink"])
def test_c10_output_reader_rejects_symlink_and_hardlink(tmp_path, unsafe_kind):
    original = tmp_path / "original.json"
    original.write_bytes(b"{}\n")
    candidate = tmp_path / "candidate.json"
    if unsafe_kind == "symlink":
        candidate.symlink_to(original)
    else:
        os.link(original, candidate)

    with pytest.raises(c10.StageC10IntegrityError, match="unsafe"):
        c10._safe_read_regular_bytes(candidate, "payload")


def test_c10_output_reader_rejects_fd_identity_change(tmp_path, monkeypatch):
    path = tmp_path / "payload.json"
    path.write_bytes(b"{}\n")
    real_fstat = os.fstat
    calls = 0

    def changed_fstat(descriptor):
        nonlocal calls
        calls += 1
        info = real_fstat(descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_mode=info.st_mode, st_nlink=info.st_nlink,
                st_dev=info.st_dev, st_ino=info.st_ino, st_size=info.st_size,
                st_mtime_ns=info.st_mtime_ns + 1,
                st_ctime_ns=info.st_ctime_ns,
            )
        return info

    monkeypatch.setattr(c10.os, "fstat", changed_fstat)
    with pytest.raises(c10.StageC10IntegrityError, match="unstable"):
        c10._safe_read_regular_bytes(path, "payload")


def test_c10_output_reader_uses_open_fd_if_path_is_swapped(
    tmp_path, monkeypatch
):
    path = tmp_path / "payload.json"
    opened = tmp_path / "opened.json"
    original = b"{\"sealed\":true}\n"
    path.write_bytes(original)
    real_lstat = Path.lstat
    path_lstat_calls = 0

    def swap_before_second_path_identity_check(candidate):
        nonlocal path_lstat_calls
        if candidate == path:
            path_lstat_calls += 1
            if path_lstat_calls == 2:
                path.rename(opened)
                path.write_bytes(b"{\"replacement\":true}\n")
        return real_lstat(candidate)

    monkeypatch.setattr(Path, "lstat", swap_before_second_path_identity_check)

    with pytest.raises(c10.StageC10IntegrityError, match="unstable"):
        c10._safe_read_regular_bytes(path, "payload")


def test_c10_directory_reader_rejects_root_path_swap(tmp_path, monkeypatch):
    root = tmp_path / "c10"
    moved = tmp_path / "opened-c10"
    replacement = tmp_path / "replacement-c10"
    root.mkdir()
    replacement.mkdir()
    (root / "payload.json").write_bytes(b"{\"sealed\":true}\n")
    (replacement / "payload.json").write_bytes(b"{\"replacement\":true}\n")
    real_open_chain = c10._open_directory_chain
    open_calls = 0

    def swap_before_final_root_identity_check(candidate):
        nonlocal open_calls
        open_calls += 1
        if open_calls == 2:
            root.rename(moved)
            replacement.rename(root)
        return real_open_chain(candidate)

    monkeypatch.setattr(
        c10, "_open_directory_chain", swap_before_final_root_identity_check
    )

    with pytest.raises(c10.StageC10IntegrityError, match="unsafe, or unstable"):
        c10._safe_read_output_directory(root, {"payload.json"})


def test_c10_directory_reader_rejects_symlinked_parent(tmp_path):
    real_parent = tmp_path / "real"
    real_root = real_parent / "c10"
    real_root.mkdir(parents=True)
    (real_root / "payload.json").write_bytes(b"{}\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(c10.StageC10IntegrityError, match="unsafe, or unstable"):
        c10._safe_read_output_directory(alias / "c10", {"payload.json"})


def test_c10_producer_attestation_binds_archive_commit_tree_and_source(
    monkeypatch
):
    source_tree = "c" * 40
    source_path = "vitnns_rebuild/src/vitnns"
    calls = []

    def fake_run(arguments, **_kwargs):
        calls.append(tuple(arguments))
        if arguments[1] == "status":
            return SimpleNamespace(stdout="")
        target = arguments[-1]
        outputs = {
            f"{c10.C10_PRODUCER_ARCHIVE_REF}^{{commit}}": GIT_COMMIT,
            f"{GIT_COMMIT}^{{tree}}": GIT_TREE,
            "--show-prefix": "vitnns_rebuild/",
            f"{GIT_COMMIT}:{source_path}": source_tree,
            f"HEAD:{source_path}": source_tree,
        }
        return SimpleNamespace(stdout=outputs[target] + "\n")

    monkeypatch.setattr(c10.subprocess, "run", fake_run)

    attestation = c10._resolve_c10_producer_attestation(GIT_COMMIT)

    assert attestation == {
        "commit": GIT_COMMIT,
        "dirty": False,
        "tree": GIT_TREE,
        "archive_ref": c10.C10_PRODUCER_ARCHIVE_REF,
        "source_surface": c10.C10_CODE_SURFACE,
        "source_tree": source_tree,
    }
    assert ("git", "rev-parse", f"HEAD:{source_path}") in calls
    assert (
        "git", "status", "--porcelain", "--untracked-files=all",
        "--", c10.C10_CODE_SURFACE,
    ) in calls


def test_c10_producer_attestation_rejects_live_source_tree_drift(monkeypatch):
    source_path = "vitnns_rebuild/src/vitnns"

    def fake_run(arguments, **_kwargs):
        if arguments[1] == "status":
            return SimpleNamespace(stdout="")
        target = arguments[-1]
        outputs = {
            f"{c10.C10_PRODUCER_ARCHIVE_REF}^{{commit}}": GIT_COMMIT,
            f"{GIT_COMMIT}^{{tree}}": GIT_TREE,
            "--show-prefix": "vitnns_rebuild/",
            f"{GIT_COMMIT}:{source_path}": "c" * 40,
            f"HEAD:{source_path}": "d" * 40,
        }
        return SimpleNamespace(stdout=outputs[target] + "\n")

    monkeypatch.setattr(c10.subprocess, "run", fake_run)

    with pytest.raises(c10.StageC10IntegrityError, match="source surface"):
        c10._resolve_c10_producer_attestation(GIT_COMMIT)


def test_c10_producer_attestation_rejects_ref_change_during_resolution(
    monkeypatch
):
    source_path = "vitnns_rebuild/src/vitnns"
    ref_calls = 0

    def fake_run(arguments, **_kwargs):
        nonlocal ref_calls
        if arguments[1] == "status":
            return SimpleNamespace(stdout="")
        target = arguments[-1]
        if target == f"{c10.C10_PRODUCER_ARCHIVE_REF}^{{commit}}":
            ref_calls += 1
            return SimpleNamespace(
                stdout=(GIT_COMMIT if ref_calls == 1 else "e" * 40) + "\n"
            )
        outputs = {
            f"{GIT_COMMIT}^{{tree}}": GIT_TREE,
            "--show-prefix": "vitnns_rebuild/",
            f"{GIT_COMMIT}:{source_path}": "c" * 40,
            f"HEAD:{source_path}": "c" * 40,
        }
        return SimpleNamespace(stdout=outputs[target] + "\n")

    monkeypatch.setattr(c10.subprocess, "run", fake_run)

    with pytest.raises(c10.StageC10IntegrityError, match="changed during"):
        c10._resolve_c10_producer_attestation(GIT_COMMIT)
