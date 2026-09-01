from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from vitnns.constants import REPO_ROOT
from vitnns.recovery_v2 import (
    DEFAULT_BINDING_PATH, DEFAULT_BINDING_SCHEMA_PATH,
    HISTORICAL_STATE_DICT_SHA256, RecoveryV2BindingError,
    canonical_state_dict_tensor_digest, load_bound_stage_a_manifest_row,
    load_recovery_v2_binding,
    verify_recovery_v2_lfs_identity,
)
from vitnns.io_utils import sha256_file


MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_CANONICAL_TENSOR_DIGEST = (
    "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
)
MODEL_IDENTITY = "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718"


def test_recovery_v2_binding_is_closed_and_isolated_from_historical_constants():
    binding = load_recovery_v2_binding(verify_assets=False)
    schema = json.loads(DEFAULT_BINDING_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert binding.model["state_dict_sha256"] == MODEL_SHA256
    assert binding.model["canonical_tensor_digest"] == MODEL_CANONICAL_TENSOR_DIGEST
    assert binding.model["identity"] == MODEL_IDENTITY
    assert binding.model["historical_state_dict_sha256"] == HISTORICAL_STATE_DICT_SHA256
    assert binding.model["historical_artifact_recovered"] is False
    assert "49_vit" not in str(binding.model_path)
    assert binding.model_path.is_relative_to(REPO_ROOT / "assets")
    assert binding.data_root == REPO_ROOT / "assets/data/ILSVRC2012_val"


def _write_binding(tmp_path: Path, document: dict) -> dict:
    path = tmp_path / "binding.json"
    schema_path = tmp_path / "binding.schema.json"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    schema_path.write_bytes(DEFAULT_BINDING_SCHEMA_PATH.read_bytes())
    return {"path": path, "schema_path": schema_path, "repo_root": tmp_path}


def test_recovery_v2_binding_rejects_historical_alias(tmp_path: Path):
    document = json.loads(DEFAULT_BINDING_PATH.read_text(encoding="utf-8"))
    document["model"]["state_dict_sha256"] = HISTORICAL_STATE_DICT_SHA256
    with pytest.raises(RecoveryV2BindingError, match="schema failure|must not alias"):
        load_recovery_v2_binding(**_write_binding(tmp_path, document))


@pytest.mark.parametrize("field", ["state_dict_path", "data_root"])
def test_recovery_v2_binding_rejects_legacy_runtime_paths(tmp_path: Path, field: str):
    document = json.loads(DEFAULT_BINDING_PATH.read_text(encoding="utf-8"))
    section = document["model"] if field == "state_dict_path" else document["stage_c_input"]
    section[field] = "../49_vit /forbidden"
    with pytest.raises(RecoveryV2BindingError, match="schema failure|active project"):
        load_recovery_v2_binding(**_write_binding(tmp_path, document))


def test_recovery_v2_binding_rejects_unknown_fields(tmp_path: Path):
    document = json.loads(DEFAULT_BINDING_PATH.read_text(encoding="utf-8"))
    document["unexpected"] = True
    with pytest.raises(RecoveryV2BindingError, match="schema failure"):
        load_recovery_v2_binding(**_write_binding(tmp_path, document))


def test_canonical_tensor_digest_is_sorted_key_order_independent():
    torch = pytest.importorskip("torch")
    left = OrderedDict((
        ("z", torch.tensor([[1.0, -2.0]], dtype=torch.float32)),
        ("a", torch.tensor([3], dtype=torch.int64)),
    ))
    right = OrderedDict(reversed(tuple(left.items())))
    assert canonical_state_dict_tensor_digest(left) == canonical_state_dict_tensor_digest(right)


@pytest.mark.parametrize(
    "mutated",
    [
        lambda torch: {"x": torch.tensor([1.0, 3.0], dtype=torch.float32)},
        lambda torch: {"x": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        lambda torch: {"x": torch.tensor([[1.0, 2.0]], dtype=torch.float32)},
        lambda torch: {"y": torch.tensor([1.0, 2.0], dtype=torch.float32)},
    ],
)
def test_canonical_tensor_digest_binds_value_dtype_shape_and_key(mutated):
    torch = pytest.importorskip("torch")
    baseline = {"x": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    assert canonical_state_dict_tensor_digest(baseline) != (
        canonical_state_dict_tensor_digest(mutated(torch))
    )


def test_recovery_stage_a_has_dedicated_cli_without_legacy_pickle():
    from vitnns.cli import build_parser

    args = build_parser().parse_args([
        "verify-recovery-v2", "--output-dir", "evidence/stage_a/test",
    ])
    assert args.command == "verify-recovery-v2"
    assert args.binding is None
    assert args.device == "cuda:0"
    assert not hasattr(args, "compare_legacy_pickle")


def test_recovery_stage_a_cli_dispatches_and_returns_success(monkeypatch, capsys):
    from vitnns import cli
    from vitnns import recovery_stage_a

    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"numeric_gates": {"status": "PASS"}}

    monkeypatch.setattr(recovery_stage_a, "run_recovery_v2_stage_a", fake_run)
    assert cli.main([
        "verify-recovery-v2", "--output-dir", "evidence/stage_a/unit-dispatch",
        "--device", "cuda:0",
    ]) == 0
    assert observed["repo_root"] == REPO_ROOT
    assert observed["binding_path"] == DEFAULT_BINDING_PATH
    assert observed["output_dir"] == Path("evidence/stage_a/unit-dispatch")
    assert observed["device"] == "cuda:0"
    assert "Recovery-v2 Stage A verification: PASS" in capsys.readouterr().out


def test_bound_stage_a_manifest_row_is_exact():
    binding = load_recovery_v2_binding()
    row = load_bound_stage_a_manifest_row(binding)
    assert row.relative_path == binding.stage_c_input["sample_relative_path"]
    assert row.image_sha256 == binding.stage_c_input["sample_sha256"]
    assert row.wnid == "n02113978"
    assert row.imagenet1k_index == 268


def test_recovery_v2_asset_hashes_and_tensor_identity():
    from vitnns.recovery_v2 import verify_recovery_v2_state_dict

    binding = load_recovery_v2_binding(verify_assets=True)
    identity = verify_recovery_v2_state_dict(binding)
    assert identity == {
        "state_dict_sha256": MODEL_SHA256,
        "canonical_tensor_digest": MODEL_CANONICAL_TENSOR_DIGEST,
        "key_count": 152,
        "parameter_count": 86_567_656,
    }
    assert verify_recovery_v2_lfs_identity(binding) == {
        "path": "vitnns_rebuild/assets/models/vit_base_patch16_224_weights.semantic_candidate.pth",
        "oid_sha256": MODEL_SHA256,
        "hydrated": True,
    }


def test_historical_contracts_remain_bytewise_unchanged():
    expected = {
        "config/model.json": "cc8d386736b3f0ae299e75184e86772b21b8139cab0a7c0eab17a961d1ad55a3",
        "config/stage_d.json": "88fdc9cb2b4a5bd74ff7d8e68c79463cafac6437e29b5564c2115a13f1f1136d",
        "config/stage_d_run_spec.json": "43c61f190a9ad5811d878a7516039287f3914548f5444934fc06f41769c97d20",
        "config/stage_d_scope_decision.json": "217ef1fc00c3dbe971b35ff8d0793fb1eb928a7316db4ac0a5bcd34edfb6c06d",
        "selection/stage_c_selection.json": "5dff9b5aec77cab22b157b9bf2b5d63b2252b7e50ebe728250206d655a142254",
        "src/vitnns/stage_c_c9.py": "376888492c01e6b9b04399524ce0132a1c8e8fe6bf1ad879169da1dddf41e03c",
        "src/vitnns/stage_c_c10_source.py": "f23cd66151617ab058c9376c79045b007e28a2973564379b63d7e38b84a1954e",
        "config/stage_c_c9_run_spec.json": "6f36f2e1064c0377e337f88fefe7e5184f1fe24ac58e03f85ca9262bfd62f74f",
    }
    assert {path: sha256_file(REPO_ROOT / path) for path in expected} == expected


def _clean_git_state() -> dict:
    return {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "branch": "stage-d-d3-pass",
        "remote_tracking_ref": "refs/remotes/origin/stage-d-d3-pass",
        "remote_tracking_commit": "c" * 40,
        "head_matches_remote_tracking": False,
        "remote_tracking_is_ancestor": True,
        "ahead_count": 1,
        "behind_count": 0,
        "dirty": False,
        "status_entries": [],
    }


def test_recovery_stage_a_rejects_dirty_tree_before_any_other_preflight(
    monkeypatch, tmp_path: Path,
):
    from vitnns import recovery_stage_a

    dirty = _clean_git_state()
    dirty["dirty"] = True
    dirty["status_entries"] = [{"status": "??", "path": "sentinel"}]
    monkeypatch.setattr(recovery_stage_a, "capture_source_git_state", lambda _root: dirty)
    monkeypatch.setattr(
        recovery_stage_a, "_validate_output_dir",
        lambda *_a, **_kw: pytest.fail("output validation followed dirty-tree failure"),
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(recovery_stage_a.RecoveryStageAError, match="clean Git tree"):
        recovery_stage_a.run_recovery_v2_stage_a(
            repo_root=REPO_ROOT, binding_path=DEFAULT_BINDING_PATH, output_dir=output,
        )
    assert not output.exists()


def test_recovery_stage_a_rejects_output_outside_evidence_before_binding_load(
    monkeypatch, tmp_path: Path,
):
    from vitnns import recovery_stage_a

    monkeypatch.setattr(
        recovery_stage_a, "capture_source_git_state", lambda _root: _clean_git_state(),
    )
    monkeypatch.setattr(
        recovery_stage_a, "load_recovery_v2_binding",
        lambda *_a, **_kw: pytest.fail("binding loaded before output-path rejection"),
    )
    output = tmp_path / "outside-evidence"
    with pytest.raises(recovery_stage_a.RecoveryStageAError, match="repository-relative"):
        recovery_stage_a.run_recovery_v2_stage_a(
            repo_root=REPO_ROOT, binding_path=DEFAULT_BINDING_PATH, output_dir=output,
        )
    assert not output.exists()


def test_recovery_stage_a_rejects_existing_output_root(tmp_path: Path):
    from vitnns.recovery_stage_a import RecoveryStageAError, _validate_output_dir

    output = REPO_ROOT / "evidence/stage_a/unit-existing-root"
    output.mkdir(parents=True, exist_ok=False)
    with pytest.raises(RecoveryStageAError, match="not fresh"):
        _validate_output_dir(REPO_ROOT, output)
    output.rmdir()


def test_recovery_stage_a_safe_output_is_lfs_managed():
    from vitnns.recovery_stage_a import _validate_output_dir

    output = REPO_ROOT / "evidence/stage_a/unit-path-policy-must-not-exist"
    assert not output.exists()
    resolved, relative = _validate_output_dir(REPO_ROOT, output)
    assert resolved == output
    assert relative == "evidence/stage_a/unit-path-policy-must-not-exist"


def test_recovery_stage_a_allows_local_ahead_but_rejects_behind_or_diverged():
    from vitnns.recovery_stage_a import RecoveryStageAError, _require_clean_source_git

    ahead = _clean_git_state()
    _require_clean_source_git(ahead)
    for mutation in (
        {"remote_tracking_is_ancestor": False},
        {"behind_count": 1},
        {"ahead_count": 0, "head_matches_remote_tracking": False},
    ):
        invalid = {**ahead, **mutation}
        with pytest.raises(RecoveryStageAError, match="clean Git tree"):
            _require_clean_source_git(invalid)


def test_terminal_pass_is_not_published_when_git_end_gate_fails(monkeypatch, tmp_path: Path):
    from vitnns import recovery_stage_a

    output = tmp_path / "attempt"
    output.mkdir()
    monkeypatch.setattr(
        recovery_stage_a, "_require_final_git_state",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            recovery_stage_a.RecoveryStageAError("simulated Git drift")
        ),
    )
    payload = recovery_stage_a._json_bytes(recovery_stage_a._with_canonical_digest({
        "schema_version": 1,
        "document_type": "recovery_v2_stage_a_verification",
        "status": "PASS",
    }))
    with pytest.raises(recovery_stage_a.RecoveryStageAError, match="simulated Git drift"):
        recovery_stage_a._finalize_terminal(
            REPO_ROOT, _clean_git_state(), output, payload,
        )
    assert not (output / "stage_a_verification.json").exists()
    assert (output / ".stage_a_verification.pending").is_file()


def test_recovery_stage_a_conda_binding_matches_active_environment():
    from vitnns.recovery_stage_a import _validate_conda_environment

    record = _validate_conda_environment(REPO_ROOT, load_recovery_v2_binding())
    assert record["prefix_match"] is True
    assert record["executable_under_prefix"] is True
    assert record["versions_match"] is True
    assert record["observed_versions"] == {
        "python": "3.10.21",
        "torch": "2.2.2+cu121",
        "torchvision": "0.17.2+cu121",
        "timm": "0.9.16",
    }


def test_recovery_stage_a_refuses_cpu_machine_binding():
    from vitnns.recovery_stage_a import RecoveryStageAError, _resolve_device

    with pytest.raises(RecoveryStageAError, match="requires an explicit CUDA"):
        _resolve_device("cpu")


@pytest.mark.parametrize(
    ("lfs_line", "message"),
    [
        ("", "no unique Git LFS entry"),
        (f"{'0' * 64} * vitnns_rebuild/assets/models/vit_base_patch16_224_weights.semantic_candidate.pth\n", "OID mismatch"),
        (f"{MODEL_SHA256} - vitnns_rebuild/assets/models/vit_base_patch16_224_weights.semantic_candidate.pth\n", "not hydrated"),
    ],
)
def test_recovery_v2_lfs_identity_fails_closed(monkeypatch, lfs_line: str, message: str):
    from vitnns import recovery_v2

    responses = iter((
        SimpleNamespace(stdout=str(REPO_ROOT.parent) + "\n"),
        SimpleNamespace(stdout=lfs_line),
    ))
    monkeypatch.setattr(recovery_v2.subprocess, "run", lambda *_a, **_kw: next(responses))
    with pytest.raises(RecoveryV2BindingError, match=message):
        verify_recovery_v2_lfs_identity(load_recovery_v2_binding())
