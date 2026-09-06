"""E3-R2 budget-only continuation tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator

import vitnns.stage_e_e3_r2 as r2
from vitnns.attribution import IGResult

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_is_budget_only_and_keeps_scientific_contract() -> None:
    config = r2._validate_config(REPO_ROOT)
    assert config["canonical_digest"] == r2.EXPECTED_CONFIG_DIGEST
    assert config["budget"] == {
        "gpu_hours_max": 128.0, "cpu_hours_max": 512.0,
        "retained_bytes_max": 34359738368,
        "e3_v1_and_r1_cost_included": True, "waiver_allowed": False,
    }
    assert tuple(config["numerical_recovery"]["complete_schedule"]) == r2.COMPLETE_SCHEDULE
    assert config["numerical_recovery"]["atol"] == 1e-4
    assert config["numerical_recovery"]["rtol"] == 0.01
    assert config["scientific_contract"]["seed_change"] is False
    assert config["scientific_contract"]["fidelity_change"] is False
    assert config["scientific_contract"]["statistics_change"] is False
    Draft202012Validator.check_schema(json.loads((REPO_ROOT/r2.COMPLETION_SCHEMA_PATH).read_text()))


def test_upstream_e3_fail_r1_blocked_and_528_2472_partition() -> None:
    upstream = r2.validate_upstream(REPO_ROOT)
    inherited, eligible = r2._partition(REPO_ROOT)
    assert upstream["e3_v1"]["status"] == "FAIL"
    assert upstream["e3_r1"]["status"] == "BLOCKED"
    assert Counter(value[0] for value in inherited.values()) == Counter({"E3_V1_INHERITED":516,"E3_R1_INHERITED":12})
    assert len(eligible) == 2472
    assert not {unit["selection_order"] for unit in eligible} & set(r2.r1.PILOT_ORDERS)


def test_r2_attempt_path_is_fresh_and_cannot_alias_upstream() -> None:
    unit = {"canonical_digest": "a"*64}
    path = r2._r2_attempt_path(REPO_ROOT, unit)
    assert r2.ROOT_PATH in path.as_posix() and r2.ATTEMPT_ID in path.as_posix()
    assert r2.r1.ROOT_PATH not in path.as_posix() and r2.e3.ROOT_PATH not in path.as_posix()


def test_shards_cover_only_exact_2472_complement() -> None:
    _, eligible = r2._partition(REPO_ROOT)
    by_shard = {index: {unit["canonical_digest"] for unit in eligible if unit["selection_order"] % 8 == index} for index in range(8)}
    assert sum(len(value) for value in by_shard.values()) == 2472
    assert not any(by_shard[left] & by_shard[right] for left in range(8) for right in range(left+1,8))


@pytest.mark.parametrize("shard_id,shard_count", [(-1,8),(8,8),(0,7),(True,8)])
def test_invalid_shard_identity_fails_closed(shard_id: int, shard_count: int) -> None:
    with pytest.raises(r2.StageEE3R2Error): r2.run_shard(REPO_ROOT,shard_id=shard_id,shard_count=shard_count,device="cpu")


def test_no_e4_execution_surface() -> None:
    public={name for name in dir(r2) if name.startswith(("run_","start_","execute_"))}
    assert public=={"run_shard"}


def test_full_union_rejects_a_missing_terminal_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = {"units": [{"canonical_digest": "a" * 64}]}
    monkeypatch.setattr(r2, "_v1_plan", lambda _repo: plan)
    monkeypatch.setattr(r2, "_source_attempt", lambda _repo, _unit: None)
    with pytest.raises(r2.StageEE3R2Error, match="missing terminal union"):
        r2._all_sources(REPO_ROOT)


def test_enlarged_budget_still_has_hard_limits() -> None:
    assert r2.GPU_HOURS_MAX == 128.0
    assert r2.CPU_SECONDS_MAX == 512.0 * 3600.0
    assert r2.RETAINED_BYTES_MAX == 32 * 1024**3


def test_cross_root_publish_validates_r2_origin(tmp_path: Path) -> None:
    unit={"canonical_digest":"a"*64,"sample_id":"sample","selection_order":3,"stratum":"correct","block":4,"head":2,"method":r2.e3.NNS_METHOD,"target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","coordinate_comparison_identity":r2.e3.comparison_identity(4,2),"image_sha256":"b"*64}
    attempts=tuple({"steps":step,"delta":[1.0],"sum":[0.0],"abs":[1.0],"rel":[1.0],"pass":[False]} for step in r2.COMPLETE_SCHEDULE)
    ig=IGResult(attribution=torch.ones((1,3,224,224),dtype=torch.float32),conservation={"delta":torch.tensor([1.0]),"sum":torch.tensor([0.0]),"abs":torch.tensor([1.0]),"rel":torch.tensor([1.0]),"pass":torch.tensor([False])},steps=32768,converged=False,attempts=attempts,metadata={"step_schedule":list(r2.COMPLETE_SCHEDULE),"step_chunk_size":8,"atol":1e-4,"rtol":0.01})
    diagnostic=np.ones((3,224,224),dtype=np.float32); target={"frozen_target":{"runtime_model_binding":{"sha256":"x"}}}; response=SimpleNamespace(to_dict=lambda:{"sigma":1})
    result=r2.r1._failure_result(unit=unit,target_identity=target,response=response,error=SimpleNamespace(ig_result=ig),elapsed_seconds=1.0,cpu_seconds=2.0,forward_calls=7,peak_cuda_bytes=8,clean_sha256="c"*64,diagnostic=diagnostic,recovery_id=r2.RECOVERY_ID,attempt_id=r2.ATTEMPT_ID,origin="E3_R2_NEW")
    path=r2.r1._publish_attempt(tmp_path,unit,result,diagnostic,root_path=r2.ROOT_PATH,attempt_id=r2.ATTEMPT_ID,recovery_id=r2.RECOVERY_ID)
    observed=r2.r1._validate_r1_attempt(tmp_path,unit,root_path=r2.ROOT_PATH,attempt_id=r2.ATTEMPT_ID,recovery_id=r2.RECOVERY_ID,origin="E3_R2_NEW")
    assert path.is_dir() and observed["result"]["origin"]=="E3_R2_NEW"
