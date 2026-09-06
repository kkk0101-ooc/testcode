"""E3-R2 budget-only continuation tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import vitnns.stage_e_e3_r2 as r2

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
