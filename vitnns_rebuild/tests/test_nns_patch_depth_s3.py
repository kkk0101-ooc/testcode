from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading

import numpy as np

from vitnns import nns_patch_depth_s3 as s3
from vitnns import nns_patch_depth_s3_worker as worker


def test_low_level_formula_row_major() -> None:
    image = np.zeros((3,224,224), dtype=np.float32)
    image[0] = 0.25
    image[1] = 0.5
    image[2] = 0.75
    result = s3.low_level_features(image)
    assert result.shape == (196,5) and result.dtype == np.float32
    np.testing.assert_array_equal(result[0], np.asarray([0.5/14,0.5/14,0.25,0.5,0.75],dtype=np.float32))
    np.testing.assert_array_equal(result[14], np.asarray([0.5/14,1.5/14,0.25,0.5,0.75],dtype=np.float32))


def test_ridge_train_normalization_alpha_tie_and_roundtrip() -> None:
    rng = np.random.default_rng(17)
    images = [rng.normal(size=(196,8)) for _ in range(6)]
    targets = [image @ np.arange(8,dtype=np.float64)/32.0 + 0.2 for image in images]
    model = s3.fit_ridge(np.concatenate(images), np.concatenate(targets), 0.1)
    expected_mean = np.concatenate(images).mean(0)
    expected_scale = np.concatenate(images).std(0,ddof=0)
    np.testing.assert_allclose(model.mean, expected_mean, rtol=0, atol=0)
    np.testing.assert_allclose(model.scale, expected_scale, rtol=0, atol=0)
    assert s3.select_alpha({0.01:1.0,0.1:1.0,1.0:2.0,10.0:3.0,100.0:4.0}) == 0.01
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"array.npy"
        record = s3.write_array_atomic(path, model.weight, dtype="<f8", axes=["feature"], producer_commit="0"*40, source_identity="synthetic", recomputation_rule="synthetic")
        assert record["sha256"] == s3.sha_file(path)
        np.testing.assert_array_equal(np.load(path,allow_pickle=False),model.weight)
        scalar_path = Path(directory)/"scalar.npy"
        scalar_record = s3.write_array_atomic(scalar_path, np.asarray(0.25,dtype="<f8"), dtype="<f8", axes=[], producer_commit="0"*40, source_identity="synthetic", recomputation_rule="synthetic")
        assert scalar_record["shape"] == [] and np.load(scalar_path,allow_pickle=False).shape == ()


def test_label_shuffle_constant_feature_and_equal_image_weight() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(392,5))
    y = x[:,0]*0.2
    model = s3.fit_ridge(x,y,1.0)
    shuffled = rng.permutation(y)
    shuffled_model = s3.fit_ridge(x,shuffled,1.0)
    assert np.mean((s3.predict_ridge(model,x)-y)**2) < np.mean((s3.predict_ridge(shuffled_model,x)-y)**2)
    constant = np.ones((392,3))
    constant_model = s3.fit_ridge(constant,y,1.0)
    assert np.array_equal(constant_model.scale,np.ones(3))
    p = [np.zeros(196),np.ones(196)]
    t = [np.zeros(196),np.zeros(196)]
    assert s3.equal_image_mse(p,t) == 0.5


def test_crossfit_exact_holdout_and_split_leakage() -> None:
    source_ids = [f"source-{index}" for index in range(6)]
    rows = s3.crossfit_assignment("n00000001",source_ids)
    assert sorted(row["rank"] for row in rows) == list(range(6))
    assert sorted(row["fold"] for row in rows) == [0,0,1,1,2,2]
    assert sorted(row["source_image_id"] for row in rows) == source_ids


def _claim_schedule() -> list[dict[str, object]]:
    return [{"schedule_index":index,"source_image_id":f"s{index}","split":"probe_train"} for index in range(40)]


def test_dynamic_claim_uniqueness_terminal_skip_and_stale_claim_recovery() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        (repo/s3.ROOT_REL/"unit_terminals").mkdir(parents=True)
        launch1 = repo/"launch1"
        schedule = _claim_schedule()
        winners: list[int] = []
        lock = threading.Lock()
        def consume(worker_id: str, launch: Path) -> None:
            cursor = 0
            identity = {"worker_id":worker_id,"producer_commit":"0"*40}
            while True:
                result = worker.claim_next(repo,launch,schedule,"train_validation",identity,cursor)
                if result is None:
                    break
                row,_claim,cursor = result
                with lock:
                    winners.append(int(row["schedule_index"]))
        threads = [threading.Thread(target=consume,args=(f"w{index}",launch1)) for index in range(8)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        assert sorted(winners) == list(range(40)) and len(winners) == len(set(winners))
        # A stale claim in launch 1 does not block the same unit in fresh launch 2.
        result = worker.claim_next(repo,repo/"launch2",schedule,"train_validation",{"worker_id":"recovery","producer_commit":"0"*40},0)
        assert result is not None and result[0]["schedule_index"] == 0
        # An immutable global terminal does block recomputation in all launches.
        s3.write_json_new(repo/s3.ROOT_REL/"unit_terminals/unit-00000.json",{"payload":{"source_image_id":"s0"}})
        result2 = worker.claim_next(repo,repo/"launch3",schedule,"train_validation",{"worker_id":"recovery","producer_commit":"0"*40},0)
        assert result2 is not None and result2[0]["schedule_index"] == 1


def test_test_access_fail_closed_contract() -> None:
    spec = json.loads(Path("config/nns_vit_patch_depth/s3_run_spec.json").read_text())
    assert spec["access_order"]["test_before_weight_seal"] is False
    assert spec["parallel"]["static_sharding"] is False
    assert spec["parallel"]["eight_eligible_requires_eight_workers"] is True
    assert spec["parallel"]["existing_compute_pid_disqualifying"] is False
    assert spec["determinism"]["startup_probe"] == ["torch.matmul","backward"]
    assert spec["determinism"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
