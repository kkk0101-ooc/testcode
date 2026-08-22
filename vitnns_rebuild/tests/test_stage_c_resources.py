import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import vitnns.stage_c_resources as resources
from vitnns.stage_c_resources import (
    CUDA_MEASURED,
    CUDA_MEASUREMENT_FAILED,
    CUDA_NOT_APPLICABLE,
    PAYLOAD_FILE_LIST_SCOPE,
    PHASE_CLOCK,
    PHASE_SYNCHRONIZATION,
    REQUIRED_PHASES,
    ROOT_TREE_SCOPE,
    TOTAL_BOUNDARY,
    CallCounter,
    PhaseTimer,
    ResourceMeter,
    measurement_environment,
    measure_artifact_bytes,
    measure_artifact_tree_bytes,
)


def test_cpu_phase_timer_accumulates_required_phases_and_records_failure():
    clock = iter((10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 18.0, 19.0, 21.0, 30.0))
    timer = PhaseTimer(device="cpu", clock=lambda: next(clock))
    counter = CallCounter()
    counter.record_forward(2)
    counter.record_backward(3)
    calls_before = counter.snapshot()

    timer.start()
    with timer.measure("attribution_or_map_compute"):
        pass
    with timer.measure("attribution_or_map_compute"):
        pass
    with timer.measure("fidelity_compute"):
        pass
    with pytest.raises(LookupError, match="synthetic phase failure"):
        with timer.measure("integrity_checking"):
            raise LookupError("synthetic phase failure")
    timer.stop()

    payload = timer.to_dict()
    assert payload == {
        "clock": PHASE_CLOCK,
        "synchronization": PHASE_SYNCHRONIZATION,
        "phases": {
            "attribution_or_map_compute": {
                "wall_time_seconds": 3.0,
                "invocations": 2,
            },
            "fidelity_compute": {
                "wall_time_seconds": 2.0,
                "invocations": 1,
            },
            "integrity_checking": {
                "wall_time_seconds": 2.0,
                "invocations": 1,
            },
            "artifact_finalize": {
                "wall_time_seconds": 0.0,
                "invocations": 0,
            },
        },
        "synchronized_total_wall_time_seconds": 20.0,
        "total_boundary": TOTAL_BOUNDARY,
        "phases_are_mutually_exclusive": False,
    }
    assert tuple(payload["phases"]) == REQUIRED_PHASES
    assert all(
        payload["synchronized_total_wall_time_seconds"]
        >= phase["wall_time_seconds"]
        for phase in payload["phases"].values()
    )
    assert counter.snapshot() == calls_before
    assert json.loads(json.dumps(payload)) == payload


def test_cuda_phase_timer_synchronizes_total_and_both_sides_of_every_phase(
    monkeypatch,
):
    events = []
    readings = iter(float(value) for value in range(10))
    device = torch.device("cuda:2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda actual: events.append(("sync", actual)),
    )

    def clock():
        value = next(readings)
        events.append(("clock", value))
        return value

    timer = PhaseTimer(device=device, clock=clock)
    timer.start()
    for phase in REQUIRED_PHASES:
        with timer.measure(phase):
            events.append(("body", phase))
    timer.stop()

    expected = [("sync", device), ("clock", 0.0)]
    for offset, phase in enumerate(REQUIRED_PHASES, start=1):
        expected.extend(
            [
                ("sync", device),
                ("clock", float(2 * offset - 1)),
                ("body", phase),
                ("sync", device),
                ("clock", float(2 * offset)),
            ]
        )
    expected.extend((("sync", device), ("clock", 9.0)))
    assert events == expected
    assert timer.to_dict()["synchronized_total_wall_time_seconds"] == 9.0
    assert all(
        phase == {"wall_time_seconds": 1.0, "invocations": 1}
        for phase in timer.to_dict()["phases"].values()
    )


def test_phase_timer_rejects_unknown_phase_and_invalid_clocks():
    with pytest.raises(TypeError, match="clock must be callable"):
        PhaseTimer(device="cpu", clock=None)

    invalid_reading = PhaseTimer(device="cpu", clock=lambda: float("nan"))
    with pytest.raises(RuntimeError, match="total start clock reading is not finite"):
        invalid_reading.start()

    timer = PhaseTimer(device="cpu", clock=iter((0.0, 1.0)).__next__)
    timer.start()
    with pytest.raises(ValueError, match="unknown execution-unit phase"):
        with timer.measure("not_a_phase"):
            pass
    timer.stop()


def test_measurement_environment_reports_cpu_and_mocked_cuda(monkeypatch):
    assert measurement_environment("cpu") == {
        "device_type": "cpu",
        "device": "cpu",
        "device_name": None,
        "cuda_runtime_version": None,
        "cudnn_version": None,
    }

    requested = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda device: requested.append(device) or "Mock Accelerator",
    )
    monkeypatch.setattr(torch.version, "cuda", "99.1")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 90123)

    assert measurement_environment("cuda") == {
        "device_type": "cuda",
        "device": "cuda:2",
        "device_name": "Mock Accelerator",
        "cuda_runtime_version": "99.1",
        "cudnn_version": 90123,
    }
    assert requested == [torch.device("cuda:2")]


def test_cpu_boundary_records_wall_calls_files_and_explicit_no_cuda(monkeypatch, tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"abc")
    second.write_bytes(b"12345")
    clock = iter((10.0, 12.5))
    monkeypatch.setattr(resources.time, "perf_counter", lambda: next(clock))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a CPU boundary must not call a CUDA API")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", forbidden)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", forbidden)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", forbidden)

    meter = ResourceMeter(device="cpu", artifact_files=(first, second))
    with meter:
        meter.record_forward(3)
        meter.record_backward(2)

    record = meter.record
    assert record.status == "completed"
    assert record.wall_time_seconds == 2.5
    assert record.cuda_status == CUDA_NOT_APPLICABLE
    assert record.cuda_device is None
    assert record.cuda_peak_allocated_bytes is None
    assert record.cuda_peak_reserved_bytes is None
    assert record.forward_calls == 3
    assert record.backward_calls == 2
    assert record.artifact_bytes == 8
    assert record.artifacts.scope == PAYLOAD_FILE_LIST_SCOPE
    assert [item.bytes for item in record.artifacts.files] == [3, 5]
    payload = record.to_dict()
    assert payload["cuda"] == {
        "status": "not_applicable_cpu",
        "device": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    assert json.loads(json.dumps(payload)) == payload
    with pytest.raises(FrozenInstanceError):
        record.status = "failed"


def test_exception_is_recorded_after_boundary_and_propagated(monkeypatch):
    clock = iter((2.0, 2.25))
    monkeypatch.setattr(resources.time, "perf_counter", lambda: next(clock))
    meter = ResourceMeter(device="cpu")

    with pytest.raises(LookupError, match="synthetic failure"):
        with meter:
            meter.record_forward()
            raise LookupError("synthetic failure")

    assert meter.record.status == "failed"
    assert meter.record.forward_calls == 1
    assert meter.record.to_dict()["exception"] == {
        "type": "builtins.LookupError",
        "message": "synthetic failure",
    }
    assert meter.record.instrumentation_errors == ()


def test_body_exception_wins_over_exit_artifact_failure_and_both_are_recorded():
    meter = ResourceMeter(device="cpu", artifact_files=("missing-artifact",))
    with pytest.raises(LookupError, match="primary body failure") as caught:
        with meter:
            meter.record_forward()
            raise LookupError("primary body failure")

    record = meter.record
    assert record.status == "failed"
    assert record.exception_type == "builtins.LookupError"
    assert record.artifact_bytes == 0
    assert [item.phase for item in record.instrumentation_errors] == [
        "artifact_payload_measurement"
    ]
    assert "regular file" in record.instrumentation_errors[0].message
    retained = caught.value.stage_c_resource_instrumentation_errors
    assert retained[0]["phase"] == "artifact_payload_measurement"
    if hasattr(caught.value, "__notes__"):
        assert any(
            "resource instrumentation also failed" in note
            for note in caught.value.__notes__
        )
    assert json.loads(json.dumps(record.to_dict())) == record.to_dict()


def test_add_note_failure_never_masks_body_exception_and_is_retained():
    class AnnotationFailure(LookupError):
        def add_note(self, note):
            raise RuntimeError("annotation failed")

    meter = ResourceMeter(device="cpu", artifact_files=("missing-artifact",))
    primary = AnnotationFailure("primary body failure")
    with pytest.raises(AnnotationFailure, match="primary body failure") as caught:
        with meter:
            raise primary

    assert caught.value is primary
    assert meter.record.exception_type.endswith(".AnnotationFailure")
    assert [item.phase for item in meter.record.instrumentation_errors] == [
        "artifact_payload_measurement",
        "exception_annotation",
    ]
    assert meter.record.instrumentation_errors[1].exception_type == (
        "builtins.RuntimeError"
    )
    assert meter.record.instrumentation_errors[1].message == "annotation failed"
    assert [item["phase"] for item in primary.stage_c_resource_instrumentation_errors] == [
        "artifact_payload_measurement",
        "exception_annotation",
    ]
    json.dumps(meter.record.to_dict(), allow_nan=False)


def test_exit_instrumentation_failure_without_body_error_raises_and_leaves_record():
    meter = ResourceMeter(device="cpu", artifact_files=("missing-artifact",))
    with pytest.raises(FileNotFoundError, match="regular file"):
        with meter:
            pass
    assert meter.record.status == "failed"
    assert meter.record.exception_type is None
    assert meter.record.instrumentation_errors[0].phase == "artifact_payload_measurement"


def test_nonfinite_clock_does_not_mask_body_exception(monkeypatch):
    clock = iter((1.0, float("nan")))
    monkeypatch.setattr(resources.time, "perf_counter", lambda: next(clock))
    meter = ResourceMeter(device="cpu")
    with pytest.raises(ValueError, match="primary"):
        with meter:
            raise ValueError("primary")
    assert meter.record.wall_time_seconds is None
    assert meter.record.instrumentation_errors[0].phase == "wall_clock_after"


def test_record_unavailable_early_meter_is_single_use_and_counts_validate():
    meter = ResourceMeter(device="cpu")
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = meter.record
    with pytest.raises(RuntimeError, match="inside the boundary"):
        meter.record_forward()
    counter = CallCounter()
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            counter.record_forward(bad)
    with meter:
        pass
    with pytest.raises(RuntimeError, match="single-use"):
        with meter:
            pass
    with pytest.raises(RuntimeError, match="inside the boundary"):
        meter.record_backward()


def test_shared_counter_reports_only_calls_inside_this_boundary():
    counter = CallCounter()
    counter.record_forward(4)
    meter = ResourceMeter(device="cpu", call_counter=counter)
    with meter:
        counter.record_forward(2)
        counter.record_backward(3)
    assert meter.record.calls.to_dict() == {
        "accounting_unit": "batched_model_invocation",
        "forward_calls": 2,
        "backward_calls": 3,
    }
    assert counter.snapshot().forward_calls == 6


def test_call_counter_wrappers_count_actual_callable_invocations():
    counter = CallCounter()
    forward = counter.counted_forward(lambda value: value + 1)
    backward = counter.counted_backward(lambda value: value * 2)
    assert forward(3) == 4
    assert forward(7) == 8
    assert backward(5) == 10
    assert counter.snapshot().to_dict() == {
        "accounting_unit": "batched_model_invocation",
        "forward_calls": 2, "backward_calls": 1,
    }


def test_artifact_measurement_requires_explicit_unique_regular_files(tmp_path):
    artifact = tmp_path / "artifact.dat"
    artifact.write_bytes(b"abcd")
    empty = measure_artifact_bytes(())
    assert empty.total_bytes == 0
    measured = measure_artifact_bytes([artifact])
    assert measured.total_bytes == 4
    assert measured.scope == PAYLOAD_FILE_LIST_SCOPE
    assert measured.root is None
    assert measured.files[0].path == str(artifact.resolve())
    assert json.loads(json.dumps(measured.to_dict())) == measured.to_dict()

    with pytest.raises(TypeError, match="explicit iterable"):
        measure_artifact_bytes(artifact)
    with pytest.raises(ValueError, match="duplicate"):
        measure_artifact_bytes((artifact, str(artifact)))
    with pytest.raises(FileNotFoundError, match="regular file"):
        measure_artifact_bytes((tmp_path,))
    with pytest.raises(FileNotFoundError, match="regular file"):
        measure_artifact_bytes((tmp_path / "missing",))


def test_exhaustive_root_tree_bytes_are_distinct_from_explicit_payload_scope(tmp_path):
    root = tmp_path / "tree"
    nested = root / "nested"
    nested.mkdir(parents=True)
    payload = root / "payload.bin"
    control = nested / "status.json"
    payload.write_bytes(b"123")
    control.write_bytes(b"12345")

    selected = measure_artifact_bytes((payload,))
    exhaustive = measure_artifact_tree_bytes(root)
    assert selected.total_bytes == 3
    assert selected.scope == PAYLOAD_FILE_LIST_SCOPE
    assert exhaustive.total_bytes == 8
    assert exhaustive.scope == ROOT_TREE_SCOPE
    assert exhaustive.root == str(root.resolve())
    assert [Path(item.path).name for item in exhaustive.files] == [
        "status.json", "payload.bin"
    ]
    assert json.loads(json.dumps(exhaustive.to_dict())) == exhaustive.to_dict()

    link = root / "alias"
    try:
        link.symlink_to(payload)
    except OSError:
        pytest.skip("filesystem does not permit symlinks")
    with pytest.raises(ValueError, match="symlink"):
        measure_artifact_tree_bytes(root)


def test_cuda_boundary_synchronizes_resets_then_reads_both_peaks(monkeypatch):
    events = []
    device = torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda actual: events.append(("synchronize", actual))
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda actual: events.append(("reset", actual)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda actual: events.append(("allocated", actual)) or 123,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda actual: events.append(("reserved", actual)) or 456,
    )
    clock = iter((5.0, 6.0))
    monkeypatch.setattr(resources.time, "perf_counter", lambda: next(clock))

    meter = ResourceMeter(device=device)
    with meter:
        meter.record_forward()
        meter.record_backward()

    assert events == [
        ("synchronize", device),
        ("reset", device),
        ("synchronize", device),
        ("allocated", device),
        ("reserved", device),
    ]
    assert meter.record.cuda_status == CUDA_MEASURED
    assert meter.record.cuda_device == "cuda:1"
    assert meter.record.cuda_peak_allocated_bytes == 123
    assert meter.record.cuda_peak_reserved_bytes == 456


def _mock_cuda_measurement(monkeypatch, events, *, current_device=1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append(("sync", str(device)))
    )
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats",
        lambda device: events.append(("reset", str(device))),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 10)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 20)


def test_bare_cuda_resolves_to_concrete_device_and_overlap_is_rejected(monkeypatch):
    events = []
    _mock_cuda_measurement(monkeypatch, events, current_device=1)
    outer = ResourceMeter(device="cuda")
    overlapping = ResourceMeter(device="cuda:1")
    with outer:
        assert outer.device == torch.device("cuda:1")
        with pytest.raises(RuntimeError, match="already active for cuda:1"):
            with overlapping:
                pass
    assert outer.record.cuda_device == "cuda:1"
    assert overlapping.device == torch.device("cuda:1")
    with overlapping:
        pass
    assert overlapping.record.cuda_status == CUDA_MEASURED


def test_cuda_startup_failure_releases_owner_and_same_meter_can_retry(monkeypatch):
    calls = {"sync": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def synchronize(_device):
        calls["sync"] += 1
        if calls["sync"] == 1:
            raise RuntimeError("startup sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 2)
    meter = ResourceMeter(device="cuda")
    with pytest.raises(RuntimeError, match="startup sync failed"):
        with meter:
            pass
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = meter.record
    with meter:
        pass
    assert meter.record.cuda_device == "cuda:0"


def test_body_exception_wins_over_cuda_exit_failure_and_owner_is_released(monkeypatch):
    calls = {"sync": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def synchronize(_device):
        calls["sync"] += 1
        if calls["sync"] == 2:
            raise RuntimeError("exit sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 2)
    first = ResourceMeter(device="cuda")
    with pytest.raises(KeyError, match="body"):
        with first:
            raise KeyError("body")
    assert first.record.cuda_status == CUDA_MEASUREMENT_FAILED
    assert first.record.instrumentation_errors[0].phase == "cuda_synchronize_after"

    second = ResourceMeter(device="cuda:0")
    with second:
        pass
    assert second.record.cuda_status == CUDA_MEASURED


def test_negative_cuda_peak_is_instrumentation_failure_not_body_mask(monkeypatch):
    events = []
    _mock_cuda_measurement(monkeypatch, events, current_device=0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: -1)
    meter = ResourceMeter(device="cuda:0")
    with pytest.raises(IndexError, match="primary"):
        with meter:
            raise IndexError("primary")
    assert meter.record.cuda_status == CUDA_MEASUREMENT_FAILED
    assert meter.record.cuda_peak_allocated_bytes is None
    assert meter.record.instrumentation_errors[0].phase == "cuda_peak_allocated"


def test_unavailable_cuda_fails_before_timing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    meter = ResourceMeter(device="cuda:0")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        with meter:
            pass
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = meter.record
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)
    with meter:
        pass
    assert meter.record.cuda_status == CUDA_MEASURED


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tiny_real_cuda_measurement():
    device = torch.device("cuda", torch.cuda.current_device())
    meter = ResourceMeter(device=device)
    with meter:
        value = torch.ones(4, device=device)
        assert float(value.sum().item()) == 4.0
        meter.record_forward()
    assert meter.record.cuda_status == CUDA_MEASURED
    assert meter.record.cuda_peak_allocated_bytes >= value.numel() * value.element_size()
    assert meter.record.cuda_peak_reserved_bytes >= meter.record.cuda_peak_allocated_bytes
