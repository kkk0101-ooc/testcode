"""Non-interactive JSON CLI for Stage C planning and orchestration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

from .constants import DEFAULT_STATE_DICT, REPO_ROOT
from .io_utils import sha256_file
from .runner import seed_everything
from .stage_c_config import load_stage_c_config
from .stage_c_provenance import capture_git_state
from .stage_c_benchmark import (
    StageCBenchmarkBlockedError,
    StageCBenchmarkIntegrityError,
    capture_benchmark_environment,
    require_fresh_c7_output_root,
    require_c7_launch_gate,
    validate_c7_diagnostic_plan,
    validate_c7_execution_plan,
    write_benchmark_report,
    write_c7_diagnostic_gate,
    write_c7_launch_authorization,
)
from .stage_c_plan import (
    build_execution_plan,
    estimate_stage_c_run,
    load_run_spec,
    load_stage_c_registry,
    load_stage_c_selection,
)
from .stage_c_runner import (
    ProductionStageCBackend,
    StageCBatchValidationError,
    aggregate_stage_c_batch,
    run_stage_c_batch,
    run_stage_c_sample,
    validate_stage_c_batch,
)


EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_PREFLIGHT_BLOCKED = 2
EXIT_TERMINAL_FAILURE = 3
EXIT_INTEGRITY_FAILURE = 4
STAGE_C_SEED = 20260821


class ArgumentError(ValueError):
    """Raised instead of allowing argparse to emit non-JSON errors."""


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ArgumentError(message)


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--data-root", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(
        prog="vitnns-stage-c",
        description="Auditable Stage C planning, execution, validation, and L0 aggregation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    estimate = commands.add_parser("estimate")
    _add_plan_arguments(estimate)

    for name in ("run-sample", "run-batch"):
        command = commands.add_parser(name)
        _add_plan_arguments(command)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT)
        command.add_argument(
            "--no-resume", action="store_true",
            help="always allocate a new attempt, preserving every old attempt",
        )
        command.add_argument(
            "--c7-diagnostic-gate", type=Path,
            help=(
                "required for the exact formal C7 plan; must name a passing "
                "report from the separate one-unit diagnostic workload"
            ),
        )
        if name == "run-sample":
            command.add_argument("--sample-id", required=True)

    benchmark = commands.add_parser("benchmark")
    _add_plan_arguments(benchmark)
    benchmark.add_argument("--output-root", type=Path, required=True)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT)
    benchmark.add_argument("--c7-diagnostic-gate", type=Path, required=True)

    for name in ("validate", "aggregate", "c7-diagnostic-gate", "benchmark-report"):
        command = commands.add_parser(name)
        _add_plan_arguments(command)
        command.add_argument("--output-root", type=Path, required=True)
        if name in {"aggregate", "c7-diagnostic-gate", "benchmark-report"}:
            command.add_argument("--output", type=Path)
        if name in {"c7-diagnostic-gate", "benchmark-report"}:
            command.add_argument("--device", default="cuda:0")
        if name == "benchmark-report":
            command.add_argument("--c7-diagnostic-gate", type=Path, required=True)
    return parser


def _write_json(value: Any, stream: TextIO) -> None:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    print(
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ),
        file=stream,
    )


def _load_pipeline(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, Any]:
    config = load_stage_c_config(args.config)
    registry_kwargs: dict[str, Any] = {}
    if args.registry is not None:
        registry_kwargs["registry_path"] = args.registry
    if args.data_root is not None:
        registry_kwargs["data_root"] = args.data_root
    registry = load_stage_c_registry(config, **registry_kwargs)
    selection = load_stage_c_selection(
        config, selection_path=args.selection, required=False
    )
    run_spec = load_run_spec(args.run_spec)
    plan = build_execution_plan(
        config, registry, run_spec, selection=selection
    )
    estimate = estimate_stage_c_run(config, plan)
    return config, registry, selection, run_spec, plan, estimate


def _default_backend_factory(
    *, config: Any, registry: Any, plan: Any, args: argparse.Namespace
) -> ProductionStageCBackend:
    rows = registry.by_sample_id()
    data_root = Path(registry.data_root).resolve()

    def load_sample(sample: Any, model: Any) -> dict[str, Any]:
        import torch
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transform_functional
        from .modeling import canonical_preprocess_spec

        sample_id = getattr(sample, "sample_id", None)
        if sample_id not in rows:
            raise ValueError(f"sample is absent from the registry: {sample_id}")
        row = rows[sample_id]
        image_path = data_root / row.relative_path
        if sha256_file(image_path) != row.image_sha256:
            raise ValueError(f"image SHA-256 mismatch: {row.sample_id}")
        specification = canonical_preprocess_spec(model)
        if (
            specification.input_size != (3, 224, 224)
            or specification.resize_shorter_side != 248
            or specification.center_crop != (224, 224)
            or specification.interpolation != "bicubic"
            or specification.mean != (0.5, 0.5, 0.5)
            or specification.std != (0.5, 0.5, 0.5)
        ):
            raise RuntimeError("canonical preprocessing contract drifted")
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            resized = transform_functional.resize(
                rgb, specification.resize_shorter_side,
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            )
            cropped = transform_functional.center_crop(
                resized, list(specification.center_crop)
            )
            canonical = transform_functional.pil_to_tensor(cropped).to(
                dtype=torch.float32
            ).div(255.0).unsqueeze(0)
        mean = torch.tensor(specification.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(specification.std, dtype=torch.float32).view(1, 3, 1, 1)
        normalized = (canonical - mean) / std
        return {
            "canonical_rgb": canonical,
            "normalized_input": normalized,
            "registry_row": row,
            "image_path": image_path,
        }

    return ProductionStageCBackend(
        sample_loader=load_sample, state_dict_path=args.state_dict,
        config=config, registry=registry, plan=plan, config_path=args.config,
    )


def _make_backend(
    factory: Callable[..., Any],
    *, config: Any, registry: Any, plan: Any, args: argparse.Namespace,
) -> Any:
    return factory(config=config, registry=registry, plan=plan, args=args)


def _is_exact_c7_plan(plan: Any, config: Any) -> bool:
    """Recognize the frozen C7 plan without changing other run semantics."""

    try:
        result = validate_c7_execution_plan(plan, config)
    except (TypeError, ValueError):
        return False
    return bool(result.get("valid")) if isinstance(result, dict) else False


def _is_exact_c7_diagnostic_plan(plan: Any, config: Any) -> bool:
    try:
        result = validate_c7_diagnostic_plan(plan, config)
    except (TypeError, ValueError):
        return False
    return bool(result.get("valid")) if isinstance(result, dict) else False


def _default_report_path(output_root: Path, suffix: str) -> Path:
    root = Path(output_root)
    return root.with_name(root.name + suffix)


def _configure_c7_execution(device: str) -> None:
    """Establish the frozen deterministic boundary for C7 only."""

    import torch

    parsed = torch.device(device)
    if parsed.type == "cuda" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "C7 CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "(or :16:8) before process start"
        )
    seed_everything(STAGE_C_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if parsed.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def main(
    argv: list[str] | None = None,
    *,
    backend_factory: Callable[..., Any] | None = None,
) -> int:
    try:
        try:
            args = build_parser().parse_args(argv)
        except SystemExit as error:
            return EXIT_SUCCESS if error.code == 0 else EXIT_INPUT_ERROR

        config, registry, _selection, _spec, plan, estimate = _load_pipeline(args)
        if args.command == "estimate":
            _write_json(estimate, sys.stdout)
            return EXIT_SUCCESS if estimate.within_budget else EXIT_PREFLIGHT_BLOCKED

        if args.command == "validate":
            result = validate_stage_c_batch(args.output_root, plan)
            _write_json(result, sys.stdout)
            if not result.valid:
                return EXIT_INTEGRITY_FAILURE
            return EXIT_SUCCESS if result.successful else EXIT_TERMINAL_FAILURE

        if args.command == "aggregate":
            result = aggregate_stage_c_batch(
                args.output_root, plan, output_path=args.output
            )
            _write_json(result, sys.stdout)
            return (
                EXIT_SUCCESS if result.get("successful", True)
                else EXIT_TERMINAL_FAILURE
            )

        if args.command == "c7-diagnostic-gate":
            destination = args.output or _default_report_path(
                args.output_root, ".c7-diagnostic-gate.json"
            )
            _configure_c7_execution(args.device)
            git_state = capture_git_state(REPO_ROOT)
            result = write_c7_diagnostic_gate(
                destination, args.output_root, plan, config=config,
                environment=capture_benchmark_environment(args.device),
                run_spec_path=args.run_spec,
                run_spec_file_sha256=sha256_file(args.run_spec),
                observed_git_state=git_state,
            )
            _write_json(result, sys.stdout)
            return (
                EXIT_SUCCESS
                if result.get("status") == "PASS"
                else EXIT_PREFLIGHT_BLOCKED
            )

        if args.command == "benchmark-report":
            destination = args.output or _default_report_path(
                args.output_root, ".benchmark-report.json"
            )
            _configure_c7_execution(args.device)
            git_state = capture_git_state(REPO_ROOT)
            result = write_benchmark_report(
                destination, args.output_root, plan, config=config,
                environment=capture_benchmark_environment(args.device),
                run_spec_path=args.run_spec,
                run_spec_file_sha256=sha256_file(args.run_spec),
                recomputed_estimate=estimate,
                observed_git_state=git_state,
                c7_diagnostic_gate=args.c7_diagnostic_gate,
            )
            _write_json(result, sys.stdout)
            if result.get("status") == "PASS":
                return EXIT_SUCCESS
            exit_class = result.get("exit_class")
            if exit_class == "integrity":
                return EXIT_INTEGRITY_FAILURE
            if exit_class == "preflight":
                return EXIT_PREFLIGHT_BLOCKED
            return EXIT_TERMINAL_FAILURE

        if args.command == "benchmark" and not _is_exact_c7_plan(plan, config):
            raise ArgumentError(
                "benchmark requires the exact checked-in C7 execution scope"
            )
        is_c7 = _is_exact_c7_plan(plan, config)
        is_c7_diagnostic = _is_exact_c7_diagnostic_plan(plan, config)
        authorization = None
        if is_c7:
            if args.c7_diagnostic_gate is None:
                _write_json(
                    {
                        "document_type": "stage_c_c7_launch_gate",
                        "status": "BLOCKED",
                        "blocked_reasons": [
                            "the exact C7 run requires --c7-diagnostic-gate"
                        ],
                    },
                    sys.stdout,
                )
                return EXIT_PREFLIGHT_BLOCKED
            git_state = capture_git_state(REPO_ROOT)
            authorization = require_c7_launch_gate(
                args.c7_diagnostic_gate,
                plan,
                config=config,
                output_root=args.output_root,
                observed_git_state=git_state,
                run_spec_path=args.run_spec,
            )
        elif is_c7_diagnostic:
            require_fresh_c7_output_root(
                args.output_root, kind="diagnostic"
            )
        if not estimate.within_budget:
            _write_json(estimate, sys.stdout)
            return EXIT_PREFLIGHT_BLOCKED
        if is_c7 or is_c7_diagnostic:
            _configure_c7_execution(args.device)
        if authorization is not None:
            write_c7_launch_authorization(args.output_root, authorization)
        factory = backend_factory or _default_backend_factory
        backend = _make_backend(
            factory, config=config, registry=registry, plan=plan, args=args
        )
        budget = getattr(config, "budget", None)
        caps = budget.to_dict() if callable(getattr(budget, "to_dict", None)) else None
        common = {
            "output_root": args.output_root,
            "backend": backend,
            "device": args.device,
            "resume": False if args.command == "benchmark" else not args.no_resume,
            "estimate": estimate,
            "caps": caps,
        }
        if args.command == "run-sample":
            result = run_stage_c_sample(plan, args.sample_id, **common)
        else:
            result = run_stage_c_batch(plan, **common)
        _write_json(result, sys.stdout)
        if result.successful:
            return EXIT_SUCCESS
        if result.orchestration_complete:
            return EXIT_TERMINAL_FAILURE
        return EXIT_INTEGRITY_FAILURE
    except StageCBatchValidationError as error:
        _write_json(
            {
                "error": {
                    "message": str(error),
                    "type": type(error).__name__,
                }
            },
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except StageCBenchmarkIntegrityError as error:
        _write_json(
            {
                "error": {
                    "message": str(error),
                    "type": type(error).__name__,
                }
            },
            sys.stderr,
        )
        return EXIT_INTEGRITY_FAILURE
    except StageCBenchmarkBlockedError as error:
        report = getattr(error, "report", None)
        if isinstance(report, dict):
            _write_json(report, sys.stdout)
        else:
            reasons = list(getattr(error, "reasons", ()) or (str(error),))
            _write_json(
                {
                    "document_type": "stage_c_c7_launch_gate",
                    "status": "BLOCKED",
                    "blocked_reasons": reasons,
                },
                sys.stdout,
            )
        return EXIT_PREFLIGHT_BLOCKED
    except ArgumentError as error:
        _write_json(
            {"error": {"message": str(error), "type": "ArgumentError"}},
            sys.stderr,
        )
        return EXIT_INPUT_ERROR
    except Exception as error:
        _write_json(
            {
                "error": {
                    "message": str(error),
                    "type": type(error).__name__,
                }
            },
            sys.stderr,
        )
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_INPUT_ERROR", "EXIT_INTEGRITY_FAILURE",
    "EXIT_PREFLIGHT_BLOCKED", "EXIT_SUCCESS",
    "EXIT_TERMINAL_FAILURE", "ArgumentError", "build_parser", "main",
]
