"""Non-interactive JSON CLI for Stage C planning and orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

from .constants import DEFAULT_STATE_DICT, REPO_ROOT
from .io_utils import sha256_file
from .stage_c_config import load_stage_c_config
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
        if name == "run-sample":
            command.add_argument("--sample-id", required=True)

    for name in ("validate", "aggregate"):
        command = commands.add_parser(name)
        _add_plan_arguments(command)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "aggregate":
            command.add_argument("--output", type=Path)
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

        if not estimate.within_budget:
            _write_json(estimate, sys.stdout)
            return EXIT_PREFLIGHT_BLOCKED
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
            "resume": not args.no_resume,
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
