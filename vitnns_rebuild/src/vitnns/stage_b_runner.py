from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .constants import DEFAULT_STATE_DICT
from .dataset import ManifestRow
from .extractor import NativeViTExtractor
from .io_utils import sha256_file
from .modeling import (
    PreprocessSpec,
    canonical_preprocess_spec,
    canonical_transform,
    load_canonical_model,
)
from .provenance import git_state
from .runner import seed_everything


STAGE_B_SCHEMA_VERSION = 1
STAGE_B_METHOD = "conditional_fixed_attention_value_path_ig"
SUPPORTED_CLASS_TARGETS = frozenset({"predicted", "ground_truth", "index"})
SUPPORTED_BASELINES = frozenset({"zero_normalized", "black_rgb"})
SUPPORTED_PROFILES = frozenset({"debug", "paper", "audit"})
ALL_BLOCKS = tuple(range(12))
NATIVE_ATTRIBUTION_SHAPE = (3, 224, 224)
ENDPOINT_PARITY_THRESHOLD = 1e-6


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")


def _require_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a boolean")
    return value


def _require_int(
    value: Any, context: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{context} must be <= {maximum}")
    return value


def _require_float(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{context} must be > 0")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


@dataclass(frozen=True)
class ClassTargetConfig:
    """Class-logit context used for prediction reporting, not Value-path conservation."""

    mode: str
    index: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClassTargetConfig":
        value = _require_mapping(value, "class_target")
        _require_keys(value, required={"mode"}, optional={"index"}, context="class_target")
        mode = value["mode"]
        if mode not in SUPPORTED_CLASS_TARGETS:
            raise ValueError(
                f"class_target.mode must be one of {sorted(SUPPORTED_CLASS_TARGETS)}"
            )
        if mode == "index":
            if "index" not in value:
                raise ValueError("class_target.index is required when mode is 'index'")
            index = _require_int(value["index"], "class_target.index", minimum=0, maximum=999)
        else:
            if "index" in value:
                raise ValueError(
                    f"class_target.index is not allowed when mode is {mode!r}"
                )
            index = None
        return cls(mode=mode, index=index)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode}
        if self.index is not None:
            result["index"] = self.index
        return result


@dataclass(frozen=True)
class ValueTargetConfig:
    """One scalar in per-head pre-projection O_h = A_h @ V_h."""

    head: int
    query_token: int
    head_channel: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValueTargetConfig":
        value = _require_mapping(value, "value_target")
        _require_keys(
            value,
            required={"head", "query_token", "head_channel"},
            context="value_target",
        )
        return cls(
            head=_require_int(value["head"], "value_target.head", minimum=0, maximum=11),
            query_token=_require_int(
                value["query_token"], "value_target.query_token", minimum=0, maximum=196
            ),
            head_channel=_require_int(
                value["head_channel"], "value_target.head_channel", minimum=0, maximum=63
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class IGConfig:
    quadrature: str
    steps_schedule: tuple[int, ...]
    step_chunk_size: int
    atol: float
    rtol: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IGConfig":
        value = _require_mapping(value, "ig")
        _require_keys(
            value,
            required={"quadrature", "steps_schedule", "step_chunk_size", "atol", "rtol"},
            context="ig",
        )
        if value["quadrature"] != "gauss_legendre":
            raise ValueError("ig.quadrature must be 'gauss_legendre'")
        raw_schedule = value["steps_schedule"]
        if not isinstance(raw_schedule, list) or not raw_schedule:
            raise ValueError("ig.steps_schedule must be a non-empty JSON array")
        schedule = tuple(
            _require_int(step, f"ig.steps_schedule[{index}]", minimum=1)
            for index, step in enumerate(raw_schedule)
        )
        if any(left >= right for left, right in zip(schedule, schedule[1:])):
            raise ValueError("ig.steps_schedule must be strictly increasing")
        chunk = _require_int(value["step_chunk_size"], "ig.step_chunk_size", minimum=1)
        if chunk > schedule[-1]:
            raise ValueError("ig.step_chunk_size cannot exceed the largest scheduled step count")
        atol = _require_float(value["atol"], "ig.atol", minimum=0.0)
        rtol = _require_float(value["rtol"], "ig.rtol", minimum=0.0)
        if atol == 0.0 and rtol == 0.0:
            raise ValueError("ig.atol and ig.rtol cannot both be zero")
        return cls(
            quadrature="gauss_legendre",
            steps_schedule=schedule,
            step_chunk_size=chunk,
            atol=atol,
            rtol=rtol,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quadrature": self.quadrature,
            "steps_schedule": list(self.steps_schedule),
            "step_chunk_size": self.step_chunk_size,
            "atol": self.atol,
            "rtol": self.rtol,
        }


@dataclass(frozen=True)
class RepeatabilityConfig:
    enabled: bool
    atol: float
    rtol: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepeatabilityConfig":
        value = _require_mapping(value, "repeatability")
        _require_keys(value, required={"enabled", "atol", "rtol"}, context="repeatability")
        return cls(
            enabled=_require_bool(value["enabled"], "repeatability.enabled"),
            atol=_require_float(value["atol"], "repeatability.atol", minimum=0.0),
            rtol=_require_float(value["rtol"], "repeatability.rtol", minimum=0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SanityConfig:
    enabled: bool
    perturbation: str
    seed: int
    noise_std: float
    clamp_to_valid_range: bool
    min_relative_l1_change: float
    epsilon: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SanityConfig":
        value = _require_mapping(value, "sanity")
        _require_keys(
            value,
            required={
                "enabled",
                "perturbation",
                "seed",
                "noise_std",
                "clamp_to_valid_range",
                "min_relative_l1_change",
                "epsilon",
            },
            context="sanity",
        )
        if value["perturbation"] != "seeded_gaussian_noise":
            raise ValueError("sanity.perturbation must be 'seeded_gaussian_noise'")
        enabled = _require_bool(value["enabled"], "sanity.enabled")
        noise_std = _require_float(
            value["noise_std"],
            "sanity.noise_std",
            minimum=0.0 if not enabled else None,
            strictly_positive=enabled,
        )
        threshold = _require_float(
            value["min_relative_l1_change"],
            "sanity.min_relative_l1_change",
            minimum=0.0,
        )
        if enabled and threshold <= 0.0:
            raise ValueError("sanity.min_relative_l1_change must be > 0 when enabled")
        return cls(
            enabled=enabled,
            perturbation="seeded_gaussian_noise",
            seed=_require_int(value["seed"], "sanity.seed", minimum=0),
            noise_std=noise_std,
            clamp_to_valid_range=_require_bool(
                value["clamp_to_valid_range"], "sanity.clamp_to_valid_range"
            ),
            min_relative_l1_change=threshold,
            epsilon=_require_float(value["epsilon"], "sanity.epsilon", strictly_positive=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ComputeBudgetConfig:
    """Declared caps for batched model invocations in one sample run."""

    max_forward_calls: int
    max_backward_calls: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComputeBudgetConfig":
        value = _require_mapping(value, "compute_budget")
        _require_keys(
            value,
            required={"max_forward_calls", "max_backward_calls"},
            context="compute_budget",
        )
        return cls(
            max_forward_calls=_require_int(
                value["max_forward_calls"], "compute_budget.max_forward_calls", minimum=1
            ),
            max_backward_calls=_require_int(
                value["max_backward_calls"], "compute_budget.max_backward_calls", minimum=1
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class StageBConfig:
    schema_version: int
    method: str
    class_target: ClassTargetConfig
    value_target: ValueTargetConfig
    baseline: str
    blocks: tuple[int, ...]
    ig: IGConfig
    profile: str
    budget_bytes: int
    repeatability: RepeatabilityConfig
    sanity: SanityConfig
    compute_budget: ComputeBudgetConfig
    formal: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageBConfig":
        value = _require_mapping(value, "Stage B config")
        required = {
            "schema_version",
            "method",
            "class_target",
            "value_target",
            "baseline",
            "blocks",
            "ig",
            "profile",
            "budget_bytes",
            "repeatability",
            "sanity",
            "compute_budget",
            "formal",
        }
        _require_keys(value, required=required, context="Stage B config")
        schema = _require_int(value["schema_version"], "schema_version")
        if schema != STAGE_B_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {STAGE_B_SCHEMA_VERSION}, got {schema}"
            )
        if value["method"] != STAGE_B_METHOD:
            raise ValueError(
                f"method must be {STAGE_B_METHOD!r}; class-logit fixed-A is not a pure Value path"
            )
        if value["baseline"] not in SUPPORTED_BASELINES:
            raise ValueError(f"baseline must be one of {sorted(SUPPORTED_BASELINES)}")
        profile = value["profile"]
        if profile not in SUPPORTED_PROFILES:
            raise ValueError(f"profile must be one of {sorted(SUPPORTED_PROFILES)}")
        raw_blocks = value["blocks"]
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise ValueError("blocks must be a non-empty JSON array")
        blocks = tuple(
            _require_int(block, f"blocks[{index}]", minimum=0, maximum=11)
            for index, block in enumerate(raw_blocks)
        )
        if any(left >= right for left, right in zip(blocks, blocks[1:])):
            raise ValueError("blocks must be unique and strictly increasing")
        if profile == "audit" and blocks != ALL_BLOCKS:
            raise ValueError("audit profile requires every block in order: 0..11")
        repeatability = RepeatabilityConfig.from_dict(value["repeatability"])
        sanity = SanityConfig.from_dict(value["sanity"])
        formal = _require_bool(value["formal"], "formal")
        if formal and (not repeatability.enabled or not sanity.enabled):
            raise ValueError(
                "formal Stage B runs require repeatability and sanity gates to be enabled"
            )
        return cls(
            schema_version=schema,
            method=STAGE_B_METHOD,
            class_target=ClassTargetConfig.from_dict(value["class_target"]),
            value_target=ValueTargetConfig.from_dict(value["value_target"]),
            baseline=value["baseline"],
            blocks=blocks,
            ig=IGConfig.from_dict(value["ig"]),
            profile=profile,
            budget_bytes=_require_int(value["budget_bytes"], "budget_bytes", minimum=1),
            repeatability=repeatability,
            sanity=sanity,
            compute_budget=ComputeBudgetConfig.from_dict(value["compute_budget"]),
            formal=formal,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "class_target": self.class_target.to_dict(),
            "value_target": self.value_target.to_dict(),
            "baseline": self.baseline,
            "blocks": list(self.blocks),
            "ig": self.ig.to_dict(),
            "profile": self.profile,
            "budget_bytes": self.budget_bytes,
            "repeatability": self.repeatability.to_dict(),
            "sanity": self.sanity.to_dict(),
            "compute_budget": self.compute_budget.to_dict(),
            "formal": self.formal,
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_stage_b_config(path: Path) -> StageBConfig:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage B config not found: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number is not allowed: {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Stage B JSON config: {error}") from error
    return StageBConfig.from_dict(value)


def resolve_class_target(
    target: ClassTargetConfig, logits: torch.Tensor, manifest_row: ManifestRow
) -> int:
    """Resolve the reporting class without conflating it with the internal Value target."""
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"Expected one-sample logits [1,classes], got {tuple(logits.shape)}")
    class_count = int(logits.shape[1])
    if target.mode == "predicted":
        return int(logits[0].argmax().item())
    if target.mode == "ground_truth":
        if manifest_row.imagenet1k_index is None:
            raise ValueError("ground_truth class target requires a labeled ManifestRow")
        result = int(manifest_row.imagenet1k_index)
    elif target.mode == "index":
        if target.index is None:
            raise ValueError("index class target has no index")
        result = target.index
    else:
        raise ValueError(f"Unsupported class target mode: {target.mode!r}")
    if not 0 <= result < class_count:
        raise ValueError(f"Resolved class index {result} is outside [0,{class_count - 1}]")
    return result


def build_baseline(
    inputs: torch.Tensor, baseline: str, preprocess: PreprocessSpec
) -> torch.Tensor:
    if inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError(f"Expected normalized RGB inputs [B,3,H,W], got {tuple(inputs.shape)}")
    if baseline == "zero_normalized":
        return torch.zeros_like(inputs)
    if baseline == "black_rgb":
        mean = torch.as_tensor(preprocess.mean, dtype=inputs.dtype, device=inputs.device)
        std = torch.as_tensor(preprocess.std, dtype=inputs.dtype, device=inputs.device)
        if torch.any(std <= 0):
            raise ValueError("Preprocessing standard deviations must be positive")
        black = (-mean / std).reshape(1, 3, 1, 1)
        return black.expand_as(inputs).clone()
    raise ValueError(f"Unsupported baseline: {baseline!r}")


def deterministic_input_perturbation(
    inputs: torch.Tensor,
    sanity: SanityConfig,
    preprocess: PreprocessSpec,
) -> torch.Tensor:
    if inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError(f"Expected normalized RGB inputs [B,3,H,W], got {tuple(inputs.shape)}")
    if sanity.perturbation != "seeded_gaussian_noise":
        raise ValueError(f"Unsupported perturbation: {sanity.perturbation!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sanity.seed)
    noise = torch.randn(
        tuple(inputs.shape), generator=generator, device="cpu", dtype=torch.float32
    ).to(device=inputs.device, dtype=inputs.dtype)
    perturbed = inputs + noise * sanity.noise_std
    if sanity.clamp_to_valid_range:
        mean = torch.as_tensor(preprocess.mean, device=inputs.device, dtype=inputs.dtype)
        std = torch.as_tensor(preprocess.std, device=inputs.device, dtype=inputs.dtype)
        low = ((torch.zeros_like(mean) - mean) / std).reshape(1, 3, 1, 1)
        high = ((torch.ones_like(mean) - mean) / std).reshape(1, 3, 1, 1)
        perturbed = torch.maximum(torch.minimum(perturbed, high), low)
    return perturbed


def map_change_metrics(
    reference: torch.Tensor, changed: torch.Tensor, *, epsilon: float
) -> dict[str, float]:
    if reference.shape != changed.shape:
        raise ValueError(
            f"Attribution map shapes differ: {tuple(reference.shape)} vs {tuple(changed.shape)}"
        )
    ref = reference.detach().to(dtype=torch.float64, device="cpu")
    other = changed.detach().to(dtype=torch.float64, device="cpu")
    difference = other - ref
    reference_l1 = float(ref.abs().sum().item())
    difference_l1 = float(difference.abs().sum().item())
    return {
        "reference_l1": reference_l1,
        "difference_l1": difference_l1,
        "relative_l1_change": difference_l1 / max(reference_l1, epsilon),
        "max_abs_change": float(difference.abs().max().item()),
    }


def compare_attribution_maps(
    first: torch.Tensor, second: torch.Tensor, *, atol: float, rtol: float
) -> dict[str, Any]:
    if first.shape != second.shape:
        raise ValueError(f"Repeat maps have different shapes: {first.shape} vs {second.shape}")
    difference = (first.detach().cpu() - second.detach().cpu()).abs()
    return {
        "pass": bool(torch.allclose(first, second, atol=atol, rtol=rtol, equal_nan=False)),
        "max_abs_error": float(difference.max().item()),
        "atol": atol,
        "rtol": rtol,
    }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metric_float(value: Mapping[str, Any], key: str) -> float:
    if key not in value:
        raise ValueError(f"IG conservation is missing {key!r}")
    raw = value[key]
    if isinstance(raw, torch.Tensor):
        if raw.numel() != 1:
            raise ValueError(f"IG conservation {key} must contain exactly one value")
        raw = raw.detach().cpu().item()
    elif isinstance(raw, np.ndarray):
        if raw.size != 1:
            raise ValueError(f"IG conservation {key} must contain exactly one value")
        raw = raw.item()
    elif isinstance(raw, (tuple, list)):
        if len(raw) != 1:
            raise ValueError(f"IG conservation {key} must contain exactly one value")
        raw = raw[0]
    result = _require_float(raw, f"IG conservation {key}")
    return result


def native_signed_maps(result: Any, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    attribution = getattr(result, "attribution", None)
    if not isinstance(attribution, torch.Tensor):
        raise TypeError("IG result.attribution must be a torch.Tensor")
    if attribution.shape != inputs.shape:
        raise ValueError(
            f"IG attribution must have the native input shape {tuple(inputs.shape)}, "
            f"got {tuple(attribution.shape)}"
        )
    if attribution.shape[0] != 1 or attribution.shape[1] != 3:
        raise ValueError("Stage B currently requires one RGB sample")
    if not bool(torch.isfinite(attribution).all().item()):
        raise ValueError("IG attribution contains a non-finite value")
    signed_rgb = attribution[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
    rgb_sum = signed_rgb.sum(dim=0).contiguous()
    return signed_rgb, rgb_sum


def _summarize_ig_result(
    result: Any, signed_rgb: torch.Tensor, *, atol: float, rtol: float
) -> dict[str, Any]:
    raw = getattr(result, "conservation", None)
    if not isinstance(raw, Mapping):
        raise TypeError("IG result.conservation must be a mapping")
    delta = _metric_float(raw, "delta")
    reported_sum = _metric_float(raw, "sum")
    reported_abs = _metric_float(raw, "abs")
    reported_rel = _metric_float(raw, "rel")
    reported_pass_value = raw.get("pass")
    if isinstance(reported_pass_value, torch.Tensor):
        if reported_pass_value.numel() != 1:
            raise ValueError("IG conservation 'pass' must contain exactly one value")
        reported_pass = bool(reported_pass_value.detach().cpu().item())
    elif isinstance(reported_pass_value, np.ndarray):
        if reported_pass_value.size != 1:
            raise ValueError("IG conservation 'pass' must contain exactly one value")
        reported_pass = bool(reported_pass_value.item())
    elif isinstance(reported_pass_value, (tuple, list)):
        if len(reported_pass_value) != 1 or type(reported_pass_value[0]) is not bool:
            raise ValueError("IG conservation 'pass' must contain one boolean")
        reported_pass = reported_pass_value[0]
    elif type(reported_pass_value) is bool:
        reported_pass = reported_pass_value
    else:
        raise ValueError("IG conservation 'pass' must be a boolean")
    native_sum = float(signed_rgb.to(torch.float64).sum().item())
    native_abs = abs(native_sum - delta)
    threshold = atol + rtol * abs(delta)
    native_pass = native_abs <= threshold
    if not reported_pass or not native_pass:
        raise AssertionError(
            "Stage B conservation failed: "
            f"reported_pass={reported_pass}, native_abs={native_abs}, threshold={threshold}"
        )
    return {
        "delta": delta,
        "sum": reported_sum,
        "abs": reported_abs,
        "rel": reported_rel,
        "pass": reported_pass,
        "native_signed_rgb_sum": native_sum,
        "native_abs_error": native_abs,
        "gate_threshold": threshold,
        "native_pass": native_pass,
        "steps": _jsonable(getattr(result, "steps", None)),
        "converged": bool(getattr(result, "converged", reported_pass)),
        "attempts": _jsonable(getattr(result, "attempts", [])),
        "metadata": _jsonable(getattr(result, "metadata", {})),
    }


def _measure_endpoint_parity(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    direct_logits: torch.Tensor,
    block_index: int,
    fixed_attention: torch.Tensor,
    intervention_factory: Any,
) -> dict[str, Any]:
    with intervention_factory(model, block_index, fixed_attention) as intervention:
        with torch.inference_mode():
            intervened_logits = model(inputs).detach().clone()
    call_count = int(getattr(intervention, "call_count", 0))
    error = float((direct_logits - intervened_logits).abs().max().item())
    passed = call_count == 1 and error < ENDPOINT_PARITY_THRESHOLD
    result = {
        "max_abs_error": error,
        "threshold_exclusive": ENDPOINT_PARITY_THRESHOLD,
        "intervention_call_count": call_count,
        "pass": passed,
    }
    if not passed:
        raise AssertionError(f"Block {block_index} fixed-A endpoint parity failed: {result}")
    return result


def estimate_stage_b_run(config: StageBConfig) -> dict[str, Any]:
    try:
        from .retention import estimate_output_bytes
    except ImportError as error:  # pragma: no cover - only relevant during partial installations
        raise RuntimeError("Stage B retention module is unavailable") from error
    retained = int(
        estimate_output_bytes(
            layer_count=len(config.blocks),
            shape=NATIVE_ATTRIBUTION_SHAPE[1:],
            profile=config.profile,
            head_count=1,
            target_count=1,
        )
    )
    if retained < 0:
        raise RuntimeError("Retention estimator returned a negative byte count")
    # Each adaptive IG attempt starts afresh: two endpoint forwards plus one
    # batched forward/backward invocation per step chunk.  Count the complete
    # declared schedule so preflight remains a conservative upper bound.
    backward_calls_per_attribution = sum(
        math.ceil(steps / config.ig.step_chunk_size)
        for steps in config.ig.steps_schedule
    )
    # integrated_gradients evaluates the baseline/input endpoints once, then
    # restarts only the quadrature accumulation for every scheduled step count.
    worst_case_ig_steps = sum(config.ig.steps_schedule)
    forward_calls_per_attribution = backward_calls_per_attribution + 2
    attribution_runs_per_block = (
        1 + int(config.repeatability.enabled) + int(config.sanity.enabled)
    )
    base_forward_calls = 1 + int(config.sanity.enabled)
    parity_calls_per_block = 1 + int(config.sanity.enabled)
    estimated_forward_calls = base_forward_calls + len(config.blocks) * (
        parity_calls_per_block
        + attribution_runs_per_block * forward_calls_per_attribution
    )
    estimated_backward_calls = (
        len(config.blocks)
        * attribution_runs_per_block
        * backward_calls_per_attribution
    )
    storage_within_budget = retained <= config.budget_bytes
    compute_within_budget = (
        estimated_forward_calls <= config.compute_budget.max_forward_calls
        and estimated_backward_calls <= config.compute_budget.max_backward_calls
    )
    return {
        "profile": config.profile,
        "blocks": list(config.blocks),
        "native_signed_rgb_shape": list(NATIVE_ATTRIBUTION_SHAPE),
        "retained_metric_shape": list(NATIVE_ATTRIBUTION_SHAPE[1:]),
        "retention_estimate_bytes": retained,
        "estimated_bytes": retained,
        "budget_bytes": config.budget_bytes,
        "storage_within_budget": storage_within_budget,
        "compute": {
            "accounting_unit": "batched_model_invocation",
            "schedule_restarts_for_each_step_count": True,
            "worst_case_ig_steps_per_attribution": worst_case_ig_steps,
            "forward_calls_per_attribution": forward_calls_per_attribution,
            "backward_calls_per_attribution": backward_calls_per_attribution,
            "attribution_runs_per_block": attribution_runs_per_block,
            "estimated_forward_calls": estimated_forward_calls,
            "estimated_backward_calls": estimated_backward_calls,
            "max_forward_calls": config.compute_budget.max_forward_calls,
            "max_backward_calls": config.compute_budget.max_backward_calls,
            "within_budget": compute_within_budget,
        },
        "within_budget": storage_within_budget and compute_within_budget,
    }


def _writer_actual_bytes(writer: Any) -> int:
    budget = getattr(writer, "budget", None)
    for owner in (budget, writer):
        if owner is None:
            continue
        for name in ("actual_bytes", "used_bytes", "bytes_written", "consumed_bytes"):
            value = getattr(owner, name, None)
            if isinstance(value, int) and value >= 0:
                return value
    manifest = getattr(writer, "manifest", None)
    if isinstance(manifest, Mapping):
        for name in ("actual_bytes", "total_bytes", "bytes_written"):
            value = manifest.get(name)
            if isinstance(value, int) and value >= 0:
                return value
        records = manifest.get("artifacts", manifest.get("entries", []))
        if isinstance(records, Sequence):
            return sum(
                int(record.get("size_bytes", 0))
                for record in records
                if isinstance(record, Mapping)
            )
    return 0


def _run_value_path_ig(
    function: Any,
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    block_index: int,
    fixed_attention: torch.Tensor,
    config: StageBConfig,
) -> Any:
    target = config.value_target
    return function(
        model,
        inputs,
        baselines,
        block_index=block_index,
        head=target.head,
        query_token=target.query_token,
        head_channel=target.head_channel,
        fixed_attention=fixed_attention,
        steps=config.ig.steps_schedule,
        step_chunk_size=config.ig.step_chunk_size,
        atol=config.ig.atol,
        rtol=config.ig.rtol,
    )


def run_stage_b_sample(
    *,
    repo_root: Path,
    image_path: Path,
    manifest_row: ManifestRow,
    output_dir: Path,
    config_path: Path,
    device: str = "cpu",
    state_dict_path: Path = DEFAULT_STATE_DICT,
    lock_path: Path | None = None,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Run one auditable Stage B sample through the real unfused timm model.

    The configured method explains one scalar in O_h=A_h@V_h for each block.
    The recorded class target is prediction context only and is deliberately not
    presented as the Value-path target.
    """
    try:
        from .attribution import (
            conditional_fixed_attention_value_path_ig,
            fixed_attention_intervention,
        )
        from .retention import (
            ArtifactArray,
            ArtifactKind,
            ArtifactWriter,
            BudgetTracker,
            audit_replay_report,
            get_retention_policy,
            paper_float16_roundtrip_error,
        )
        from .stage_b_provenance import (
            build_stage_b_provenance,
            tensor_sha256,
            validate_stage_b_provenance,
        )
    except ImportError as error:  # pragma: no cover - only relevant during partial installations
        raise RuntimeError(
            "Stage B requires vitnns.attribution, vitnns.retention, and "
            "vitnns.stage_b_provenance"
        ) from error

    repo_root = Path(repo_root).resolve()
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    state_dict_path = Path(state_dict_path).resolve()
    lock_path = Path(lock_path or repo_root / "requirements.lock.txt").resolve()
    config = load_stage_b_config(config_path)

    estimate = estimate_stage_b_run(config)
    if not estimate["within_budget"]:
        raise RuntimeError(
            "Stage B retention budget is insufficient before execution: "
            f"estimated={estimate['estimated_bytes']}, budget={config.budget_bytes}"
        )
    code_state = git_state(repo_root)
    if config.formal and code_state["dirty"]:
        raise RuntimeError("Formal Stage B runs require a clean Git tree")
    if sha256_file(image_path) != manifest_row.image_sha256:
        raise ValueError("Image SHA-256 no longer matches its ManifestRow")

    seed_everything(seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "Deterministic CUDA attribution requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before process start"
            )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = load_canonical_model(
        state_dict_path, device=torch_device, verify_hash=True, unfused_attention=True
    )
    if any(block.attn.fused_attn for block in model.blocks):
        raise RuntimeError("Stage B requires the real unfused attention path")
    preprocess = canonical_preprocess_spec(model)
    transform = canonical_transform(model)
    with Image.open(image_path) as source:
        direct_input = transform(source.convert("RGB")).unsqueeze(0).to(torch_device)
    if tuple(direct_input.shape[1:]) != NATIVE_ATTRIBUTION_SHAPE:
        raise RuntimeError(
            f"Unexpected canonical input shape: {tuple(direct_input.shape[1:])}"
        )
    direct_input_hash = tensor_sha256(direct_input)
    baseline = build_baseline(direct_input, config.baseline, preprocess)

    # This exact tensor object is used by both the direct and attribution paths.
    direct = NativeViTExtractor(model).extract(direct_input)
    class_target_index = resolve_class_target(config.class_target, direct.logits, manifest_row)

    perturbed_input: torch.Tensor | None = None
    perturbed_direct: Any = None
    if config.sanity.enabled:
        perturbed_input = deterministic_input_perturbation(direct_input, config.sanity, preprocess)
        perturbed_direct = NativeViTExtractor(model).extract(perturbed_input)

    policy = get_retention_policy(config.profile)
    budget = BudgetTracker(max_bytes=config.budget_bytes)

    layer_records: list[dict[str, Any]] = []
    signed_maps: dict[int, torch.Tensor] = {}
    summed_maps: dict[int, torch.Tensor] = {}
    quantization: dict[str, Any] = {}
    audit_replay: dict[str, Any] = {}

    for block_index in config.blocks:
        fixed_attention = direct.layers[block_index].attention_used
        parity = _measure_endpoint_parity(
            model=model,
            inputs=direct_input,
            direct_logits=direct.logits,
            block_index=block_index,
            fixed_attention=fixed_attention,
            intervention_factory=fixed_attention_intervention,
        )
        result = _run_value_path_ig(
            conditional_fixed_attention_value_path_ig,
            model=model,
            inputs=direct_input,
            baselines=baseline,
            block_index=block_index,
            fixed_attention=fixed_attention,
            config=config,
        )
        signed_rgb, rgb_sum = native_signed_maps(result, direct_input)
        conservation = _summarize_ig_result(
            result, signed_rgb, atol=config.ig.atol, rtol=config.ig.rtol
        )
        signed_maps[block_index] = signed_rgb
        summed_maps[block_index] = rgb_sum

        repeatability: dict[str, Any] = {"enabled": config.repeatability.enabled}
        if config.repeatability.enabled:
            repeated = _run_value_path_ig(
                conditional_fixed_attention_value_path_ig,
                model=model,
                inputs=direct_input,
                baselines=baseline,
                block_index=block_index,
                fixed_attention=fixed_attention,
                config=config,
            )
            repeated_rgb, _ = native_signed_maps(repeated, direct_input)
            repeatability.update(
                compare_attribution_maps(
                    signed_rgb,
                    repeated_rgb,
                    atol=config.repeatability.atol,
                    rtol=config.repeatability.rtol,
                )
            )
            if not repeatability["pass"]:
                raise AssertionError(
                    f"Block {block_index} repeatability gate failed: {repeatability}"
                )

        sanity_record: dict[str, Any] = {"enabled": config.sanity.enabled}
        if config.sanity.enabled:
            assert perturbed_input is not None and perturbed_direct is not None
            perturbed_attention = perturbed_direct.layers[block_index].attention
            sanity_parity = _measure_endpoint_parity(
                model=model,
                inputs=perturbed_input,
                direct_logits=perturbed_direct.logits,
                block_index=block_index,
                fixed_attention=perturbed_attention,
                intervention_factory=fixed_attention_intervention,
            )
            sanity_result = _run_value_path_ig(
                conditional_fixed_attention_value_path_ig,
                model=model,
                inputs=perturbed_input,
                baselines=build_baseline(perturbed_input, config.baseline, preprocess),
                block_index=block_index,
                fixed_attention=perturbed_attention,
                config=config,
            )
            sanity_rgb, _ = native_signed_maps(sanity_result, perturbed_input)
            sanity_conservation = _summarize_ig_result(
                sanity_result, sanity_rgb, atol=config.ig.atol, rtol=config.ig.rtol
            )
            change = map_change_metrics(
                signed_rgb, sanity_rgb, epsilon=config.sanity.epsilon
            )
            change_pass = (
                change["relative_l1_change"]
                >= config.sanity.min_relative_l1_change
            )
            sanity_record.update(
                {
                    "perturbed_input_sha256": tensor_sha256(perturbed_input),
                    "endpoint_parity": sanity_parity,
                    "conservation": sanity_conservation,
                    "map_change": change,
                    "minimum_relative_l1_change": config.sanity.min_relative_l1_change,
                    "pass": change_pass,
                }
            )
            if not change_pass:
                raise AssertionError(
                    f"Block {block_index} input-perturbation sanity gate failed: {sanity_record}"
                )

        layer_records.append(
            {
                "block_index": block_index,
                "fixed_attention_shape": list(fixed_attention.shape),
                "fixed_attention_sha256": tensor_sha256(fixed_attention),
                "direct_attention_recomputation_max_abs_error": (
                    direct.layers[block_index].attention_max_abs_error
                ),
                "endpoint_parity": parity,
                "conservation": conservation,
                "repeatability": repeatability,
                "sanity": sanity_record,
                "signed_rgb_shape": list(signed_rgb.shape),
                "rgb_sum_shape": list(rgb_sum.shape),
                "native_resolution": True,
                "resize_applied": False,
            }
        )

    metric_arrays: dict[str, np.ndarray] = {}
    raw_arrays: dict[str, np.ndarray] = {}
    for block_index in config.blocks:
        signed = signed_maps[block_index].numpy()
        summed = summed_maps[block_index].numpy()
        if config.profile == "paper":
            metric_arrays[f"block_{block_index:02d}_rgb_sum"] = summed.astype(np.float16)
            quantization[str(block_index)] = _jsonable(
                paper_float16_roundtrip_error(
                    ArtifactArray(summed, ArtifactKind.METRIC_READY)
                )
            )
        else:
            metric_arrays[f"block_{block_index:02d}_rgb_sum"] = summed
        if config.profile in {"audit", "debug"}:
            raw_arrays[f"block_{block_index:02d}_signed_rgb_float32"] = signed
        if config.profile == "audit":
            replay = audit_replay_report(
                ArtifactArray(signed, ArtifactKind.RAW),
                ArtifactArray(summed, ArtifactKind.METRIC_READY),
            )
            if not replay["pass"]:
                raise AssertionError(f"Block {block_index} audit replay failed: {replay}")
            audit_replay[str(block_index)] = replay

    # Claim the output directory only after all numerical gates have passed,
    # so an attribution/conservation failure cannot leave a result-looking
    # directory behind. From this point every write is guarded and atomic.
    writer = ArtifactWriter(output_dir, policy, budget=budget)
    writer.write_npz(
        "metric_ready", "maps.npz", metric_arrays, compressed=True, granularity="aggregate"
    )
    if raw_arrays:
        writer.write_npz(
            "raw", "maps_float32.npz", raw_arrays, compressed=True, granularity="aggregate"
        )

    metrics = {
        "schema_version": 1,
        "method": config.method,
        "class_context_target_index": class_target_index,
        "value_target": config.value_target.to_dict(),
        "baseline": config.baseline,
        "native_resolution": {
            "signed_rgb_shape_per_block": list(NATIVE_ATTRIBUTION_SHAPE),
            "rgb_sum_shape_per_block": list(NATIVE_ATTRIBUTION_SHAPE[1:]),
            "resize_applied": False,
        },
        "layers": layer_records,
        "paper_float16_roundtrip_error": quantization,
        "audit_replay": audit_replay,
    }
    writer.write_json("metric_ready", "metrics.json", metrics, granularity="aggregate")

    actual_before_provenance = _writer_actual_bytes(writer)
    retention_record = dict(estimate)
    retention_record.update(
        {
            "actual_bytes": actual_before_provenance,
            "actual_bytes_scope": (
                "all retained artifact payloads including provenance; writer manifest/status "
                "control files are excluded"
            ),
        }
    )
    provenance = build_stage_b_provenance(
        repo_root=repo_root,
        state_dict_path=state_dict_path,
        image_path=image_path,
        config_path=config_path,
        lock_path=lock_path,
        manifest_row=manifest_row,
        preprocess=preprocess,
        direct_logits=direct.logits,
        direct_input=direct_input,
        baseline=baseline,
        direct_input_sha256=direct_input_hash,
        device=torch_device,
        seed=seed,
        config=config.to_dict(),
        resolved_class_index=class_target_index,
        layer_records=layer_records,
        retention=retention_record,
        code_state=code_state,
    )
    # ArtifactWriter serializes JSON with this exact canonical representation.
    # Solve the tiny self-reference so provenance records the final payload-byte
    # total including provenance itself, not merely the pre-provenance subtotal.
    projected_actual = actual_before_provenance
    for _ in range(8):
        provenance["retention"]["actual_bytes"] = projected_actual
        payload_bytes = len(
            (json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            )
        )
        updated = actual_before_provenance + payload_bytes
        if updated == projected_actual:
            break
        projected_actual = updated
    else:
        raise RuntimeError("Could not stabilize Stage B actual-byte provenance")
    validate_stage_b_provenance(provenance)
    writer.write_json(
        "metric_ready", "provenance.json", provenance, granularity="aggregate"
    )
    if _writer_actual_bytes(writer) != projected_actual:
        raise RuntimeError(
            "Stage B actual artifact bytes differ from the value recorded in provenance"
        )
    finalization = writer.finalize()
    if finalization.get("status") != "complete":
        raise RuntimeError(f"Stage B artifact finalization did not complete: {finalization}")
    if writer.actual_bytes > config.budget_bytes:
        raise RuntimeError(
            "Stage B output tree exceeded its declared byte budget after finalization"
        )
    if writer.actual_bytes > estimate["estimated_bytes"]:
        raise RuntimeError(
            "Stage B output tree exceeded its conservative preflight estimate"
        )
    return provenance
