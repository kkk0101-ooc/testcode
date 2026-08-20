from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import DEFAULT_SCANNER_ROOT, DEFAULT_STATE_DICT, REPO_ROOT
from .dataset import build_manifest
from .stage_b_runner import estimate_stage_b_run, load_stage_b_config, run_stage_b_sample


DEFAULT_CONFIG = REPO_ROOT / "config" / "stage_b.json"
DEFAULT_RELATIVE_IMAGE = "n03976467/n03976467_9546.JPEG"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "stage_b"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ViT-NNS Stage B: conditional fixed-attention IG for a block-internal "
            "pre-projection O=A@V scalar"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser(
        "estimate", help="validate the strict JSON config and estimate retained output bytes"
    )
    estimate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    attribute = subparsers.add_parser(
        "attribute", help="run one manifest sample through the Stage B gates"
    )
    attribute.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    attribute.add_argument("--data-root", type=Path, default=DEFAULT_SCANNER_ROOT)
    attribute.add_argument("--relative-image", default=DEFAULT_RELATIVE_IMAGE)
    attribute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    attribute.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT)
    attribute.add_argument("--device", default="cpu")
    attribute.add_argument("--seed", type=int, default=20260820)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_stage_b_config(args.config)
    estimate = estimate_stage_b_run(config)
    if args.command == "estimate":
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0 if estimate["within_budget"] else 2

    if not estimate["within_budget"]:
        raise SystemExit(
            "Stage B budget preflight failed: "
            f"estimated={estimate['estimated_bytes']} budget={estimate['budget_bytes']}"
        )
    rows = build_manifest(args.data_root, require_labels=True)
    by_path = {row.relative_path: row for row in rows}
    if args.relative_image not in by_path:
        raise SystemExit(f"Image is not in manifest: {args.relative_image}")
    provenance = run_stage_b_sample(
        repo_root=REPO_ROOT,
        image_path=args.data_root / args.relative_image,
        manifest_row=by_path[args.relative_image],
        output_dir=args.output_dir,
        config_path=args.config,
        device=args.device,
        state_dict_path=args.state_dict,
        seed=args.seed,
    )
    print("Stage B attribution: COMPLETE")
    print(json.dumps({
        "sample_id": provenance["input"]["sample_id"],
        "method": provenance["method"]["name"],
        "blocks": provenance["configuration"]["blocks"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
