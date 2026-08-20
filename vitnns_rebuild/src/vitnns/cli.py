from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_SCANNER_ROOT, DEFAULT_STATE_DICT, DEFAULT_VAL_ROOT, REPO_ROOT
from .dataset import build_manifest, manifest_metadata, write_manifest_csv
from .io_utils import sha256_file, write_json
from .provenance import git_state
from .runner import run_stage_a_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ViT-NNS Stage A baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="build a deterministic dataset manifest")
    manifest.add_argument("--data-root", type=Path, required=True)
    manifest.add_argument("--output-dir", type=Path, required=True)
    manifest.add_argument("--devkit", type=Path)
    manifest.add_argument("--allow-unlabeled", action="store_true")

    verify = subparsers.add_parser("verify", help="run one-sample Stage A parity verification")
    verify.add_argument("--data-root", type=Path, default=DEFAULT_SCANNER_ROOT)
    verify.add_argument("--relative-image", default="n03976467/n03976467_9546.JPEG")
    verify.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "stage_a")
    verify.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT)
    verify.add_argument("--device", default="cpu")
    verify.add_argument("--compare-legacy-pickle", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        rows = build_manifest(
            args.data_root, devkit_path=args.devkit, require_labels=not args.allow_unlabeled
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_manifest_csv(args.output_dir / "manifest.csv", rows)
        metadata = manifest_metadata(args.data_root, rows, args.devkit)
        lock_path = REPO_ROOT / "requirements.lock.txt"
        metadata["code"] = git_state(REPO_ROOT)
        metadata["dependencies"] = {
            "lock_path": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
        }
        write_json(args.output_dir / "manifest_metadata.json", metadata)
        print(f"wrote {len(rows)} samples to {args.output_dir}")
        return 0

    rows = build_manifest(args.data_root, require_labels=True)
    by_path = {row.relative_path: row for row in rows}
    if args.relative_image not in by_path:
        raise SystemExit(f"Image is not in manifest: {args.relative_image}")
    provenance = run_stage_a_sample(
        repo_root=REPO_ROOT,
        image_path=args.data_root / args.relative_image,
        manifest_row=by_path[args.relative_image],
        output_dir=args.output_dir,
        device=args.device,
        state_dict_path=args.state_dict,
        compare_legacy_pickle=args.compare_legacy_pickle,
    )
    print("Stage A verification: PASS")
    print(provenance["validation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

