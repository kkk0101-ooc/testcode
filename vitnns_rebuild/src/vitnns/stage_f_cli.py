"""Non-interactive JSON CLI for the Stage F F0 display-only surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .constants import REPO_ROOT
from .stage_f_renderer import (
    StageFRendererError, assert_f0_source_boundary, load_contracts,
    parse_upstream_recipes, render_fixture, validate_render_report,
    verify_preconditions,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vitnns-stage-f")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-preconditions", "validate-contracts", "parse-recipes", "validate-boundary"):
        child = sub.add_parser(name); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    render = sub.add_parser("render-golden"); render.add_argument("--repo-root", type=Path, default=REPO_ROOT); render.add_argument("--fixture", type=Path, required=True); render.add_argument("--output-dir", type=Path, required=True); render.add_argument("--renderer-commit")
    validate = sub.add_parser("validate-render"); validate.add_argument("--repo-root", type=Path, default=REPO_ROOT); validate.add_argument("--report", type=Path, required=True)
    initialize = sub.add_parser("initialize-f0"); initialize.add_argument("--repo-root", type=Path, default=REPO_ROOT); initialize.add_argument("--renderer-commit", required=True); initialize.add_argument("--started-at", required=True); initialize.add_argument("--startup-gate", type=Path, required=True)
    build = sub.add_parser("build-f0"); build.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    tests = sub.add_parser("record-tests"); tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); tests.add_argument("--passed", type=int, required=True); tests.add_argument("--failed", type=int, required=True); tests.add_argument("--test-command", required=True)
    finalize = sub.add_parser("finalize-f0"); finalize.add_argument("--repo-root", type=Path, default=REPO_ROOT); finalize.add_argument("--completed-at", required=True)
    completion = sub.add_parser("validate-completion"); completion.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f1_start = sub.add_parser("initialize-f1"); f1_start.add_argument("--repo-root", type=Path, default=REPO_ROOT); f1_start.add_argument("--renderer-commit", required=True); f1_start.add_argument("--started-at", required=True)
    f1_build = sub.add_parser("build-f1"); f1_build.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f1_tests = sub.add_parser("record-f1-tests"); f1_tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); f1_tests.add_argument("--passed", type=int, required=True); f1_tests.add_argument("--failed", type=int, required=True); f1_tests.add_argument("--test-command", required=True)
    f1_finalize = sub.add_parser("finalize-f1"); f1_finalize.add_argument("--repo-root", type=Path, default=REPO_ROOT); f1_finalize.add_argument("--completed-at", required=True)
    f1_completion = sub.add_parser("validate-f1-completion"); f1_completion.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f1_gate = sub.add_parser("verify-f1-startup"); f1_gate.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f2_start = sub.add_parser("initialize-f2"); f2_start.add_argument("--repo-root", type=Path, default=REPO_ROOT); f2_start.add_argument("--renderer-commit", required=True); f2_start.add_argument("--started-at", required=True)
    f2_build = sub.add_parser("build-f2"); f2_build.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f2_tests = sub.add_parser("record-f2-tests"); f2_tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); f2_tests.add_argument("--passed", type=int, required=True); f2_tests.add_argument("--failed", type=int, required=True); f2_tests.add_argument("--test-command", required=True)
    f2_finalize = sub.add_parser("finalize-f2"); f2_finalize.add_argument("--repo-root", type=Path, default=REPO_ROOT); f2_finalize.add_argument("--completed-at", required=True)
    f2_completion = sub.add_parser("validate-f2-completion"); f2_completion.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f2_gate = sub.add_parser("verify-f2-startup"); f2_gate.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f3_start = sub.add_parser("initialize-f3"); f3_start.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3_start.add_argument("--renderer-commit", required=True); f3_start.add_argument("--started-at", required=True)
    f3_build = sub.add_parser("build-f3"); f3_build.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f3_tests = sub.add_parser("record-f3-tests"); f3_tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3_tests.add_argument("--passed", type=int, required=True); f3_tests.add_argument("--failed", type=int, required=True); f3_tests.add_argument("--test-command", required=True)
    f3_finalize = sub.add_parser("finalize-f3"); f3_finalize.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3_finalize.add_argument("--completed-at", required=True)
    f3_completion = sub.add_parser("validate-f3-completion"); f3_completion.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f3_gate = sub.add_parser("verify-f3-startup"); f3_gate.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f3_selection = sub.add_parser("validate-f3-selection"); f3_selection.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f3_outputs = sub.add_parser("validate-f3-outputs"); f3_outputs.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def run(arguments: list[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(arguments)
    if args.command == "verify-preconditions": return verify_preconditions(args.repo_root)
    if args.command == "validate-contracts": return {"status": "PASS", "contracts": {name: value["canonical_digest"] for name, value in load_contracts(args.repo_root).items()}}
    if args.command == "parse-recipes": return parse_upstream_recipes(args.repo_root)
    if args.command == "validate-boundary": return assert_f0_source_boundary(args.repo_root)
    if args.command == "render-golden": return render_fixture(args.fixture, args.output_dir, repo_root=args.repo_root, renderer_commit=args.renderer_commit)
    if args.command == "validate-render": return validate_render_report(args.report, repo_root=args.repo_root)
    from . import stage_f_f0 as f0
    if args.command == "initialize-f0":
        startup = json.loads(args.startup_gate.read_text(encoding="utf-8"))
        return f0.initialize(args.repo_root, renderer_commit=args.renderer_commit, started_at=args.started_at, startup_gate=startup)
    if args.command == "build-f0": return f0.build_outputs(args.repo_root)
    if args.command == "record-tests": return f0.record_test_report(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    if args.command == "finalize-f0": return f0.finalize(args.repo_root, completed_at=args.completed_at)
    if args.command == "validate-completion": return f0.validate_completion(args.repo_root)
    from . import stage_f_f1 as f1
    if args.command == "verify-f1-startup": return f1.verify_startup_gates(args.repo_root)
    if args.command == "initialize-f1": return f1.initialize(args.repo_root, renderer_commit=args.renderer_commit, started_at=args.started_at)
    if args.command == "build-f1": return f1.build_outputs(args.repo_root)
    if args.command == "record-f1-tests": return f1.record_test_report(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    if args.command == "finalize-f1": return f1.finalize(args.repo_root, completed_at=args.completed_at)
    if args.command == "validate-f1-completion": return f1.validate_completion(args.repo_root)
    from . import stage_f_f2 as f2
    if args.command == "verify-f2-startup": return f2.verify_startup_gates(args.repo_root)
    if args.command == "initialize-f2": return f2.initialize(args.repo_root, renderer_commit=args.renderer_commit, started_at=args.started_at)
    if args.command == "build-f2": return f2.build_outputs(args.repo_root)
    if args.command == "record-f2-tests": return f2.record_test_report(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    if args.command == "finalize-f2": return f2.finalize(args.repo_root, completed_at=args.completed_at)
    if args.command == "validate-f2-completion": return f2.validate_completion(args.repo_root)
    from . import stage_f_f3 as f3
    if args.command == "verify-f3-startup": return f3.verify_startup_gates(args.repo_root)
    if args.command == "initialize-f3": return f3.initialize(args.repo_root, renderer_commit=args.renderer_commit, started_at=args.started_at)
    if args.command == "build-f3": return f3.build_outputs(args.repo_root)
    if args.command == "record-f3-tests": return f3.record_test_report(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command)
    if args.command == "finalize-f3": return f3.finalize(args.repo_root, completed_at=args.completed_at)
    if args.command == "validate-f3-selection": return f3.validate_selection_manifests(args.repo_root)
    if args.command == "validate-f3-outputs": return f3.validate_outputs(args.repo_root)
    return f3.validate_completion(args.repo_root)


def main(arguments: list[str] | None = None) -> int:
    try:
        result = run(arguments)
    except (StageFRendererError, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
