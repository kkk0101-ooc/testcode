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
    f3h0_check = sub.add_parser("validate-f3h0-protocol"); f3h0_check.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3h0_check.add_argument("--require-clean", action="store_true")
    f3h0_freeze = sub.add_parser("freeze-f3h0-protocol"); f3h0_freeze.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3h0_freeze.add_argument("--frozen-at", required=True); f3h0_freeze.add_argument("--test-passed", required=True, type=int); f3h0_freeze.add_argument("--test-failed", required=True, type=int); f3h0_freeze.add_argument("--test-command", required=True)
    f3h0_completion = sub.add_parser("validate-f3h0-completion"); f3h0_completion.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    for name in ("validate-f3h1-implementation", "validate-f3h1-completion", "seal-f3h-r1-compute", "validate-f3h-initialized", "authorize-f3h-bulk", "reconcile-f3h", "build-f3h-components", "build-f3h-consensus", "assess-f3h-basis", "render-f3h", "build-f3h-audits", "validate-f3h-outputs", "validate-f3h-completion"):
        child = sub.add_parser(name); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f3h1_freeze = sub.add_parser("freeze-f3h1-implementation"); f3h1_freeze.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3h1_freeze.add_argument("--completed-at", required=True); f3h1_freeze.add_argument("--passed", type=int, required=True); f3h1_freeze.add_argument("--failed", type=int, required=True); f3h1_freeze.add_argument("--test-command", required=True)
    f3h_init = sub.add_parser("initialize-f3h"); f3h_init.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3h_init.add_argument("--started-at", required=True)
    f3h_pilot = sub.add_parser("run-f3h-pilot"); f3h_pilot.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3h_pilot.add_argument("--device", default="cuda" + ":0")
    f3h_shard = sub.add_parser("run-f3h-shard"); f3h_shard.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3h_shard.add_argument("--shard-id", type=int, required=True); f3h_shard.add_argument("--shard-count", type=int, default=8); f3h_shard.add_argument("--device", default="cuda" + ":0")
    f3h_gradcam = sub.add_parser("run-f3h-gradcam"); f3h_gradcam.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3h_gradcam.add_argument("--device", default="cuda" + ":0")
    f3h_tests = sub.add_parser("record-f3h-tests"); f3h_tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3h_tests.add_argument("--passed", type=int, required=True); f3h_tests.add_argument("--failed", type=int, required=True); f3h_tests.add_argument("--test-command", required=True)
    f3h_final = sub.add_parser("finalize-f3h"); f3h_final.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3h_final.add_argument("--completed-at", required=True)
    f3c0_check = sub.add_parser("validate-f3c0-protocol"); f3c0_check.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3c0_check.add_argument("--require-clean", action="store_true")
    f3c0_freeze = sub.add_parser("freeze-f3c0-protocol"); f3c0_freeze.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3c0_freeze.add_argument("--frozen-at", required=True); f3c0_freeze.add_argument("--test-passed", required=True, type=int); f3c0_freeze.add_argument("--test-failed", required=True, type=int); f3c0_freeze.add_argument("--test-command", required=True)
    f3c0_completion = sub.add_parser("validate-f3c0-completion"); f3c0_completion.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    for name in ("verify-f3c1-startup", "validate-f3c1-engineering", "build-f3c1-audits", "build-f3c1-false-pass", "validate-f3c1-outputs", "validate-f3c1-completion"):
        child = sub.add_parser(name); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    f3c1_init = sub.add_parser("initialize-f3c1"); f3c1_init.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3c1_init.add_argument("--started-at", required=True)
    f3c1_init_r6 = sub.add_parser("initialize-f3c1-r6"); f3c1_init_r6.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3c1_init_r6.add_argument("--started-at", required=True)
    f3c1_run = sub.add_parser("run-f3c1-engineering"); f3c1_run.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3c1_run.add_argument("--device", default="cuda:0")
    f3c1_tests = sub.add_parser("record-f3c1-tests"); f3c1_tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3c1_tests.add_argument("--passed", type=int, required=True); f3c1_tests.add_argument("--failed", type=int, required=True); f3c1_tests.add_argument("--initial-failed-attempts", type=int, default=0); f3c1_tests.add_argument("--test-command", required=True)
    f3c1_final = sub.add_parser("finalize-f3c1"); f3c1_final.add_argument("--repo-root", type=Path, default=REPO_ROOT); f3c1_final.add_argument("--completed-at", required=True)
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
    if args.command.startswith(("validate-f3h0", "freeze-f3h0")):
        from . import stage_f_f3h_protocol as f3h0
        if args.command == "validate-f3h0-protocol": return f3h0.validate_protocol(args.repo_root, require_clean=args.require_clean)
        if args.command == "freeze-f3h0-protocol": return f3h0.freeze_protocol(args.repo_root, frozen_at=args.frozen_at, test_passed=args.test_passed, test_failed=args.test_failed, test_command=args.test_command)
        return f3h0.validate_completion(args.repo_root)
    if args.command.endswith("f3c0-protocol") or args.command == "validate-f3c0-completion":
        from . import stage_f_f3c_protocol as f3c0
        if args.command == "validate-f3c0-protocol": return f3c0.validate_protocol(args.repo_root, require_clean=args.require_clean)
        if args.command == "freeze-f3c0-protocol": return f3c0.freeze(args.repo_root, frozen_at=args.frozen_at, test_passed=args.test_passed, test_failed=args.test_failed, test_command=args.test_command)
        return f3c0.validate_completion(args.repo_root)
    if "f3c1" in args.command:
        from . import stage_f_f3c1 as f3c1
        commands = {
            "verify-f3c1-startup": lambda: f3c1.verify_startup_gates(args.repo_root),
            "initialize-f3c1": lambda: f3c1.initialize(args.repo_root, started_at=args.started_at),
            "initialize-f3c1-r6": lambda: f3c1.initialize_r6_from_r5(args.repo_root, started_at=args.started_at),
            "run-f3c1-engineering": lambda: f3c1.run_engineering_check(args.repo_root, device=args.device),
            "validate-f3c1-engineering": lambda: f3c1.validate_engineering_terminal(args.repo_root),
            "record-f3c1-tests": lambda: f3c1.record_synthetic_tests(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command, initial_failed_attempts=args.initial_failed_attempts),
            "build-f3c1-audits": lambda: f3c1.build_audits(args.repo_root),
            "build-f3c1-false-pass": lambda: f3c1.build_false_pass_audit(args.repo_root),
            "validate-f3c1-outputs": lambda: f3c1.validate_outputs(args.repo_root),
            "finalize-f3c1": lambda: f3c1.finalize(args.repo_root, completed_at=args.completed_at),
            "validate-f3c1-completion": lambda: f3c1.validate_completion(args.repo_root),
        }
        return commands[args.command]()
    if "f3h" in args.command:
        from . import stage_f_f3h as f3h
        commands = {
            "validate-f3h1-implementation": lambda: f3h.validate_implementation(args.repo_root),
            "validate-f3h1-completion": lambda: f3h.validate_implementation_completion(args.repo_root),
            "seal-f3h-r1-compute": lambda: f3h.seal_r1_compute_failure(args.repo_root),
            "freeze-f3h1-implementation": lambda: f3h.freeze_implementation(args.repo_root, completed_at=args.completed_at, passed=args.passed, failed=args.failed, command=args.test_command),
            "initialize-f3h": lambda: f3h.initialize(args.repo_root, started_at=args.started_at),
            "validate-f3h-initialized": lambda: f3h.validate_initialized(args.repo_root),
            "run-f3h-pilot": lambda: f3h.run_pilot(args.repo_root, device=args.device),
            "authorize-f3h-bulk": lambda: f3h.authorize_bulk(args.repo_root),
            "run-f3h-shard": lambda: f3h.run_shard(args.repo_root, shard_id=args.shard_id, shard_count=args.shard_count, device=args.device),
            "run-f3h-gradcam": lambda: f3h.run_gradcam(args.repo_root, device=args.device),
            "reconcile-f3h": lambda: f3h.reconcile_terminals(args.repo_root),
            "build-f3h-components": lambda: f3h.build_components(args.repo_root),
            "build-f3h-consensus": lambda: f3h.build_consensus_manifest(args.repo_root),
            "assess-f3h-basis": lambda: f3h.assess_class_conditioned_basis(args.repo_root),
            "render-f3h": lambda: f3h.render_outputs(args.repo_root),
            "build-f3h-audits": lambda: f3h.build_final_audits(args.repo_root),
            "record-f3h-tests": lambda: f3h.record_tests(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command),
            "validate-f3h-outputs": lambda: f3h.validate_outputs(args.repo_root),
            "finalize-f3h": lambda: f3h.finalize(args.repo_root, completed_at=args.completed_at),
            "validate-f3h-completion": lambda: f3h.validate_completion(args.repo_root),
        }
        return commands[args.command]()
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
