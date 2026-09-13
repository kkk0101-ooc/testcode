#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import py_compile
import tempfile
from pathlib import Path

from vitnns import stage_y_y4_r5 as y

def tests(repo):
    class Results:
        def __init__(self):
            self.rows = []
        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                self.rows.append(report.nodeid)
    result = Results()
    import pytest
    rc = pytest.main([
        str(repo / "tests/test_stage_y_y4_r5.py"),
        str(repo / "tests/test_stage_y_y4_r4.py"),
        str(repo / "tests/test_stage_y_y4_r3.py"),
        str(repo / "tests/test_stage_y_y4_r1.py"),
        "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
    ], plugins=[result])
    y.require(rc == 0, "focused tests failed", y.StartupError)
    return len(result.rows)

def preflight(repo, output):
    gate = y.startup(repo, True)
    y.write_schema_sources(repo)
    y.validate_schema_sources(repo)
    count = tests(repo)
    with tempfile.TemporaryDirectory(prefix="y4-r5-pyc-") as tmp:
        for index, path in enumerate(y.PRODUCER_FILES):
            if path.suffix == ".py":
                py_compile.compile(str(repo / path), cfile=f"{tmp}/{index}.pyc", doraise=True)
    payload = {
        "startup_commit": y.STARTUP_COMMIT,
        "checks": [
            "Y4 through Y4-R4 immutable",
            "sealed R3/R4 cut retained",
            "spatial cancellation registered field preserved",
            "non-numeric classification stratum excluded from trajectories",
            "registered zero-mass NA propagation retained",
            "all prior outcome-blind gates inherited",
        ],
        "test_count": count,
        "false_pass_attack_count": len(y.FALSE_PASS_ATTACKS),
        "startup_refs_digest": gate["startup_refs_digest"],
        "formal_outcome_access": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(y.canonical_bytes(payload) + b"\n")
    return {"status": "PASS", **payload}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "preflight", "seal-producer", "analyze", "seal", "validate", "validate-publication"
    ))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    y.runtime_guard()
    try:
        if args.command == "preflight":
            out = preflight(repo, args.output)
        elif args.command == "seal-producer":
            out = y.seal_producer(repo, json.loads(args.preflight.read_text()))
        else:
            from scripts import validate_stage_y_y4_r5 as validator
            if args.command == "analyze":
                out = validator.produce(repo)
            elif args.command == "seal":
                out = validator.seal(repo)
            else:
                out = validator.validate_completion(repo, args.command == "validate-publication")
    except Exception as error:
        print(json.dumps({
            "status": getattr(error, "code", "BLOCKED_INFRASTRUCTURE"),
            "error": str(error), "next_authorization": "NONE",
        }, sort_keys=True))
        return 2
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
