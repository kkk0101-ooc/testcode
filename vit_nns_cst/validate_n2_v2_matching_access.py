#!/usr/bin/env python3
"""Validate archived openat traces for N2-V2 matching R2."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_matching_r2_20260922"
ACCEPTED = ROOT / "evidence/n2_v2_i7_20260922/accepted_scene_only_assets.json"
CONTRACT = ROOT / "config/n2_v2/input_identifiability_contract.json"
LEDGER = ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json"
ASSETS = ROOT / "evidence/n2_v2_i5_20260922"
SENSITIVE = re.compile(r"(?i)(checkpoint|behavior|activation|endpoint|path.?nns|jvp|intervention|\.(pt|pth|ckpt|safetensors)$)")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_paths(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r'openat\([^,]+, "([^"]+)", ([^)]+)\)', line)
        if not match:
            continue
        target = Path(match.group(1))
        if not target.is_absolute():
            continue
        try:
            relative = str(target.resolve().relative_to(ROOT))
        except (ValueError, OSError):
            continue
        rows.append((relative, match.group(2)))
    return rows


def main() -> None:
    audit = json.loads((OUT / "access_audit.json").read_text())
    accepted = json.loads(ACCEPTED.read_text())
    allowed = {str(ACCEPTED.relative_to(ROOT)), str(CONTRACT.relative_to(ROOT)), str(LEDGER.relative_to(ROOT))}
    allowed |= {str((ASSETS / row["image_path"]).relative_to(ROOT)) for row in accepted["records"]}
    allowed |= {str(path.relative_to(ROOT)) for directory in ("recipient_images", "recipient_masks") for path in (ASSETS / "private" / directory).iterdir() if path.is_file()}
    specifications = (
        ("production_trace", OUT / "production_openat.log", "build_n2_v2_matching.py"),
        ("validator_trace", OUT / "validator_openat.log", "validate_n2_v2_matching.py"),
    )
    for key, trace, implementation in specifications:
        rows = trace_paths(trace)
        unique = set(rows)
        paths = {path for path, _ in unique}
        directories = {path for path, flags in unique if "O_DIRECTORY" in flags}
        outputs = {path for path in paths if path.startswith("evidence/n2_v2_matching_r2_20260922/")}
        observed_inputs = paths & allowed
        unexpected = paths - allowed - directories - outputs - {implementation}
        sensitive = sorted(path for path in paths if SENSITIVE.search(path))
        expected = audit[key]
        assert sha(trace) == expected["trace_sha256"]
        assert len(rows) == expected["repository_open_record_count"]
        assert len(unique) == expected["repository_unique_path_flag_pair_count"]
        assert len(observed_inputs) == len(allowed) == expected["allowed_input_file_count_observed"] == 70
        assert not (allowed - observed_inputs)
        assert not unexpected
        assert not sensitive
        print(json.dumps({"trace": trace.name, "status": "PASS", "allowed_inputs": len(observed_inputs), "unexpected": 0, "sensitive": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
