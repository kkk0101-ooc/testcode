#!/usr/bin/env python3
"""Materialize the first frozen discovery member for Step 4."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
EP0 = REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918"
ARCHIVE = REPO / "vitnns_rebuild/assets/imagenet/ILSVRC2012_img_val.tar"
EXPECTED_ARCHIVE_SHA256 = "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_new(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    binding = json.loads((ROOT / "step4_source_binding.json").read_text())
    if binding["archive"]["sha256"] != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("FROZEN_ARCHIVE_BINDING")
    cohort = json.loads((EP0 / "cohort_registry.json").read_text())
    rows = sorted((row for row in cohort["rows"] if row["split"] == "discovery"),
                  key=lambda row: row["split_order"])
    if len(rows) != 64 or [row["split_order"] for row in rows] != list(range(64)):
        raise RuntimeError("DISCOVERY_REGISTRY")
    selected = rows[0]
    cache = ROOT / "step4_runtime/discovery" / selected["sample_id"]
    receipt_path = ROOT / "step4_runtime/discovery_receipt.json"
    if cache.exists() or receipt_path.exists():
        raise RuntimeError("DISCOVERY_OUTPUT_EXISTS")
    matches = []
    with tarfile.open(ARCHIVE, "r:") as archive:
        for member in archive:
            if Path(member.name).name == selected["sample_id"]:
                if not member.isfile():
                    raise RuntimeError("DISCOVERY_MEMBER_NOT_FILE")
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError("DISCOVERY_MEMBER_READ")
                matches.append((member.name, stream.read()))
    if len(matches) != 1:
        raise RuntimeError("DISCOVERY_MEMBER_COUNT:" + str(len(matches)))
    archive_member, payload = matches[0]
    write_new(cache, payload)
    receipt = {
        "schema_version": 1, "stage": "ENDPOINT_PCIA_PEFA_STEP4",
        "status": "PASS_DISCOVERY_MATERIALIZATION", "split": "discovery",
        "selected_by": "frozen split_order 0 before payload access",
        "sample_id": selected["sample_id"], "split_order": 0,
        "class_id": selected["class_id"], "wnid": selected["wnid"],
        "archive_member": archive_member, "image_bytes": len(payload),
        "image_sha256": sha_bytes(payload), "cache_path": cache.relative_to(REPO).as_posix(),
        "archive_member_reads": 1, "discovery_decode_count": 0,
        "calibration_member_reads": 0, "evaluation_member_reads": 0,
        "replacement_count": 0,
    }
    write_new(receipt_path, json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False).encode() + b"\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
