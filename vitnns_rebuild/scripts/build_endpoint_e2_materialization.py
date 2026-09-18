#!/usr/bin/env python3
"""Materialize and hash exactly the 400 frozen ENDPOINT-E2 members."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

REPO = Path(__file__).resolve().parents[1]
EP0 = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918"
TAR = REPO / "assets/imagenet/ILSVRC2012_img_val.tar"
TAR_BYTES = 6744924160
TAR_SHA256 = "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0"


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def write_new(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.cache_root.exists():
        raise RuntimeError("MATERIALIZATION_ROOT_EXISTS")
    if TAR.stat().st_size != TAR_BYTES or sha_file(TAR) != TAR_SHA256:
        raise RuntimeError("ARCHIVE_IDENTITY")
    cohort = json.loads((EP0 / "cohort_registry.json").read_text())
    rows = sorted(
        (row for row in cohort["rows"] if row["split"] == "evaluation"),
        key=lambda row: row["split_order"],
    )
    expected = {row["sample_id"]: row for row in rows}
    if len(rows) != 400 or len(expected) != 400 or [row["split_order"] for row in rows] != list(range(400)):
        raise RuntimeError("EVALUATION_REGISTRY")
    records = []
    seen = set()
    args.cache_root.mkdir(parents=True)
    with tarfile.open(TAR, "r:") as archive:
        for member in archive:
            name = Path(member.name).name
            if name not in expected:
                continue
            if name in seen or not member.isfile():
                raise RuntimeError("DUPLICATE_OR_NONFILE_MEMBER")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("ARCHIVE_MEMBER_READ")
            data = handle.read()
            seen.add(name)
            destination = args.cache_root / name
            write_new(destination, data)
            frozen = expected[name]
            records.append({
                "split_order": frozen["split_order"], "sample_id": name,
                "class_id": frozen["class_id"], "wnid": frozen["wnid"],
                "source_object_id": frozen["source_object_id"],
                "archive_member": member.name, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "cache_path": destination.relative_to(REPO).as_posix(),
            })
    records.sort(key=lambda row: row["split_order"])
    missing = sorted(set(expected) - seen)
    if missing or len(records) != 400:
        raise RuntimeError("MISSING_EVALUATION_MEMBERS:" + ",".join(missing))
    receipt = {
        "schema_version": 1, "stage": "ENDPOINT-E2",
        "status": "PASS_EVALUATION_MATERIALIZATION",
        "archive": {"path": TAR.relative_to(REPO).as_posix(), "bytes": TAR_BYTES, "sha256": TAR_SHA256},
        "authorized_member_count": 400, "archive_member_reads": 400,
        "evaluation_decode_count": 0, "calibration_member_reads": 0,
        "discovery_member_reads": 0, "replacement_count": 0,
        "records": records, "records_digest": digest(records),
    }
    write_new(args.receipt, json.dumps(
        receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False,
    ).encode() + b"\n")
    print(json.dumps({"status": receipt["status"], "records_digest": receipt["records_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
