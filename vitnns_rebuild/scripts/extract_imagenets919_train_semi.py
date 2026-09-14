#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile


REPO = Path(__file__).resolve().parents[1]
PART_ROOT = REPO / "assets/imagenet/ILSVRC2012_img_train.parts"
OUTPUT_ROOT = REPO / "assets/imagenet/imagenets919_train_semi"
SOURCE = REPO / "evidence/nns_vit_patch_depth/s0_r1_20260914/source_evidence/imagenets_data_names_ImageNetSemi_im919_train_txt.json"
RECEIPT = OUTPUT_ROOT / "extraction_receipt.json"
PARTS = (
    ("ILSVRC2012_img_train.tar.001", 41943040000, "aa2ac6c08c01ff312f4a1dc58e61e1591e4604cdc1f2f40d84715d15c863ebeb"),
    ("ILSVRC2012_img_train.tar.002", 41943040000, "42cd63901b9dd676d0e449f160a192c7f3aec0d2e34b7202f80bfab86018c7be"),
    ("ILSVRC2012_img_train.tar.003", 41943040000, "1350fdba89f83945683fb8556c24f55383fb1c58288cb34e99efb9ef90fb58f9"),
    ("ILSVRC2012_img_train.tar.004", 22068357120, "0df379b9fe167d3fd843639796672bb0cdbcef38220c50dfd183cf1b73c15d39"),
)
WHOLE_BYTES = 147897477120
WHOLE_MD5 = "1d675b47d978889d74fa0da5fadfb00e"
WNID = re.compile(r"n\d{8}")
JPEG = re.compile(r"n\d{8}_\d+\.JPEG")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(value):
    return dict(value, canonical_digest=digest(value))


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    before = path.stat()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "file changed while hashing")
    return h.hexdigest()


def jpeg_header(raw):
    require(raw[:2] == b"\xff\xd8", "selected member is not JPEG")
    offset = 2
    width = height = None
    orientation = None
    exif_state = "ABSENT"
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset < len(raw):
        require(raw[offset] == 0xFF, "JPEG marker alignment")
        while offset < len(raw) and raw[offset] == 0xFF:
            offset += 1
        require(offset < len(raw), "truncated JPEG marker")
        marker = raw[offset]
        offset += 1
        if marker in (0xDA, 0xD9):
            break
        if marker in {0x01, 0xD8, *range(0xD0, 0xD8)}:
            continue
        require(offset + 2 <= len(raw), "truncated JPEG segment")
        size = int.from_bytes(raw[offset:offset + 2], "big")
        require(size >= 2 and offset + size <= len(raw), "invalid JPEG segment length")
        segment = raw[offset + 2:offset + size]
        if marker in sof:
            require(len(segment) >= 6, "short JPEG SOF")
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
        if marker == 0xE1 and segment.startswith(b"Exif\0\0"):
            tiff = segment[6:]
            try:
                require(len(tiff) >= 8 and tiff[:2] in (b"II", b"MM"), "bad TIFF header")
                order = "<" if tiff[:2] == b"II" else ">"
                require(struct.unpack_from(order + "H", tiff, 2)[0] == 42, "bad TIFF version")
                position = struct.unpack_from(order + "I", tiff, 4)[0]
                count = struct.unpack_from(order + "H", tiff, position)[0]
                require(position + 2 + count * 12 <= len(tiff), "short EXIF IFD")
                exif_state = "ORIENTATION_TAG_ABSENT"
                for index in range(count):
                    item = position + 2 + index * 12
                    tag, kind, count_value = struct.unpack_from(order + "HHI", tiff, item)
                    if tag == 274:
                        require(kind == 3 and count_value == 1, "invalid EXIF orientation")
                        orientation = struct.unpack_from(order + "H", tiff, item + 8)[0]
                        require(1 <= orientation <= 8, "out-of-range EXIF orientation")
                        exif_state = "PARSED"
            except (ValueError, struct.error):
                exif_state = "MALFORMED_OR_UNSUPPORTED"
                orientation = None
        offset += size
    require(width and height, "missing JPEG dimensions")
    return {
        "width": width,
        "height": height,
        "exif_orientation": orientation,
        "exif_state": exif_state,
        "rgb_pixels_decoded": False,
    }


def load_roster():
    source = json.loads(SOURCE.read_text())
    unsigned = {key: value for key, value in source.items() if key != "canonical_digest"}
    require(source["canonical_digest"] == digest(unsigned), "source canonical digest mismatch")
    text = source["decoded_blob"]["text"]
    require(hashlib.sha256(text.encode()).hexdigest() == source["decoded_blob"]["sha256"], "source blob mismatch")
    rows = []
    by_source_wnid = {}
    for line in text.splitlines():
        image, mask = line.split()
        image_path = PurePosixPath(image)
        mask_path = PurePosixPath(mask)
        require(len(image_path.parts) == len(mask_path.parts) == 2, "invalid official relative path")
        require(image_path.parent == mask_path.parent and image_path.stem == mask_path.stem, "image-mask pair mismatch")
        require(JPEG.fullmatch(image_path.name), "invalid train image name")
        source_wnid = image_path.name.split("_", 1)[0]
        require(WNID.fullmatch(source_wnid), "invalid source WNID")
        row = {
            "annotation_wnid": image_path.parts[0],
            "image_relative_path": image,
            "mask_relative_path": mask,
            "source_wnid": source_wnid,
        }
        rows.append(row)
        by_source_wnid.setdefault(source_wnid, {})[image_path.name] = row
    require(len(rows) == 9190 and len(by_source_wnid) == 924, "unexpected official train-semi roster")
    return rows, by_source_wnid, source


class ConcatenatedParts(io.RawIOBase):
    def __init__(self, paths):
        self.streams = [path.open("rb") for path in paths]
        self.index = 0
        self.position = 0

    def readable(self):
        return True

    def readinto(self, target):
        view = memoryview(target)
        written = 0
        while written < len(view) and self.index < len(self.streams):
            count = self.streams[self.index].readinto(view[written:])
            if count:
                written += count
                self.position += count
            else:
                self.index += 1
        return written

    def tell(self):
        return self.position

    def close(self):
        for stream in self.streams:
            stream.close()
        super().close()


def verify_parts():
    records = []
    whole_md5 = hashlib.md5()
    whole_sha256 = hashlib.sha256()
    total = 0
    for name, expected_bytes, expected_sha256 in PARTS:
        path = PART_ROOT / name
        require(path.is_file() and path.stat().st_size == expected_bytes, "part size mismatch: " + name)
        part_hash = hashlib.sha256()
        before = path.stat()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                part_hash.update(chunk)
                whole_md5.update(chunk)
                whole_sha256.update(chunk)
                total += len(chunk)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "part changed while hashing")
        require(part_hash.hexdigest() == expected_sha256, "part SHA-256 mismatch: " + name)
        records.append({"path": str(path.relative_to(REPO)), "bytes": before.st_size, "sha256": part_hash.hexdigest()})
    require(total == WHOLE_BYTES and whole_md5.hexdigest() == WHOLE_MD5, "whole archive identity mismatch")
    return records, {"bytes": total, "md5": whole_md5.hexdigest(), "sha256": whole_sha256.hexdigest()}


def extract_selected(by_source_wnid):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    require(not RECEIPT.exists(), "terminal extraction receipt already exists")
    extracted = []
    outer_count = 0
    selected_class_count = 0
    part_paths = [PART_ROOT / name for name, _, _ in PARTS]
    with ConcatenatedParts(part_paths) as raw:
        buffered = io.BufferedReader(raw, buffer_size=8 * 1024 * 1024)
        with tarfile.open(fileobj=buffered, mode="r|") as outer:
            for outer_member in outer:
                outer_count += 1
                require(outer_member.isfile(), "unexpected outer tar entry")
                outer_name = PurePosixPath(outer_member.name).name
                require(outer_name.endswith(".tar") and WNID.fullmatch(outer_name[:-4]), "invalid class tar")
                source_wnid = outer_name[:-4]
                wanted = by_source_wnid.get(source_wnid)
                if wanted is None:
                    continue
                selected_class_count += 1
                class_stream = outer.extractfile(outer_member)
                require(class_stream is not None, "cannot read class tar")
                found = set()
                with tarfile.open(fileobj=class_stream, mode="r|") as inner:
                    for image_member in inner:
                        require(image_member.isfile(), "unexpected inner tar entry")
                        image_name = PurePosixPath(image_member.name).name
                        if image_name not in wanted:
                            continue
                        source = inner.extractfile(image_member)
                        require(source is not None and image_name not in found, "invalid selected member")
                        data = source.read()
                        require(len(data) == image_member.size, "short selected image")
                        row = wanted[image_name]
                        destination = OUTPUT_ROOT / row["image_relative_path"]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            require(destination.read_bytes() == data, "existing selected image mismatch")
                        else:
                            with destination.open("xb") as stream:
                                stream.write(data)
                                stream.flush()
                                os.fsync(stream.fileno())
                        found.add(image_name)
                        extracted.append({
                            **row,
                            "bytes": len(data),
                            "image_header": jpeg_header(data),
                            "local_path": str(destination.relative_to(REPO)),
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "source_class_tar": outer_member.name,
                            "source_member": image_member.name,
                        })
                require(found == set(wanted), "selected members missing from " + outer_name)
    extracted.sort(key=lambda row: row["image_relative_path"])
    require(outer_count == 1000 and selected_class_count == 924 and len(extracted) == 9190, "extraction count mismatch")
    return extracted, outer_count, selected_class_count


def verify_extraction(rows):
    receipt = json.loads(RECEIPT.read_text())
    unsigned = {key: value for key, value in receipt.items() if key != "canonical_digest"}
    require(receipt["canonical_digest"] == digest(unsigned), "receipt canonical digest mismatch")
    records = receipt["payload"]["records"]
    require(len(records) == len(rows) == 9190, "receipt count mismatch")
    require(receipt["payload"]["records_digest"] == digest(records), "records digest mismatch")
    require([row["image_relative_path"] for row in records] == sorted(row["image_relative_path"] for row in rows), "roster mismatch")
    for record in records:
        path = REPO / record["local_path"]
        require(path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"], "extracted identity mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    rows, by_source_wnid, source = load_roster()
    if args.verify:
        receipt = verify_extraction(rows)
        print("PASS: 9190 extracted train-semi originals match sealed byte identities")
        print(json.dumps({"records_digest": receipt["payload"]["records_digest"], "whole_archive": receipt["payload"]["whole_archive"]}, sort_keys=True))
        return
    part_records, whole = verify_parts()
    records, outer_count, selected_class_count = extract_selected(by_source_wnid)
    receipt = signed({
        "schema_version": 1,
        "document_type": "imagenets919_train_semi_extraction_receipt",
        "source_list": {
            "path": str(SOURCE.relative_to(REPO)),
            "canonical_digest": source["canonical_digest"],
            "decoded_blob_sha256": source["decoded_blob"]["sha256"],
        },
        "payload": {
            "status": "PASS",
            "parts": part_records,
            "whole_archive": whole,
            "outer_class_tar_count": outer_count,
            "selected_class_tar_count": selected_class_count,
            "selected_image_count": len(records),
            "records": records,
            "records_digest": digest(records),
            "access_boundary": {
                "selected_jpeg_bytes_read_and_written": len(records),
                "rgb_pixels_decoded": 0,
                "model_loaded": False,
                "gpu_used": False,
                "inference_run": False,
            },
        },
    })
    with RECEIPT.open("xb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("PASS: extracted and sealed 9190 ImageNet-S919 train-semi originals")
    print(json.dumps({"records_digest": receipt["payload"]["records_digest"], "whole_archive": whole}, sort_keys=True))


if __name__ == "__main__":
    main()
