#!/usr/bin/env python3
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request


REPO = Path(__file__).resolve().parents[1]
PIN = "b2312ba7396151f883362bdc86d71469aa81066e"
API = "https://api.github.com/repos/LUSSeg/ImageNet-S"
FILES = ["README.md", "data/categories/ImageNetS_categories_im919.txt",
         "data/mapping/ImageNetS_im919_mapping_validation.txt", "data/mapping/ImageNetS_im919_mapping_test.txt",
         "data/names/ImageNetS_im919_validation.txt", "data/names/ImageNetSemi_im919_train.txt",
         "datapreparation/datapreparation_anno.sh", "datapreparation/datapreparation_train_semi.py",
         "datapreparation/datapreparation_val.py", "data/data.py"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def seal(value):
    return dict(value, canonical_digest=hashlib.sha256(canonical(value)).hexdigest())


def save(path, value):
    with path.open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = (REPO / args.root).resolve()
    if REPO / "evidence/nns_vit_patch_depth" not in root.parents:
        raise ValueError("root outside new study")
    if args.resume:
        if (root / "feasibility_decision.json").exists():
            raise ValueError("terminal root immutable")
        plan = json.loads((root / "acquisition_plan.json").read_text())
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root / "source_evidence").mkdir()
        plan = seal({"stage_id": "S0-R1", "status": "METADATA_ACQUISITION_ONLY",
                     "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                     "worktree_status": subprocess.check_output(["git", "-C", str(REPO), "status", "--short"], text=True),
                     "repository_pin": PIN, "metadata_files": FILES, "per_response_cap_bytes": 8 * 1024 * 1024,
                     "free_disk_bytes": shutil.disk_usage(str(REPO)).free,
                     "annotation_archive_candidate": {"name": "ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip", "official_api_size_bytes_observed": 61281494,
                                                      "download_permitted": False, "reason": "no explicit annotation-use terms located yet"},
                     "raw_images": "reuse local validation archive only; no image download", "mask_download_count": 0,
                     "license_distinction": "public documentation/index read for audit only; not annotation reuse or redistribution permission",
                     "rgb_decode_allowed": False, "mask_decode_started": False, "gpu_allowed": False})
        save(root / "acquisition_plan.json", plan)
    receipts = []
    failures = []

    def fetch(key, url, blob=None):
        dest = root / "source_evidence" / (key + ".json")
        if dest.exists():
            value = json.loads(dest.read_text())
            if value.get("http_status") == 200:
                receipts.append({"path": str(dest.relative_to(root)), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
                return value
        request = urllib.request.Request(url, headers={"User-Agent": "vitnns-s0-r1-public-metadata-audit"})
        value = {"url": url, "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "method": "GET", "http_status": None,
                 "final_url": None, "response_headers": {}, "body_sha256": None, "body_bytes": 0, "body_text": None, "decoded_blob": None, "error": None}
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError("response exceeds metadata cap")
                value.update(http_status=response.status, final_url=response.geturl(), response_headers={k.lower(): v for k, v in response.headers.items()},
                             body_sha256=hashlib.sha256(raw).hexdigest(), body_bytes=len(raw), body_text=raw.decode("utf-8"))
            if blob is not None:
                item = json.loads(value["body_text"])
                content = base64.b64decode(item["content"])
                git_sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                if git_sha != blob:
                    raise ValueError("official git blob identity mismatch")
                value["decoded_blob"] = {"git_blob_sha1": git_sha, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "text": content.decode("utf-8")}
        except (urllib.error.URLError, OSError, ValueError) as error:
            value["error"] = type(error).__name__ + ": " + str(error)
            failures.append({"key": key, "url": url, "error": value["error"]})
        if dest.exists():
            number = 2
            while dest.with_name(key + ".attempt-%d.json" % number).exists():
                number += 1
            dest = dest.with_name(key + ".attempt-%d.json" % number)
        save(dest, seal(value))
        receipts.append({"path": str(dest.relative_to(root)), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
        print(key, value["http_status"], value["body_bytes"], value["error"], flush=True)
        return value

    tree_response = fetch("imagenets_tree", API + "/git/trees/" + PIN + "?recursive=1")
    tree = json.loads(tree_response["body_text"])["tree"] if tree_response["http_status"] == 200 else []
    by_path = {r["path"]: r for r in tree}
    for path in FILES:
        if path not in by_path:
            failures.append({"key": path, "url": API, "error": "missing tree entry"})
            continue
        fetch("imagenets_" + path.replace("/", "_").replace(".", "_"), by_path[path]["url"], by_path[path]["sha"])
    for key, url in [
        ("imagenets_repository", API), ("imagenets_head", API + "/commits/main"),
        ("imagenets_releases", API + "/releases"), ("imagenets_issues", API + "/issues?state=all&per_page=100"),
        ("imagenets_issue8_comments", API + "/issues/8/comments"),
        ("imagenets_project", "https://lusseg.github.io/"), ("imagenet_terms", "https://www.image-net.org/download.php"),
        ("partimagenet_repository", "https://api.github.com/repos/TACJu/PartImageNet"),
        ("partimagenet_tree", "https://api.github.com/repos/TACJu/PartImageNet/git/trees/f4bf3df88b126d3a2d5e8671a8c2ea90de39638e?recursive=1"),
        ("partimagenet_readme", "https://api.github.com/repos/TACJu/PartImageNet/git/blobs/66c3e10fe4e5fdc7bf48c056775dc6838dfa0640"),
        ("partimagenet_dataset_card", "https://huggingface.co/api/datasets/turkeyju/PartImageNet"),
        ("imagenets_paper", "https://arxiv.org/abs/2106.03149")]:
        fetch(key, url, "66c3e10fe4e5fdc7bf48c056775dc6838dfa0640" if key == "partimagenet_readme" else None)
    name = "acquisition_receipt.json" if not (root / "acquisition_receipt.json").exists() else "acquisition_receipt_retry.json"
    save(root / name, seal({"status": "COMPLETE_WITH_UNAVAILABLE_SOURCES" if failures else "COMPLETE", "receipts": receipts, "failures": failures,
                           "annotation_archives_downloaded": 0, "annotation_masks_decoded": 0, "rgb_images_decoded": 0,
                           "gh_cli_limitation": "installed /usr/local/bin/gh is an internal hostname utility; official HTTPS API used read-only"}))


if __name__ == "__main__":
    main()
