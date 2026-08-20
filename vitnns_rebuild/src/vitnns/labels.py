from __future__ import annotations

import json
import re
import tarfile
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scipy.io import loadmat
from timm.data import ImageNetInfo

from .constants import IMAGENET_SYNSET_ORDER_SHA256, OFFICIAL_DEVKIT_SHA256
from .io_utils import sha256_file

VAL_PATTERN = re.compile(r"^ILSVRC2012_val_(\d{8})\.(?:JPEG|jpg|jpeg)$")


@dataclass(frozen=True)
class ImageNetClass:
    index: int
    wnid: str
    description: str


class ImageNet1KMapping:
    """Canonical timm/ImageFolder ImageNet-1k output-index mapping."""

    def __init__(self) -> None:
        info = ImageNetInfo("imagenet-1k")
        wnids = list(info.label_names())
        import hashlib

        digest = hashlib.sha256(("\n".join(wnids) + "\n").encode()).hexdigest()
        if digest != IMAGENET_SYNSET_ORDER_SHA256:
            raise RuntimeError(f"Unexpected ImageNet synset order hash: {digest}")
        self.classes = tuple(
            ImageNetClass(i, wnid, info.label_name_to_description(wnid))
            for i, wnid in enumerate(wnids)
        )
        self.wnid_to_index = {item.wnid: item.index for item in self.classes}
        if len(self.classes) != 1000 or len(self.wnid_to_index) != 1000:
            raise RuntimeError("ImageNet-1k mapping is not bijective")

    def index_for_wnid(self, wnid: str) -> int:
        try:
            return self.wnid_to_index[wnid]
        except KeyError as exc:
            raise ValueError(f"Unknown ImageNet-1k WNID: {wnid}") from exc

    def to_json(self) -> list[dict]:
        return [item.__dict__ for item in self.classes]


def parse_official_devkit(devkit_path: Path) -> tuple[dict[int, str], list[int]]:
    """Return official ILSVRC ID->WNID and validation-row->ILSVRC ID."""
    devkit_path = Path(devkit_path).resolve()
    actual = sha256_file(devkit_path)
    if actual != OFFICIAL_DEVKIT_SHA256:
        raise ValueError(f"Official devkit SHA-256 mismatch: {actual}")
    with TemporaryDirectory(prefix="vitnns-devkit-") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(devkit_path, "r:gz") as archive:
            wanted = {
                "ILSVRC2012_devkit_t12/data/meta.mat",
                "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
            }
            members = {member.name: member for member in archive.getmembers()}
            if not wanted.issubset(members):
                raise ValueError("Devkit is missing meta.mat or validation ground truth")
            for name in wanted:
                source = archive.extractfile(members[name])
                if source is None:
                    raise ValueError(f"Cannot read {name} from devkit")
                (tmp_path / Path(name).name).write_bytes(source.read())
        synsets = loadmat(tmp_path / "meta.mat", squeeze_me=True, struct_as_record=False)["synsets"]
        id_to_wnid = {
            int(item.ILSVRC2012_ID): str(item.WNID)
            for item in synsets
            if int(item.num_children) == 0
        }
        ground_truth = [
            int(line)
            for line in (tmp_path / "ILSVRC2012_validation_ground_truth.txt").read_text().splitlines()
        ]
    if set(id_to_wnid) != set(range(1, 1001)) or len(set(id_to_wnid.values())) != 1000:
        raise ValueError("Devkit leaf class mapping is not a 1..1000 bijection")
    if set(id_to_wnid.values()) != set(ImageNet1KMapping().wnid_to_index):
        raise ValueError("Devkit leaf WNIDs do not match the canonical ImageNet-1k mapping")
    if len(ground_truth) != 50_000 or not all(value in id_to_wnid for value in ground_truth):
        raise ValueError("Devkit validation ground truth is invalid")
    if set(Counter(ground_truth).values()) != {50}:
        raise ValueError("Validation ground truth must contain exactly 50 images per class")
    return id_to_wnid, ground_truth


def validation_wnid_map(devkit_path: Path) -> dict[int, str]:
    id_to_wnid, ground_truth = parse_official_devkit(devkit_path)
    return {row + 1: id_to_wnid[class_id] for row, class_id in enumerate(ground_truth)}


def dump_mapping(path: Path) -> None:
    value = {str(item.index): [item.wnid, item.description] for item in ImageNet1KMapping().classes}
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

