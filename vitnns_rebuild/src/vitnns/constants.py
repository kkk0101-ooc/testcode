from pathlib import Path

MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_SHA256 = "2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0"
FULL_MODEL_SHA256 = "a76a7c85db473b539c1756066376d851f3fdccf1cf494b63be81d6adf3236f05"
IMAGENET_SYNSET_ORDER_SHA256 = "70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15"
OFFICIAL_DEVKIT_SHA256 = "b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953"
OFFICIAL_DEVKIT_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
VALIDATION_SYNSET_LABELS_SHA256 = "313fc3f23c864dde9183c9a368809065765dd1b657a6904fded44e3e46d34604"
VALIDATION_SYNSET_LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_STATE_DICT = WORKSPACE_ROOT / "49_vit " / "model" / "vit_base_patch16_224_weights.pth"
DEFAULT_FULL_MODEL = WORKSPACE_ROOT / "49_vit " / "model" / "vit_base_patch16_224_full.pth"
DEFAULT_SCANNER_ROOT = WORKSPACE_ROOT / "49_vit " / "datas" / "imagenet-scanner"
DEFAULT_VAL_ROOT = WORKSPACE_ROOT / "49_vit " / "datas" / "ILSVRC2012_val"

