from pathlib import Path
import copy,json,pytest
from vitnns import stage_x_x1_r1_dependency as dep
REPO=Path(__file__).resolve().parents[1]
def test_amendment_is_closed_signed_and_upstream_bound():
    config=dep.load(REPO);assert config["next_authorization"]=="X1_R1_NNS_V2_IMPLEMENTATION_ONLY";assert config["limits"]["download_bytes_max"]==64*1024*1024;assert len(config["repositories"])==2
def test_allowlists_exclude_binary_scientific_assets():
    config=dep.load(REPO);assert all(pattern.endswith((".py","LICENSE","README.md","pyproject.toml","setup.py")) for row in config["repositories"] for pattern in row["allowlist"]);assert ".pth" in config["forbidden_suffixes"] and ".jpeg" in config["forbidden_suffixes"]
def test_materialized_manifest_if_present():
    if (REPO/dep.MANIFEST).exists():assert dep.validate_materialized(REPO)["status"]=="PASS"
def test_config_tamper_rejected(monkeypatch):
    original=dep.load(REPO);changed=copy.deepcopy(original);changed["limits"]["automatic_waiver"]=True
    monkeypatch.setattr(dep,"read",lambda path:changed if path==REPO/dep.CONFIG else json.loads(path.read_text()))
    with pytest.raises(dep.DependencyError):dep.load(REPO)
