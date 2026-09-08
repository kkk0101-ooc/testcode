from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_stage_e_plan_stays_byte_identical():
    assert _sha(REPO / "docs/阶段E规划.md") == (
        "bb580deda0ac6969e92c4d3763c6f6c43b241054437346abcbbcb40efd6b1adf"
    )


def test_f3_work_package_index_binds_existing_sealed_artifacts():
    index = json.loads(
        (REPO / "manifests/stage_f_f3_work_packages_v1.json").read_text(encoding="utf-8")
    )
    assert index["stage_f_scope"] == "F0_THROUGH_F3_FIGURE_PRODUCTION"
    assert index["next_stage"] == "STAGE_G_MANUSCRIPT_WRITING"
    assert index["sealed_paths_moved_or_renamed"] is False
    assert [item["canonical_name"] for item in index["packages"]] == [
        "F3-Core", "F3-Display", "F3-HeadAtlas", "F3-ClassCompare",
    ]
    for item in index["packages"]:
        path = REPO / item["completion_path"]
        assert path.is_file()
        assert _sha(path) == item["completion_sha256"]
        commit = subprocess.run(
            ["git", "rev-parse", item["archive_ref"] + "^{commit}"],
            cwd=REPO, text=True, capture_output=True, check=True,
        ).stdout.strip()
        assert commit == item["archive_commit"]
    class_compare = index["packages"][-1]
    assert _sha(REPO / class_compare["report_copy_completion_path"]) == (
        class_compare["report_copy_completion_sha256"]
    )
    assert _sha(REPO / class_compare["display_package_completion_path"]) == (
        class_compare["display_package_completion_sha256"]
    )
    copy_commit = subprocess.run(
        ["git", "rev-parse", class_compare["report_copy_archive_ref"] + "^{commit}"],
        cwd=REPO, text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert copy_commit == class_compare["report_copy_archive_commit"]


def test_current_stage_navigation_separates_figures_and_writing():
    stage_f = (REPO / "docs/阶段F规划.md").read_text(encoding="utf-8")
    stage_g = (REPO / "docs/阶段G规划.md").read_text(encoding="utf-8")
    guide = (REPO / "docs/阶段G论文写作指导.md").read_text(encoding="utf-8")
    assert "# 阶段 F：论文图表生成（F0–F3）" in stage_f
    assert "F3_FINAL_FIGURE_PACKAGE_ONLY" in stage_f
    assert "### F4：" not in stage_f
    assert "# 阶段 G：论文写作规划（原 F4）" in stage_g
    assert "STAGE_G_WRITING_G0_ONLY" in stage_g
    assert "claim → evidence → concise boundary" in guide
    assert "defensive_writing_audit.md" in guide
    assert not (REPO / "论文写作指导guideline.md").exists()


def test_stage_g_does_not_reuse_legacy_stage_e_or_f4_output_names():
    stage_g = (REPO / "docs/阶段G规划.md").read_text(encoding="utf-8")
    assert "evidence/stage_g_writing/" in stage_g
    assert "reports/stage_g_writing_<milestone>_<date>/" in stage_g
    assert "不再创建新的 F4 evidence、report 或 archive ref" in stage_g


def test_readme_routes_current_work_to_stage_f_and_stage_g():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Stage F contains F0–F3" in readme
    assert "Stage G is manuscript writing" in readme
    assert "docs/阶段路线图.md" in readme


def test_new_navigation_markdown_links_resolve():
    documents = [
        REPO.parent / "论文1实现审计与后续计划.md",
        REPO / "docs/阶段路线图.md",
        REPO / "docs/阶段F规划.md",
        REPO / "docs/阶段F_F3工作包与产物索引.md",
        REPO / "docs/阶段G规划.md",
        REPO / "docs/阶段G论文写作指导.md",
    ]
    current_targets = {
        "vitnns_rebuild/docs/阶段E规划.md",
        "vitnns_rebuild/docs/阶段F规划.md",
        "vitnns_rebuild/docs/阶段F_F3工作包与产物索引.md",
        "vitnns_rebuild/docs/阶段G规划.md",
        "vitnns_rebuild/docs/阶段G论文写作指导.md",
        "阶段E规划.md", "阶段F规划.md", "阶段F_F3工作包与产物索引.md",
        "阶段G规划.md", "阶段G论文写作指导.md",
    }
    for document in documents:
        links = re.findall(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)", document.read_text(encoding="utf-8"))
        for target in links:
            target = target.strip("<>")
            if target in current_targets:
                assert (document.parent / target).resolve().exists(), (document, target)
