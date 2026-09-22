#!/usr/bin/env python3
"""Create a compact N2-V2 I5 blinded review handoff from I4 assets."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "evidence/n2_v2_i4_20260922"
OUT = ROOT / "evidence/n2_v2_i5_20260922"
ATTEMPT_ID = "N2_V2_I5_20260922"
PER_STRATUM = 30


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def make_review_html(reviewer_id: str, card_ids: list[str], manifest_sha256: str) -> str:
    cards_json = json.dumps(card_ids, separators=(",", ":"))
    reviewer_json = json.dumps(reviewer_id)
    manifest_json = json.dumps(manifest_sha256)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>N2-V2 I5 {reviewer_id}</title>
  <style>
    body {{ margin: 0 auto; max-width: 1050px; background: #242424; color: #f4f4f4; font: 16px/1.4 sans-serif; }}
    img {{ display: block; width: 960px; height: 960px; object-fit: contain; background: #777; }}
    button, select, input {{ font-size: 16px; margin: 8px; }}
    fieldset {{ border: 1px solid #888; margin: 12px 0; }}
    #progress {{ float: right; }}
  </style>
</head>
<body>
  <h1>N2-V2 独立盲审 <span id="progress"></span></h1>
  <p>不要查看 private、来源信息、另一位审查者结果或模型信息；完成前不要讨论。</p>
  <img id="card" alt="">
  <h2 id="cardid"></h2>
  <fieldset>
    <legend>背景语义</legend>
    <select id="label">
      <option value="">请选择</option>
      <option>AQUATIC_CONTEXT</option>
      <option>NON_AQUATIC_CONTEXT</option>
      <option>AMBIGUOUS</option>
    </select>
  </fieldset>
  <fieldset>
    <legend>artifact（任一 true 即拒绝）</legend>
    <label><input type="checkbox" id="visible_non_background_object">显著非背景物体</label><br>
    <label><input type="checkbox" id="donor_removal_or_fill_trace">移除/填充痕迹</label><br>
    <label><input type="checkbox" id="abnormal_compositing_boundary">异常合成边界</label><br>
    <label><input type="checkbox" id="other_arm_identity_cue">其他 arm identity 线索</label>
  </fieldset>
  <button id="prev">上一张</button>
  <button id="save">保存并下一张</button>
  <button id="export">全部完成后导出 JSON</button>
  <script>
    const reviewer = {reviewer_json};
    const manifest = {manifest_json};
    const cards = {cards_json};
    const fields = ["visible_non_background_object", "donor_removal_or_fill_trace", "abnormal_compositing_boundary", "other_arm_identity_cue"];
    let i = 0, started = new Date().toISOString(), answers = {{}};
    function show() {{
      const card = cards[i], answer = answers[card] || {{}};
      document.getElementById("card").src = "cards/" + card + ".png";
      document.getElementById("cardid").textContent = card;
      document.getElementById("label").value = answer.semantic_label || "";
      for (const field of fields) document.getElementById(field).checked = !!answer[field];
      document.getElementById("progress").textContent = (i + 1) + " / " + cards.length;
    }}
    function save() {{
      const card = cards[i], label = document.getElementById("label").value;
      if (!label) {{ alert("请选择语义标签"); return false; }}
      const answer = {{ card_id: card, semantic_label: label }};
      for (const field of fields) answer[field] = document.getElementById(field).checked;
      answers[card] = answer;
      localStorage.setItem(reviewer, JSON.stringify({{ i, started, answers }}));
      return true;
    }}
    document.getElementById("save").onclick = () => {{ if (save() && i < cards.length - 1) {{ i++; show(); }} }};
    document.getElementById("prev").onclick = () => {{ save(); if (i > 0) {{ i--; show(); }} }};
    document.getElementById("export").onclick = () => {{
      if (!save()) return;
      const missing = cards.filter(card => !answers[card]);
      if (missing.length) {{ alert("仍有 " + missing.length + " 张未完成"); return; }}
      const bundle = {{
        schema_version: 1,
        study_version: "N2_V2",
        review_bundle_id: "N2_V2_I5_" + reviewer + "_" + Date.now(),
        reviewer_id: reviewer,
        review_card_manifest_sha256: manifest,
        review_started_utc: started,
        review_completed_utc: new Date().toISOString(),
        independence_declaration: {{
          worked_independently: true,
          discussed_before_seal: false,
          model_outcomes_visible: false,
          source_or_arm_identity_visible: false
        }},
        reviews: cards.map(card => answers[card])
      }};
      const blob = new Blob([JSON.stringify(bundle, null, 2) + "\\n"], {{ type: "application/json" }});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = reviewer + ".json";
      link.click();
    }};
    const old = JSON.parse(localStorage.getItem(reviewer) || "null");
    if (old) {{ i = old.i || 0; started = old.started || started; answers = old.answers || {{}}; }}
    show();
  </script>
</body>
</html>
"""


def select_scenes(records: list[dict]) -> list[dict]:
    selected: list[dict] = []
    for stratum in ("AQUATIC_CANDIDATE_STRATUM", "NON_AQUATIC_CANDIDATE_STRATUM"):
        rows = sorted((row for row in records if row["candidate_stratum"] == stratum), key=lambda row: row["card_id"])
        if len(rows) < PER_STRATUM:
            raise RuntimeError(f"INSUFFICIENT_SCENE_CANDIDATES:{stratum}")
        selected.extend(rows[:PER_STRATUM])
    return sorted(selected, key=lambda row: hashlib.sha256((ATTEMPT_ID + ":" + row["asset_id"]).encode()).hexdigest())


def main() -> int:
    if OUT.exists():
        raise RuntimeError("I5_ROOT_ALREADY_EXISTS")
    OUT.mkdir(parents=True)
    for relative in ("private/recipient_images", "private/recipient_masks"):
        shutil.copytree(SOURCE / relative, OUT / relative)
    shutil.copy2(SOURCE / "private/openimages_mask_archive_bindings.json", OUT / "private/openimages_mask_archive_bindings.json")

    source_scenes = json.loads((SOURCE / "private/scene_candidate_registry.json").read_text())["records"]
    selected = select_scenes(source_scenes)
    scene_records: list[dict] = []
    card_records: list[dict] = []
    for index, row in enumerate(selected, 1):
        new_row = dict(row)
        old_image = SOURCE / row["image_path"]
        image_extension = old_image.suffix.lower() or ".img"
        image_path = OUT / "private/scene_originals" / (row["asset_id"] + image_extension)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_image, image_path)
        card_id = f"N2V2-I5-CARD-{index:03d}"
        card_path = OUT / "review/cards" / f"{card_id}.png"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((960, 960), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (960, 960), (127, 127, 127))
            canvas.paste(rgb, ((960 - rgb.width) // 2, (960 - rgb.height) // 2))
            canvas.save(card_path, format="PNG", compress_level=9)
        new_row.update({
            "card_id": card_id,
            "image_path": str(image_path.relative_to(OUT)),
            "image_sha256": sha(image_path),
            "review_image_path": str(card_path.relative_to(OUT)),
            "review_image_sha256": sha(card_path),
        })
        scene_records.append(new_row)
        card_records.append({"card_id": card_id, "image_path": str(card_path.relative_to(OUT)), "image_sha256": sha(card_path)})

    recipients = json.loads((SOURCE / "private/recipient_candidate_registry.json").read_text())
    recipients["attempt_id"] = ATTEMPT_ID
    write_json(OUT / "private/recipient_candidate_registry.json", recipients)
    write_json(OUT / "private/scene_candidate_registry.json", {
        "schema_version": 1,
        "study_version": "N2_V2",
        "attempt_id": ATTEMPT_ID,
        "status": "COMPACT_CANDIDATES_NOT_ACCEPTED_PENDING_BLIND_REVIEW",
        "selection_rule": "FROM_I4_FULL_FRAME_POOL_SELECT_FIRST_30_PER_STRATUM_BY_I4_CARD_ID_THEN_RERANDOMIZE_BY_SHA256_I5_ATTEMPT_COLON_ASSET_ID",
        "records": scene_records,
    })
    review_manifest = {
        "schema_version": 1,
        "study_version": "N2_V2",
        "attempt_id": ATTEMPT_ID,
        "blinding": {
            "source_id_hidden": True,
            "dataset_class_hidden": True,
            "candidate_stratum_hidden": True,
            "model_information_hidden": True,
            "randomization_key": "SHA256_I5_ATTEMPT_ID_COLON_ASSET_ID_ASCENDING",
            "display_transform": "FULL_FRAME_ASPECT_PRESERVING_MAX_960_GREY_PAD_NO_CROP",
        },
        "records": card_records,
    }
    write_json(OUT / "review/review_card_manifest.json", review_manifest)
    manifest_sha = sha(OUT / "review/review_card_manifest.json")
    template_reviews = [{
        "card_id": row["card_id"],
        "semantic_label": "REPLACE_WITH_AQUATIC_CONTEXT_NON_AQUATIC_CONTEXT_OR_AMBIGUOUS",
        "visible_non_background_object": None,
        "donor_removal_or_fill_trace": None,
        "abnormal_compositing_boundary": None,
        "other_arm_identity_cue": None,
    } for row in card_records]
    card_ids = [row["card_id"] for row in card_records]
    for reviewer in ("reviewer_A", "reviewer_B"):
        write_json(OUT / "review/templates" / f"{reviewer}.json", {
            "schema_version": 1,
            "study_version": "N2_V2",
            "review_bundle_id": "REPLACE_AFTER_COMPLETION",
            "reviewer_id": reviewer,
            "review_card_manifest_sha256": manifest_sha,
            "review_started_utc": "REPLACE_RFC3339",
            "review_completed_utc": "REPLACE_RFC3339",
            "independence_declaration": {
                "worked_independently": True,
                "discussed_before_seal": False,
                "model_outcomes_visible": False,
                "source_or_arm_identity_visible": False,
            },
            "reviews": template_reviews,
        })
        (OUT / "review" / f"{reviewer}.html").write_text(make_review_html(reviewer, card_ids, manifest_sha))

    (OUT / "review/README.md").write_text(
        "# N2-V2 I5 双人独立盲审交接\n\n"
        "审查者 A 只打开 `reviewer_A.html`，审查者 B 只打开 `reviewer_B.html`。不要查看 `private/`、来源信息、另一位审查者的文件或模型信息；完成前不要讨论。浏览器会本地保存进度，60 张全部填写后点击导出 JSON。将两个导出的 JSON 分别交回，不要互看。\n\n"
        "语义标签只能是 `AQUATIC_CONTEXT`、`NON_AQUATIC_CONTEXT`、`AMBIGUOUS`。四个 artifact 字段逐项判断。任一审查者任一 artifact=true 将 union-reject。卡片为完整画面等比缩放并以灰色留边，没有裁切、内容填充或局部编辑。\n"
    )
    receipt = {
        "schema_version": 1,
        "study_version": "N2_V2",
        "stage_id": "N2_V2_INPUT_IDENTIFIABILITY",
        "attempt_id": ATTEMPT_ID,
        "status": "BLOCKED_INDEPENDENT_REVIEWERS",
        "reason": "Compact full-frame blinded cards are ready; two independent human review bundles remain required.",
        "predecessor_attempt_id": "N2_V2_I4_20260922",
        "fresh_root_reason": "Reduce human review load from 102 to 60 cards while preserving at least 24 required assets per candidate stratum plus six reserves.",
        "scene_candidate_count": 60,
        "candidate_stratum_counts": {"AQUATIC_CANDIDATE_STRATUM": 30, "NON_AQUATIC_CANDIDATE_STRATUM": 30},
        "minimum_required_per_context": 24,
        "reserve_per_context_before_review": 6,
        "recipient_candidate_count": 12,
        "review_card_count": 60,
        "accepted_scene_asset_count": 0,
        "sealed_review_bundle_count": 0,
        "blind_review_status": "NOT_RUN_AWAITING_HUMANS",
        "matching_status": "NOT_RUN_PRECEDING_GATE",
        "identifiability_freeze": "NOT_ISSUED",
        "behavior_gate": "NOT_AUTHORIZED",
        "model_access_count": 0,
        "checkpoint_access_count": 0,
        "gpu_use_count": 0,
    }
    write_json(OUT / "preparation_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
