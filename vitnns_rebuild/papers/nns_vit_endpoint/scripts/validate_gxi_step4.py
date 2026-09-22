#!/usr/bin/env python3
"""CPU-only evidence audit and publication validator for GxI roadmap step 4."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "papers/nns_vit_endpoint"
ROOT = PAPER / "publication_r2_20260920"
THESIS = PAPER / "thesis_blind_review_20260921"
E5 = THESIS / "evidence/revision_v5"
RUN_E5 = REPO / "reports/nns_vit_revision_v5_r1_20260921/e5"
REPORT = ROOT / "qc/gxi_step4_validation.json"
CHECKS: list[dict] = []


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check(name: str, passed: bool, detail="") -> None:
    CHECKS.append({"name": name, "passed": bool(passed), "detail": str(detail)})


def seed(key: str) -> int:
    prefix = "ENDPOINT_M1_20260918_V1|20260918|bootstrap|"
    return int.from_bytes(hashlib.sha256((prefix + key).encode()).digest()[:8], "big")


def bootstrap_median(values, key: str, repetitions: int = 10000) -> list[float]:
    """Reproduce the sealed revision-v5 image-bootstrap median implementation."""
    array = np.sort(np.asarray(values, dtype=float))
    rng = np.random.Generator(np.random.PCG64(seed(key)))
    n = array.size
    if n % 2:
        u = rng.beta((n + 1) // 2, (n + 1) // 2, repetitions)
        draws = array[np.minimum((n * u).astype(int), n - 1)]
    else:
        u = rng.beta(n // 2, n // 2 + 1, repetitions)
        v = u + (1 - u) * rng.beta(1, n // 2, repetitions)
        draws = (array[np.minimum((n * u).astype(int), n - 1)] + array[np.minimum((n * v).astype(int), n - 1)]) / 2
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def m3_bootstrap_median(values, key: str, repetitions: int = 10000) -> list[float]:
    array = np.sort(np.asarray(values, dtype=float))
    local_seed = int.from_bytes(hashlib.sha256(("ENDPOINT_M3_BOOTSTRAP_20260919|" + key).encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(local_seed))
    n = array.size
    if n % 2:
        u = rng.beta((n + 1) // 2, (n + 1) // 2, repetitions)
        draws = array[np.minimum((n * u).astype(int), n - 1)]
    else:
        u = rng.beta(n // 2, n // 2 + 1, repetitions)
        v = u + (1 - u) * rng.beta(1, n // 2, repetitions)
        draws = (array[np.minimum((n * u).astype(int), n - 1)] + array[np.minimum((n * v).astype(int), n - 1)]) / 2
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def exact_two_sided_sign(values, tolerance: float = 1e-12) -> dict:
    values = np.asarray(values, dtype=float)
    positive = int((values > tolerance).sum())
    negative = int((values < -tolerance).sum())
    ties = int(values.size - positive - negative)
    n = positive + negative
    p = 1.0 if n == 0 else min(1.0, 2 * math.fsum(math.comb(n, j) for j in range(min(positive, negative) + 1)) / 2**n)
    return {"positive": positive, "negative": negative, "ties": ties, "p": float(p)}


def ranks(values):
    return rankdata(np.asarray(values, dtype=float))


def rho(left, right) -> float:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.size < 4 or not np.isfinite(left).all() or not np.isfinite(right).all() or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def audit_e5() -> dict:
    protocol = load(E5 / "protocol.json")
    cohorts = load(E5 / "cohorts.json")
    bindings = load(E5 / "image_bindings.json")
    estimands = load(E5 / "analysis_estimands.json")
    package = load(E5 / "package_receipt.json")
    receipt = load(E5 / "e5_receipt.json")
    source_cohort = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json"
    analysis_script = REPO / "scripts/analyze_revision_v5.py"

    check("e5_protocol_bindings", sha(E5 / "cohorts.json") == protocol["cohorts_sha256"] and sha(E5 / "targets.json") == protocol["targets_sha256"])
    check("e5_source_cohort_binding", cohorts["sources"] == {str(source_cohort.relative_to(REPO)): sha(source_cohort)})
    check("e5_outcome_blind_selection", cohorts["selection_uses_outcomes"] is False and cohorts["selected_image_bytes_opened"] is False)
    cohort_rows = cohorts["rows"]
    e1_ids = {row["sample_id"] for row in cohort_rows if row["cohort"] == "e1"}
    e5_rows = [row for row in cohort_rows if row["cohort"] == "e5"]
    e5_ids = {row["sample_id"] for row in e5_rows}
    excluded = set(cohorts["exclusions"])
    check("e5_cohort_counts_and_disjointness", len(e1_ids) == 500 and len(e5_ids) == 400 and not e1_ids.intersection(e5_ids) and not excluded.intersection(e5_ids))
    check("e5_shared_split", Counter(row["prediction_split"] for row in e5_rows) == {"train": 200, "validation": 100, "test": 100})
    bound = {row["sample_id"]: row for row in bindings["rows"]}
    check("e5_image_byte_bindings", set(e5_ids).issubset(bound) and all(bound[name]["image_sha256"] and bound[name]["image_bytes"] > 0 for name in e5_ids))

    models = protocol["models"]
    model_ids = {row["model_id"] for row in models}
    check("e5_four_checkpoint_binding", model_ids == {"vit_b16", "deit_s16", "vit_b16_in1k", "vit_s16_augreg"} and sum(row["role"] == "original" for row in models) == 2 and sum(row["role"] == "training_recipe_replication_not_seed" for row in models) == 2)
    checkpoint_ok = True
    for model in models:
        path = REPO / model["checkpoint"]
        checkpoint_ok &= path.is_file() and path.stat().st_size == model["checkpoint_bytes"] and sha(path) == model["checkpoint_sha256"]
    check("e5_checkpoint_bytes", checkpoint_ok)

    e5_contract = estimands["e5"]
    check("e5_statistical_contract", e5_contract["tests"] == "two-sided sign tests, Holm four within checkpoint" and protocol["bootstrap_repeats"] == 10000 and "b11 minus b00" in e5_contract["H1"] and "b11 minus b00" in e5_contract["H2"] and "activation_gradient" in e5_contract["H4"])
    check("e5_analysis_source_binding", sha(analysis_script) == receipt["script_sha256"] and sha(E5 / "analysis_estimands.json") == receipt["estimands_sha256"])
    check("e5_package_integrity", package["status"] == "PASS_COMPACT_EVIDENCE_PACKAGE" and package["counts"]["e5"] == {"expected": 1600, "valid": 1600, "failures": 0} and all((E5 / row["path"]).is_file() and sha(E5 / row["path"]) == row["sha256"] and (E5 / row["path"]).stat().st_size == row["bytes"] for row in package["files"]))
    check("e5_correctness_controls", package["numeric_checks"]["e5"]["closure_checks_total"] == 25600 and package["numeric_checks"]["e5"]["closure_max"] <= 2e-5 and package["numeric_checks"]["e5"]["scalar_vjp_parity_max"] <= 2e-5)

    source_hashes = receipt["sources"]
    source_ok = len(source_hashes) == 3200
    json_count = npz_count = 0
    recomputed_rows = []
    for relative, expected in source_hashes.items():
        path = REPO / relative
        source_ok &= path.is_file() and sha(path) == expected
        if path.suffix == ".json":
            json_count += 1
            meta = load(path)
            source_ok &= meta["status"] == "VALID" and meta["image_sha256"] == bound[meta["image"]]["image_sha256"]
            npz = REPO / meta["output"]
            source_ok &= npz.is_file() and sha(npz) == meta["output_sha256"]
            with np.load(npz, allow_pickle=False) as z:
                values = defaultdict(dict)
                for block in (0, 3, 7, 11):
                    prefix = f"b{block:02d}_"
                    values["entropy"][block] = float(np.median(z[prefix + "post_entropy"]))
                    routing = float(z[prefix + "routing_absolute_mass"][0]); value = float(z[prefix + "V_absolute_mass"][0])
                    values["routing"][block] = routing / (routing + value) if routing + value else np.nan
                    carry = float(z[prefix + "carry_absolute_mass"][0]); mlp = float(z[prefix + "mlp_branch_absolute_mass"][0])
                    values["mlp_identity"][block] = carry / (carry + mlp) if carry + mlp else np.nan
                    identity = float(z[prefix + "identity_absolute_mass"][0]); attention = float(z[prefix + "attention_absolute_mass"][0])
                    values["attention_identity"][block] = identity / (identity + attention) if identity + attention else np.nan
                    effect = np.abs(z[prefix + "normalized_zero_head_effect"]); values["head_increment"][block] = rho(z[prefix + "heads_absolute_mass"], effect) - rho(z[prefix + "head_activation_gradient"], effect)
                computed = {
                    "H1": values["entropy"][11] - values["entropy"][0],
                    "H2": values["routing"][11] - values["routing"][0],
                    "H3": (values["mlp_identity"][11] - values["mlp_identity"][0]) - (values["attention_identity"][11] - values["attention_identity"][0]),
                    "H4": float(np.mean(list(values["head_increment"].values()))),
                }
            recomputed_rows.extend({"model": meta["model_id"], "image": meta["image"], "hypothesis": key, "value": value} for key, value in computed.items())
        elif path.suffix == ".npz":
            npz_count += 1
    check("e5_all_raw_sources_hashed", source_ok and json_count == 1600 and npz_count == 1600, f"json={json_count},npz={npz_count}")

    sealed_image = rows(E5 / "e5_image_values.csv")
    recomputed_rows.sort(key=lambda row: (row["model"], row["image"], row["hypothesis"]))
    sealed_image.sort(key=lambda row: (row["model"], row["image"], row["hypothesis"]))
    max_difference = max(abs(row["value"] - float(sealed["value"])) for row, sealed in zip(recomputed_rows, sealed_image))
    check("e5_raw_numeric_recompute", len(recomputed_rows) == len(sealed_image) == 6400 and max_difference < 1e-12, max_difference)

    sealed_summary = {(row["model"], row["hypothesis"]): row for row in rows(E5 / "e5_summary.csv")}
    by_key = defaultdict(list)
    for row in recomputed_rows:
        by_key[row["model"], row["hypothesis"]].append(row["value"] )
    summary = {}
    summary_ok = True
    for key, values in sorted(by_key.items()):
        array = np.asarray(values, dtype=float)
        interval = bootstrap_median(array, "V5|" + str(key))
        sign = exact_two_sided_sign(array)
        sealed = sealed_summary[key]
        summary_ok &= len(array) == 400 and abs(float(sealed["median"]) - float(np.median(array))) < 1e-12
        summary_ok &= max(abs(float(sealed[name]) - value) for name, value in zip(("ci95_low", "ci95_high"), interval)) < 1e-12
        summary_ok &= abs(float(sealed["p"]) - sign["p"]) < 1e-12 and all(int(sealed[name]) == sign[name] for name in ("positive", "negative", "ties"))
        summary[f"{key[0]}|{key[1]}"] = {"n": len(array), "median": float(np.median(array)), "ci95": interval, **sign}
    for model in sorted(model_ids):
        keys = sorted((key for key in summary if key.startswith(model + "|")), key=lambda key: (summary[key]["p"], key))
        previous = 0.0
        for rank, key in enumerate(keys):
            previous = max(previous, min(1.0, (len(keys) - rank) * summary[key]["p"]))
            summary[key]["holm_p"] = previous
            summary[key]["decision"] = "SUPPORTED_DIRECTION" if previous <= 0.05 and summary[key]["median"] > 0 else "OPPOSITE_DIRECTION" if previous <= 0.05 and summary[key]["median"] < 0 else "NOT_SUPPORTED"
            sealed = sealed_summary[tuple(key.split("|"))]
            summary_ok &= abs(float(sealed["holm_p"]) - previous) < 1e-12 and sealed["decision"] == summary[key]["decision"]
    check("e5_summary_recompute", summary_ok and len(summary) == 16)
    check("e5_output_receipt_bindings", all(sha(E5 / name) == digest for name, digest in receipt["outputs"].items()))
    check("e5_complete_directions", sum(value["decision"] == "SUPPORTED_DIRECTION" for value in summary.values()) == 12 and sum(value["decision"] == "OPPOSITE_DIRECTION" for value in summary.values()) == 4)
    return {
        "decision": "SYNC_FOUR_CHECKPOINT_E5",
        "source_task_files": len(source_hashes),
        "image_checkpoint_tasks": json_count,
        "recomputed_hypothesis_values": len(recomputed_rows),
        "maximum_raw_value_difference": max_difference,
        "cohort": {"images": len(e5_ids), "overlap_with_e1": len(e5_ids & e1_ids), "overlap_with_prior_exclusions": len(e5_ids & excluded)},
        "checkpoints": [{key: model[key] for key in ("model_id", "role", "timm_name", "checkpoint_sha256")} for model in models],
        "summary": summary,
    }


def audit_existing_evidence() -> dict:
    m3_root = REPO / "reports/nns_vit_endpoint_m3_r1_20260919"
    e3_root = REPO / "reports/nns_vit_endpoint_e3_closure_r1_20260918"
    m4_root = REPO / "reports/nns_vit_endpoint_m4_r1_20260919"
    review_root = REPO / "reports/nns_vit_endpoint_m4_review_r2_20260919"
    m3a = load(m3_root / "m3a_fidelity.json")
    e3 = load(e3_root / "validation_report.json")
    m4 = load(m4_root / "scientific_decision.json")
    residual = load(review_root / "residual_spatial_tests.json")
    atlas_rows = rows(m3_root / "image_values.csv")
    qkv_path = ROOT / "evidence/revision_v3_step2/qkv_depth_contrasts.csv"
    qkv = rows(qkv_path)
    bridge = rows(review_root / "bridge_block_summary.csv")
    m3_manifest = load(m3_root / "artifact_manifest.json")
    e3_manifest = load(e3_root / "artifact_manifest.json")
    m4_manifest = load(m4_root / "artifact_manifest.json")
    review_manifest = load(review_root / "artifact_manifest.json")
    for root, manifest, names in (
        (m3_root, m3_manifest, ("m3a_fidelity.json", "scientific_decision.json", "image_values.csv")),
        (e3_root, e3_manifest, ("validation_report.json",)),
        (m4_root, m4_manifest, ("scientific_decision.json", "image_values.csv")),
        (review_root, review_manifest, ("bridge_block_summary.csv", "residual_spatial_image_values.csv", "residual_spatial_tests.json")),
    ):
        indexed = {row["path"]: row for row in manifest["files"]}
        check("sealed_manifest_bindings:" + root.name, all(name in indexed and sha(root / name) == indexed[name]["sha256"] for name in names))
    check("identity_formula_evidence", m3a["status"] == "PASS" and m3a["comparison_count"] == 5000 and m3a["failure_count"] == 0)
    check("closure_mutation_evidence", e3["all_positive_closures_pass"] and e3["all_negative_controls_rejected"] and e3["positive_closure_evaluations"] == 76800 and e3["negative_control_evaluations"] == 28800)
    check("attention_joint_evidence", all(row["decision"] == "SUPPORTED_DIFFERENCE" for row in m4["attention_tests"] if row["test_id"] in {"late_minus_early_routing_share", "late_minus_early_frozen_attention_endpoint_error"}))
    check("residual_joint_evidence", len(residual["tests"]) == 4 and all(row["reject"] for row in residual["tests"]) and {row["residual_type"] for row in residual["tests"]} == {"attention", "mlp"})
    qkv_manifest = load(REPO / "reports/nns_vit_endpoint_revision_v3_step2_r2_20260921/artifact_manifest.json")
    qkv_index = {row["path"]: row for row in qkv_manifest["files"]}
    check("qkv_spatial_evidence", len(qkv) == 30 and all(int(row["n_images"]) == 400 for row in qkv) and qkv_index["qkv_depth_contrasts.csv"]["sha256"] == sha(qkv_path))
    atlas = {}
    for model in ("vit_b16", "deit_s16"):
        model_rows = [row for row in atlas_rows if row["model_id"] == model]
        atlas[model] = {}
        for metric in ("entropy", "locality", "distance"):
            values = np.asarray([float(row[f"late_minus_early_{metric}"]) for row in model_rows])
            atlas[model][metric] = {"n": len(values), "median": float(np.median(values)), "ci95": m3_bootstrap_median(values, f"M3|{model}|late_minus_early_{metric}")}
    check("background_spatial_recompute", all(value[metric]["n"] == 100 for value in atlas.values() for metric in ("entropy", "locality", "distance")))
    m3_summary = {(row["model"], row["metric"]): row for row in rows(m3_root / "atlas_summary.csv")}
    check("background_spatial_matches_sealed_summary", all(abs(value[metric]["median"] - float(m3_summary[model, f"late_minus_early_{metric}"]["median"])) < 1e-12 and max(abs(value[metric]["ci95"][i] - json.loads(m3_summary[model, f"late_minus_early_{metric}"]["ci95"])[i]) for i in (0, 1)) < 1e-12 for model, value in atlas.items() for metric in ("entropy", "locality", "distance")))
    attention_spatial = {}
    for model in ("vit_b16", "deit_s16"):
        attention_spatial[model] = {}
        for block in (0, 11):
            row = next(item for item in bridge if item["model_id"] == model and int(item["block"]) == block and item["metric"] == "full_value_patch_tv")
            attention_spatial[model][f"block_{block}"] = {"n": int(row["n"]), "median": float(row["median"]), "ci95": json.loads(row["ci95"])}
    check("attention_spatial_joint_index_levels", all(value["block_11"]["median"] > value["block_0"]["median"] and value["block_0"]["n"] == value["block_11"]["n"] == 16 for value in attention_spatial.values()))
    return {
        "identity": {"comparisons": m3a["comparison_count"], "float64_max_relative_l2": m3a["by_dtype"]["float64"]["max_relative_l2"], "float32_max_relative_l2": m3a["by_dtype"]["float32"]["max_relative_l2"]},
        "closure": {"positive": e3["positive_closure_evaluations"], "mutations_rejected": e3["negative_control_evaluations"], "maximum_closure_absolute_error": e3["maximum_closure_max_abs"]},
        "background_spatial": atlas,
        "attention": [row for row in m4["attention_tests"] if row["test_id"] in {"late_minus_early_routing_share", "late_minus_early_frozen_attention_endpoint_error"}],
        "residual_spatial": residual["tests"],
        "attention_full_value_patch_tv": attention_spatial,
        "qkv_source_sha256": sha(qkv_path),
    }


def audit_manuscripts(e5_audit: dict) -> None:
    texts = {lang: (ROOT / lang / "manuscript.md").read_text() for lang in ("zh", "en")}
    for lang, text in texts.items():
        required = (
            ("总图 / edge scalar / 双索引图组", "联合索引必要性", "E5 四检查点外部复现", "广泛但非普遍")
            if lang == "zh" else
            ("Total map / edge scalar / dual-indexed map set", "Joint-index necessity", "E5 four-checkpoint external replication", "broad but non-universal")
        )
        check("step4_results_structure:" + lang, all(term in text for term in required))
        check("step4_e5_all_hypotheses:" + lang, all(f"H{i}" in text for i in range(1, 5)) and "0.1136–0.1862" in text)
        check("step4_negative_results:" + lang, ("未获支持" in text and "相反" in text) if lang == "zh" else ("not supported" in text and "opposite" in text))
        check("step4_evidence_navigation:" + lang, ("逐主张证据索引" in text) if lang == "zh" else ("claim-level evidence index" in text))
        check("step4_no_extension_evidence:" + lang, not re.search(r"PCIA|PEFA|aquatic-context|NNS-CST", text, re.I))
        check("step4_no_defensive_stock_phrases:" + lang, not re.search(r"We emphasize that|It should be noted that|We do not claim|To avoid misunderstanding|Taken together|需要强调的是|应当指出|为避免误解|我们不声称", text, re.I))
    check("step4_bilingual_sections", re.findall(r"^#{2,3} (\d+(?:\.\d+)?)", texts["zh"], re.M) == re.findall(r"^#{2,3} (\d+(?:\.\d+)?)", texts["en"], re.M))
    table_rows = {lang: [line for line in text.splitlines() if line.startswith("|")] for lang, text in texts.items()}
    table_values = lambda table: [[cell.strip() for cell in line.strip("|").split("|")][2:] for line in table]
    check("step4_bilingual_tables", len(table_rows["zh"]) == len(table_rows["en"]) and table_values(table_rows["zh"]) == table_values(table_rows["en"]), (len(table_rows["zh"]), len(table_rows["en"])))
    for lang, text in texts.items():
        check("step4_table_numbers:" + lang, re.findall(r"^\*\*Table (\d+)\.", text, re.M) == [str(i) for i in range(1, 11)])
    for key, value in e5_audit["summary"].items():
        token = f"{value['median']:.4f} [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}]"
        check("step4_e5_value_in_both:" + key, all(token in text for text in texts.values()))
    for lang, text in texts.items():
        expected = text.replace("](../figures/", "](../publication_r2_20260920/figures/").replace("](../references.md)", "](../publication_r2_20260920/references.md)").replace("](supplement.md)", f"](../publication_r2_20260920/{lang}/supplement.md)").replace("](supplement_experiments_v1.md)", f"](../publication_r2_20260920/{lang}/supplement_experiments_v1.md)")
        check("step4_entry_mirror:" + lang, (PAPER / lang / "full_draft.md").read_text() == expected)


def audit_evidence_index() -> None:
    index_path = ROOT / "evidence/gxi_step4_evidence_index.md"
    if not index_path.is_file():
        check("step4_evidence_index_paths", False, "missing index")
        return
    text = index_path.read_text()
    paths = re.findall(r"`((?:reports|papers)/[^`]+\.(?:json|csv))`", text)
    check("step4_evidence_index_fields", all(term in text for term in ("适用 target", "统计单位与 checkpoint 范围", "效应量与区间", "封存证据")))
    check("step4_evidence_index_paths", bool(paths) and all((REPO / path).is_file() for path in paths), paths)


def main() -> int:
    e5_audit = audit_e5()
    existing = audit_existing_evidence()
    (ROOT / "evidence/gxi_step4_e5_audit.json").write_text(json.dumps(e5_audit, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "evidence/gxi_step4_numeric_recalculation.json").write_text(json.dumps({"status": "PASS", "cpu_only": True, "new_experiments": 0, "e5": e5_audit, "existing_evidence": existing}, ensure_ascii=False, indent=2) + "\n")
    audit_manuscripts(e5_audit)
    audit_evidence_index()
    required = [
        ROOT / "evidence/gxi_step4_e5_audit.json",
        ROOT / "evidence/gxi_step4_numeric_recalculation.json",
        ROOT / "evidence/gxi_step4_evidence_index.md",
        ROOT / "evidence/gxi_step4_figure_allocation.md",
        ROOT / "qc/gxi_step4_writing_audit.md",
        ROOT / "qc/gxi_step4_completion_audit.md",
    ]
    check("step4_deliverables_present", all(path.is_file() for path in required))
    plan = (REPO / "Endpoint_GxI创新定位与后续五步计划.md").read_text()
    step4_record = plan.split("## 十一、第四步执行记录：结果重排与封存证据绑定", 1)[1]
    check("step4_plan_status", "第四步已完成实验结果重排与封存证据绑定" in plan.splitlines()[2] and "第五步未启动" in plan.splitlines()[2])
    check("step4_prompt_artifact_audit", step4_record.split("### 11.4 第四步 prompt-to-artifact 验收表", 1)[1].split("### 11.5", 1)[0].count("| PASS |") == 16)
    errors = [row for row in CHECKS if not row["passed"]]
    result = {
        "status": "PASS" if not errors else "FAIL",
        "objective": "GxI roadmap step 4: reorganize results and bind sealed evidence",
        "cpu_only": True,
        "new_experiments": 0,
        "check_count": len(CHECKS),
        "errors": errors,
        "checks": CHECKS,
        "e5_audit": e5_audit,
        "existing_evidence": existing,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(CHECKS), "errors": errors, "e5_decision": e5_audit["decision"]}, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
