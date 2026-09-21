#!/usr/bin/env python3
"""Fail-closed structural and byte-binding audit for Endpoint-PCIA-PEFA Step 1."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "vit_pcia"

DOCS = {
    "literature": HERE / "literature_matrix.md",
    "contract": HERE / "method_contract.md",
    "assets": HERE / "asset_inventory.md",
    "decision": HERE / "step1_decision.md",
}

ASSET_HASHES = {
    "vitnns_rebuild/src/vitnns/endpoint_nns.py": "143226aba89d9b975b34ba014ee7dbba1adaf651c3cdaae0d43f4fd08c20cd4e",
    "vitnns_rebuild/src/vitnns/endpoint_vit.py": "66a32cb7a236dcca25c916eaddc775afae516367f00ea0da91aa6585eb1ca4c7",
    "vitnns_rebuild/tests/test_endpoint_nns.py": "f6f7660da30234a43210954465375ae5721d8249ff4cb427951be9ac136a6fd8",
    "vitnns_rebuild/tests/test_endpoint_vit.py": "0b6ff2c7139aa04b1ea7f36e788e544795d70791cb8af892afdd25e734805f9d",
    "vitnns_rebuild/tests/test_endpoint_ep1_queue.py": "89fdf610d2625fc8f7dd201754fed32e2db8660fcf980d9559ad023cbf691935",
    "vitnns_rebuild/assets/models/vit_base_patch16_224_weights.semantic_candidate.pth": "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d",
    "vitnns_rebuild/assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth": "cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f",
    "vitnns_rebuild/assets/imagenet/ILSVRC2012_img_val.tar": "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0",
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json": "4d1e88d946c11af3fce9ca05c139007dc0991768ea224e2067ca33b337a7e0f2",
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json": "c2f8a5075d1bae28ae122ad9190731a675cc21774f3eb0f5b6a37c9019c2b626",
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/implementation_contract.json": "dde628560d1d453e11bff3b09c48a2321db5354301928d7bcde7066a89821bef",
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/resource_contract.json": "c38d30d91e822c424fbc8e18a35595680133966414478d4c4605e7f074da9476",
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/source_bindings.json": "adf261134c0cf72035a12f3c66be5aebbfb446a102232a4beacb03205d69732b",
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/literature_registry.json": "c2551b5204de027693ec0bc24308d5010cbc2a1d52a9178c7d52b4a6486353ec",
    "vitnns_rebuild/reports/nns_vit_endpoint_ep2_r1_20260918/task_manifest.json": "3621a06a7dbd6fc921181ebcdb64eec27f0eeafeb491e09e9b5a41a1eb6c46e9",
    "vitnns_rebuild/reports/nns_vit_endpoint_e5_spatial_r1_20260918/task_manifest.json": "5ed2cdcfd9dee2fa5e206706a0864b91aaa2e77c7e7331a3cb10fb5ee7634d28",
    "vitnns_rebuild/docs/NNS_ViT_Endpoint_Scanner_EP0协议冻结_20260918.md": "4c133a5a5b0c3dfcf86dbbf1d1bb908d5bf78aaea89375feaf6944be9279d7b9",
    "vitnns_rebuild/reports/nns_vit_endpoint_ep1_r11_20260918/terminal_receipt.json": "cc81f3ac26943ff742cc4230f30fbc15f027f9f5057db71abedb5944649528bb",
    "vitnns_rebuild/reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json": "c855d2302a90b1b419b0e040a6ca13a3ceeae27bbe42c4caf5605b74af5c1b1a",
    "vitnns_rebuild/reports/nns_vit_endpoint_e3_closure_r2_20260918/artifact_manifest.json": "9868bbbe0b96e7508b771fd9ff3e580ce0ac4b12538a355f6c9ba689ae1c410f",
    "vitnns_rebuild/reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json": "6261e12a6837ef8e931ca32bf3faab2689c2e4aa8b26d9fc7e20b947b2e7d082",
    "vitnns_rebuild/reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json": "675cf4c6bec7dd681497f57bbd31b638dece7a5dfe0abc2b99e921bdbfbfe8ac",
    "vitnns_rebuild/reports/nns_vit_endpoint_e6_release_r3_20260918/validation_report.json": "53925d742b6a070a1bb4dff253237b64299a015a9fb131199ea60d997f4a515f",
    "vitnns_rebuild/reports/nns_vit_endpoint_e6_release_r3_20260918/terminal_receipt.json": "50342a81e2acfa98e155671858d3fa675a7d8eca8868b94ab7890e9bd5287f9e",
    "创新思想v1.md": "4252a0fa9c26229e875a36a72ee540d867550b0bc23da1dc8cfdc7f6734166d2",
    "vit_pcia/执行规划.md": "03ae681d9ddcadd7ab07b13592e92463775cfa55f81a914df68164689132809d",
}

ASSET_SIZES = {
    "vitnns_rebuild/src/vitnns/endpoint_nns.py": 10399,
    "vitnns_rebuild/src/vitnns/endpoint_vit.py": 26224,
    "vitnns_rebuild/tests/test_endpoint_nns.py": 9814,
    "vitnns_rebuild/tests/test_endpoint_vit.py": 14657,
    "vitnns_rebuild/tests/test_endpoint_ep1_queue.py": 4706,
    "vitnns_rebuild/assets/models/vit_base_patch16_224_weights.semantic_candidate.pth": 346329934,
    "vitnns_rebuild/assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth": 88251143,
    "vitnns_rebuild/assets/imagenet/ILSVRC2012_img_val.tar": 6744924160,
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json": 446993,
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json": 185629,
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/implementation_contract.json": 2127,
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/resource_contract.json": 3727,
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/source_bindings.json": 5564,
    "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/literature_registry.json": 33785,
    "vitnns_rebuild/reports/nns_vit_endpoint_ep2_r1_20260918/task_manifest.json": 120147,
    "vitnns_rebuild/reports/nns_vit_endpoint_e5_spatial_r1_20260918/task_manifest.json": 930755,
    "vitnns_rebuild/docs/NNS_ViT_Endpoint_Scanner_EP0协议冻结_20260918.md": 15079,
    "vitnns_rebuild/reports/nns_vit_endpoint_ep1_r11_20260918/terminal_receipt.json": 459,
    "vitnns_rebuild/reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json": 6840,
    "vitnns_rebuild/reports/nns_vit_endpoint_e3_closure_r2_20260918/artifact_manifest.json": 267,
    "vitnns_rebuild/reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json": 1518,
    "vitnns_rebuild/reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json": 682,
    "vitnns_rebuild/reports/nns_vit_endpoint_e6_release_r3_20260918/validation_report.json": 2403,
    "vitnns_rebuild/reports/nns_vit_endpoint_e6_release_r3_20260918/terminal_receipt.json": 550,
    "创新思想v1.md": 9529,
    "vit_pcia/执行规划.md": 14925,
}

METHOD_TERMS = (
    "IG", "Conductance", "Neuron IG", "Internal Influence",
    "Integrated Hessians", "Higher-Order Feature Attribution",
    "Metagame", "DeepLIFT", "LRP", "Shapley Flow",
    "Path-wise", "ACDC", "EAP", "EAP-IG", "AtP*",
    "Circuit Tracing", "DePass", "LIG", "IG-Lens",
    "DPA", "Unpack", "GAF", "DAG-SHAP",
    "DAVE", "HETA", "Sparse Inter-Layer Dependencies",
    "HiLRP", "ResLRP",
    "PA-LRP", "implementation-invariance audit", "GradSkip",
    "CA-LIG", "DAP", "iERF / SRD / ICAT",
    "Contribution Weights", "CAAP",
    "LibraGrad / LeGrad / GMAR",
)
FIELD_TERMS = (
    "被解释对象", "索引结构", "守恒终点", "真实使用端口",
    "合法切面", "二阶项", "路径/参考依赖", "实现不变性",
    "Q/K/V–head–residual–MLP", "因果边界",
)
CONTRACT_TERMS = (
    "标量目标 $F$", "参考 $b$ 与路径 $\\gamma$", "变化根变量",
    "输入坐标", "端口 ID", "合法切面", "积分规则与数值合同",
    "闭合残差", "模型与数据冻结", "随机性与可复现性",
    "资源边界", "失败类型与不可变性", "CPU 解析 fixture 数据",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_payloads(manifest_rel: str, expected_count: int, expected_digest: str) -> None:
    manifest = json.loads((ROOT / manifest_rel).read_text(encoding="utf-8"))
    bindings: dict[str, str] = {}
    for row in manifest.get("tasks", manifest.get("rows", [])):
        payload = row.get("payload", row)
        path, digest = payload.get("cache_path"), payload.get("image_sha256")
        if path and digest:
            if path in bindings:
                require(bindings[path] == digest, f"conflicting payload hash: {path}")
            bindings[path] = digest
    require(len(bindings) == expected_count, f"payload count {len(bindings)} != {expected_count}")
    for relative, expected in bindings.items():
        path = ROOT / "vitnns_rebuild" / relative
        require(path.is_file(), f"missing payload: {relative}")
        require(sha256(path) == expected, f"payload hash drift: {relative}")
    canonical = "".join(
        f"{path}\0{bindings[path]}\n" for path in sorted(bindings)
    ).encode("utf-8")
    require(hashlib.sha256(canonical).hexdigest() == expected_digest, "payload binding drift")


def main() -> None:
    for name, path in DOCS.items():
        require(path.is_file() and path.stat().st_size > 500, f"missing/incomplete {name}")
    literature = DOCS["literature"].read_text(encoding="utf-8")
    contract = DOCS["contract"].read_text(encoding="utf-8")
    assets = DOCS["assets"].read_text(encoding="utf-8")
    decision = DOCS["decision"].read_text(encoding="utf-8")

    for term in METHOD_TERMS:
        require(term in literature, f"literature family missing: {term}")
    matrix_block = literature.split("## 3. 逐字段矩阵", 1)[1].split("## 4.", 1)[0]
    matrix_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in matrix_block.splitlines()
        if line.startswith("| ") and not line.startswith("|---")
    ]
    require(len(matrix_rows) == len(METHOD_TERMS) + 1,
            f"method matrix row count drift: {len(matrix_rows) - 1}")
    require(all(len(row) == 12 for row in matrix_rows), "method matrix is not 12 columns")
    for term in FIELD_TERMS:
        require(term in literature, f"comparison field missing: {term}")
    for term in ("### 4.1 Conductance", "### 4.2 Integrated Hessians",
                 "### 5.1 $T$", "### 5.2 $R$", "### 5.3 $\\Omega$",
                 "不能证明不存在", "可证伪主张"):
        require(term in literature, f"literature audit missing: {term}")
    for arxiv_id in ("2602.06613", "2604.13258", "2607.11990",
                     "2609.01282", "2609.17152", "2506.02138",
                     "2510.18810", "2607.10365", "2602.16608",
                     "2604.18094", "2605.00474", "2606.07604",
                     "2603.13652", "2608.02396", "2411.16760",
                     "2404.03214", "2504.19414"):
        require(arxiv_id in literature, f"cutoff-neighbor source missing: {arxiv_id}")
    source_labels = [int(value) for value in re.findall(r"^- \[L(\d+)\]", literature, re.MULTILINE)]
    require(source_labels == list(range(1, 47)), f"source label drift: {source_labels}")
    for term in CONTRACT_TERMS:
        require(term in contract, f"contract field missing: {term}")
    for token in (
        "GO_STEP2_COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE",
        "STEP2_REQUIRES_EXPLICIT_USER_REQUEST",
        "FAIL_SECOND_ORDER_GRAPH_TRUNCATED", "FAIL_CUT_NOT_LEGAL",
        "64 节点", "512 节点", "LINEAR3", "MULTIINPUT3",
        "RELU_SAFE", "RELU_HIT", "0.48782485366828776",
    ):
        require(token in contract, f"frozen contract token missing: {token}")
    for token in (
        "STEP1_STATUS=PASS_CONTRACT_FROZEN",
        "NOVELTY_POSITION=COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE",
        "PCIA_FOUNDATIONAL_NOVELTY=REJECTED_CONDUCTANCE_EQUIVALENT_SPECIALIZATION",
        "T_FOUNDATIONAL_NOVELTY=REJECTED_PATH_IG_OF_ENDPOINT_COMPONENT",
        "R_OPERATOR_NOVELTY=REJECTED_WEIGHTED_HESSIAN_FAMILY",
        "R_PORT_RESOLVED_IDENTITY_COVERAGE=NOT_FOUND_IN_FROZEN_SEARCH",
        "OMEGA_OPERATOR_NOVELTY=REJECTED_STANDARD_ANTISYMMETRIC_DERIVATIVE",
        "OMEGA_PORT_CUT_ATTRIBUTION_CONTRACT_COVERAGE=NOT_FOUND_IN_FROZEN_SEARCH",
        "NO_GO_FOUNDATIONAL_NOVELTY=false", "STEP2_ELIGIBLE=true",
        "NEXT_AUTHORIZATION=STEP2_REQUIRES_EXPLICIT_USER_REQUEST",
        "STEP2_STARTED=false",
    ):
        require(token in decision, f"decision token missing: {token}")
    audit_block = decision.split("## Prompt-to-artifact 审计", 1)[1].split("## 验证命令", 1)[0]
    audit_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in audit_block.splitlines()
        if line.startswith("| ") and not line.startswith("|---")
    ]
    require(len(audit_rows) >= 35, f"prompt audit row count too small: {len(audit_rows) - 1}")
    require(all(len(row) == 3 for row in audit_rows), "prompt audit is not 3 columns")
    require(all(row[2] in {"状态", "PASS", "PASS_WITH_SCOPE_LIMIT"} for row in audit_rows),
            "prompt audit contains an unpassed row")
    require("OLD_RESULTS_RELABELED_AS_PEFA=false" in assets, "relabel guard missing")
    require("AUTHORIZED_PROVENANCE_ADVANCE" in assets, "source drift explanation missing")

    require(ASSET_HASHES.keys() == ASSET_SIZES.keys(), "asset hash/size key mismatch")
    for relative, expected in ASSET_HASHES.items():
        path = ROOT / relative
        require(path.is_file(), f"missing bound asset: {relative}")
        require(path.stat().st_size == ASSET_SIZES[relative], f"bound asset size drift: {relative}")
        require(sha256(path) == expected, f"bound asset drift: {relative}")

    cohort = json.loads((ROOT / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json").read_text())
    rows = cohort["rows"]
    counts = {split: sum(row["split"] == split for row in rows) for split in ("discovery", "calibration", "evaluation")}
    require(counts == {"discovery": 64, "calibration": 64, "evaluation": 400}, f"cohort drift: {counts}")
    require(len({row["sample_id"] for row in rows}) == 528, "cohort duplicate sample")

    verify_payloads(
        "vitnns_rebuild/reports/nns_vit_endpoint_ep2_r1_20260918/task_manifest.json",
        64, "2be0ba262f2ac569f9548d4c2ecaaaaba82eeb4c341bb78d0b1398f4a56996e8",
    )
    verify_payloads(
        "vitnns_rebuild/reports/nns_vit_endpoint_e5_spatial_r1_20260918/task_manifest.json",
        400, "20a1b32d01cf84ee9a33368aadfa1355e33c32f2c739aee629f958c6340594fa",
    )

    for commit in (
        "4d3a0a4bab5d72e9ef0766ce2d035d4489d3e792",
        "db342d6015b971f9015e0302030f4547ddfa4200",
        "7d930595d5d3956383c7ead821c08ca3206aa597",
        "e00a38c5cdc455b9797dd71f751c3fd40f27563c",
        "fa670bdc86f1b6344fd193f9c4117e6f55edb9ac",
    ):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT
        )
        require(result.returncode == 0, f"source commit is not an ancestor: {commit}")

    forbidden_step2 = (HERE / "mvp_validation.json", HERE / "src", HERE / "tests")
    require(not any(path.exists() for path in forbidden_step2), "Step 2 artifact already exists")
    require(not (HERE / "__pycache__").exists(), "generated Python cache present")

    result = {
        "status": "PASS_STEP1_CONTRACT_AND_COLLISION_AUDIT",
        "documents": {name: sha256(path) for name, path in DOCS.items()},
        "literature_families_checked": len(METHOD_TERMS),
        "comparison_fields_checked": len(FIELD_TERMS),
        "bound_assets_checked": len(ASSET_HASHES),
        "cohort_counts": counts,
        "payloads_rehashed": 464,
        "novelty_position": "COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE",
        "next_authorization": "STEP2_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
