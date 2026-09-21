#!/usr/bin/env python3
"""Fail-closed artifact and coverage audit for Endpoint-PCIA-PEFA Step 3."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPECTED_FIXTURES = {"LINEAR3", "BILINEAR2", "SATURATING3", "FANOUT2", "RESIDUAL2", "MULTIINPUT3"}
EXPECTED_SYMBOLIC = {"cut_gradient_closure", "t_integrand_decomposition", "cut_omega_zero_sum", "hessian_symmetry"}
EXPECTED_PATH = {"output_delta", "gradient_x", "endpoint", "port_d_x", "port_q_x", "ig", "a", "t", "r", "omega_x"}
EXPECTED_REPRESENTATIONS = {
    "path_reparameterization", "path_selection", "reference_point",
    "input_coordinates", "within_port_reparameterization",
    "parallel_refinement", "cross_graph_rewrite",
}
EXPECTED_POINT_CHECKS = {"value", "gradient", "port_d", "port_jac_d", "port_q", "port_jac_q", "omega"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    receipt = json.loads((HERE / "step3_symbolic_validation.json").read_text(encoding="utf-8"))
    predecessor = json.loads((HERE / "mvp_validation.json").read_text(encoding="utf-8"))
    theory = (HERE / "theory_endpoint_pcia_pefa.md").read_text(encoding="utf-8")
    decision = (HERE / "step3_decision.md").read_text(encoding="utf-8")
    runner_source = (HERE / "run_step3_symbolic.py").read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)

    require(predecessor["status"] == "PASS_STEP2_CPU_FLOAT64_REFERENCE", "Step 2 predecessor is not PASS")
    require(receipt["predecessor"]["mvp_validation_sha256"] == sha256(HERE / "mvp_validation.json"), "Step 2 receipt binding drift")
    require(receipt["status"] == "PASS_STEP3_THEORY_AND_SYMBOLIC_CONSISTENCY", "Step 3 receipt is not PASS")
    require(receipt["scope"] == "THEORY_ONLY_STEP4_TO_STEP6_NOT_AUTHORIZED", "Step 3 scope drift")
    require(receipt["environment"]["device"] == "cpu_symbolic", "non-CPU symbolic result")
    require(receipt["contract"]["smoothness"] == "C2_OPEN_NEIGHBORHOOD_OF_PATH", "smoothness contract drift")
    require(receipt["contract"]["graph"] == "FIXED_FINITE_DAG", "graph contract drift")
    require(receipt["contract"]["path"] == "PIECEWISE_C1", "path contract drift")
    require(receipt["no_go_generality"] is False, "NO_GO_GENERALITY was issued")
    require(receipt["scientific_effect_claims"] == "UNTESTED", "empirical claim boundary drift")

    imports = set()
    for node in ast.walk(runner_tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    require(not (imports & {"torch", "timm", "PIL", "cv2"}), f"forbidden model/image import: {imports}")
    build_node = next(
        node for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_fixture"
    )
    build_source = ast.get_source_segment(runner_source, build_node) or ""
    require("analytic_fixtures" not in build_source, "symbolic fixture constructor reuses Step 2 analytic implementation")

    require(set(receipt["fixtures"]) == EXPECTED_FIXTURES, "fixture coverage drift")
    require(set(receipt["coverage"]["symbolic_identity_families"]) == EXPECTED_SYMBOLIC, "symbolic identity coverage drift")
    require(set(receipt["coverage"]["path_quantities"]) == EXPECTED_PATH, "path quantity coverage drift")
    require(set(receipt["coverage"]["representation_cases"]) == EXPECTED_REPRESENTATIONS, "representation coverage drift")
    require(receipt["coverage"]["point_fractions"] == [0.0, 0.37, 1.0], "point grid drift")

    symbolic_count = point_count = path_count = 0
    for fixture_id in EXPECTED_FIXTURES:
        row = receipt["fixtures"][fixture_id]
        require(row["passed"] is True, f"fixture failed: {fixture_id}")
        require(set(row["symbolic_identities"]) == EXPECTED_SYMBOLIC, f"symbolic fields drift: {fixture_id}")
        require(all(row["symbolic_identities"].values()), f"symbolic identity false: {fixture_id}")
        symbolic_count += len(row["symbolic_identities"])
        require([item["t"] for item in row["point_checks"]] == [0.0, 0.37, 1.0], f"point grid incomplete: {fixture_id}")
        for point in row["point_checks"]:
            require(point["passed"] is True, f"point comparison failed: {fixture_id}/{point['t']}")
            require(set(point["checks"]) == EXPECTED_POINT_CHECKS, f"point fields drift: {fixture_id}")
            require(all(item["passed"] for item in point["checks"].values()), f"point metric failed: {fixture_id}")
            point_count += 1
        require(set(row["path_checks"]) == EXPECTED_PATH, f"path fields drift: {fixture_id}")
        require(all(item["passed"] for item in row["path_checks"].values()), f"path comparison failed: {fixture_id}")
        path_count += len(row["path_checks"])

    require(set(receipt["representation_checks"]) == EXPECTED_REPRESENTATIONS, "representation result drift")
    require(all(item["passed"] for item in receipt["representation_checks"].values()), "representation check failed")
    require(receipt["all_fixtures_passed"] is True, "fixture aggregate false")
    require(receipt["all_representation_checks_passed"] is True, "representation aggregate false")

    required_theory_sections = (
        "## 1. 设定与充分条件", "## 2. 基础引理", "## 3. 闭合与差分定理",
        "## 4. 等价边界", "## 5. 表征条件与不变性", "## 6. 退化情形与反例",
        "## 7. 适用边界、范围判定与可证伪性", "## 8. 与步骤 2 解析 fixture 的数值观察",
    )
    require(all(section in theory for section in required_theory_sections), "theory chain section missing")
    for token in (
        "### 定理 1：逐端口、逐 Endpoint 坐标的 PEFA 闭合",
        "### 定理 2：直接项与累计二阶项分解",
        "### 定理 3：Endpoint–IG 逐坐标差分恒等式",
        "### 定理 4：完整切面的路由曲率零和",
        "### 5.1 路径重参数化", "### 5.2 路径选择与参考点",
        "### 5.3 输入坐标变换", "### 5.4 端口内可逆重参数化",
        "### 5.5 并行细化", "### 5.6 跨图重写反例",
        "### 命题 5：仿射图退化", "### 命题 6：一维反对称退化",
        "### 命题 7：完整模型梯度场退化", "### 反例 3：$R$ 和 $\Omega$ 不是天然因果效应",
        "UNTESTED_EMPIRICAL_CLAIM", "NO_GO_GENERALITY=false",
    ):
        require(token in theory, f"theory requirement missing: {token}")
    require(theory.index("## 8. 与步骤 2 解析 fixture 的数值观察") > theory.index("## 7. 适用边界、范围判定与可证伪性"), "numeric observations not separated after theory")

    for token in (
        "STEP3_STATUS=PASS_THEORY_AND_SYMBOLIC_CONSISTENCY",
        "CORE_IDENTITIES_PROVED=4/4", "REPRESENTATION_CASES_RESOLVED=7/7",
        "SYMBOLIC_FIXTURES_PASS=6/6", "CONJECTURES_USED_AS_CONCLUSIONS=0",
        "NO_GO_GENERALITY=false", "SCIENTIFIC_EFFECT_CLAIMS=UNTESTED",
        "STEP4_REQUIRES_EXPLICIT_USER_REQUEST", "STEP4_TO_STEP6_NOT_STARTED=true",
    ):
        require(token in decision, f"decision token missing: {token}")

    audit_block = decision.split("## Prompt-to-artifact 审计", 1)[1].split("## 复现与验证", 1)[0]
    audit_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in audit_block.splitlines()
        if line.startswith("| ") and not line.startswith("|---")
    ]
    require(len(audit_rows) >= 30, f"prompt audit row count too small: {len(audit_rows) - 1}")
    require(all(len(row) == 3 for row in audit_rows), "prompt audit is not three columns")
    require(all(row[2] in {"状态", "PASS"} for row in audit_rows), "prompt audit contains non-PASS row")

    for relative, expected in receipt["source_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"source drift after Step 3: {relative}")

    print(json.dumps({
        "status": "PASS_STEP3_ARTIFACT_AND_COVERAGE_AUDIT",
        "fixtures_verified": len(EXPECTED_FIXTURES),
        "symbolic_identities_verified": symbolic_count,
        "point_comparisons_verified": point_count,
        "path_quantity_comparisons_verified": path_count,
        "representation_cases_verified": len(EXPECTED_REPRESENTATIONS),
        "prompt_audit_requirements_verified": len(audit_rows) - 1,
        "step3_symbolic_validation_sha256": sha256(HERE / "step3_symbolic_validation.json"),
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
