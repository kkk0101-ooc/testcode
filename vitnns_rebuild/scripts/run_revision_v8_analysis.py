#!/usr/bin/env python3
"""Freeze and run revision-v8 static-graph and CPU-only sensitivity analyses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "reports/nns_vit_endpoint_revision_v8_r1_20260923"
REGISTRY_ROOT = REPO / "manifests/nns_vit_endpoint/revision_v8_20260923"
PLAN = REPO / "修改意见/修改意见v8_盲审前结构与敏感性补强工作规程.md"
SOURCE_ROOT = REPO / "manifests/nns_vit_endpoint/ep0_r1_20260918"
SOURCE_TARGETS = SOURCE_ROOT / "target_registry.json"
SOURCE_INTERVENTIONS = SOURCE_ROOT / "intervention_registry.json"
SOURCE_MODELS = SOURCE_ROOT / "model_registry.json"
E3_ROOT = REPO / "reports/nns_vit_endpoint_e3_closure_r1_20260918"
M3_ROOT = REPO / "reports/nns_vit_endpoint_m3_r1_20260919"
E5_ROOT = REPO / "reports/nns_vit_revision_v5_r1_20260921/e5"
V7_ROOT = REPO / "reports/nns_vit_endpoint_revision_v7_r4_20260923"
BASELINE_ROOT = REPO / "papers/nns_vit_endpoint/publication_r2_20260920/evidence/supplement_v1"
SCRIPT = Path(__file__).resolve()
VALIDATOR = REPO / "scripts/validate_revision_v8_analysis.py"
TEST_FILE = REPO / "tests/test_revision_v8.py"
BLOCKS = tuple(range(12))
H4_BLOCKS = (0, 3, 7, 11)
TRIMS = (0, 1, 5, 10)
MASS_QUANTILES = (1, 5, 10, 25, 50, 75)
N_BOOTSTRAP = 10_000
EPS = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and array.size > 0 and np.isfinite(array).all(), "INVALID_RANK_INPUT")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float], minimum: int = 4) -> tuple[float | None, str]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        return None, "NA_SHAPE"
    if x.size < minimum:
        return None, "NA_INSUFFICIENT"
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, "NA_NONFINITE"
    rx = average_ranks(x); ry = average_ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return (None, "NA_CONSTANT") if denominator == 0 else (float(np.dot(rx, ry) / denominator), "")


def bootstrap_median(values: Sequence[float], key: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and array.size > 0 and np.isfinite(array).all(), "INVALID_BOOTSTRAP_INPUT")
    seed = int.from_bytes(hashlib.sha256(("REVISION_V8_BOOTSTRAP_R1|" + key).encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.median(array[rng.integers(0, len(array), size=(N_BOOTSTRAP, len(array)))], axis=1)
    return float(np.median(array)), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def sign_counts(values: Sequence[float]) -> dict[str, int]:
    array = np.asarray(values, dtype=np.float64)
    return {"positive": int(np.sum(array > EPS)), "negative": int(np.sum(array < -EPS)), "ties": int(np.sum(np.abs(array) <= EPS))}


def source_files() -> list[Path]:
    fixed = [SOURCE_TARGETS, SOURCE_INTERVENTIONS, SOURCE_MODELS,
             E3_ROOT / "terminal_receipt.json", E3_ROOT / "validation_report.json",
             M3_ROOT / "task_manifest.json", M3_ROOT / "image_values.csv",
             V7_ROOT / "terminal_receipt.json", V7_ROOT / "final_validation_report.json",
             BASELINE_ROOT / "baseline_analysis.json", BASELINE_ROOT / "baseline_image_values.csv",
             BASELINE_ROOT / "baseline_repeat_cohort_values.csv", BASELINE_ROOT / "baseline_repeat_image_values.csv"]
    tasks = json.loads((M3_ROOT / "task_manifest.json").read_text())["tasks"]
    for task in tasks:
        attempt = M3_ROOT / "results/attempts" / task["task_id"] / "attempt-01.json"
        result = json.loads(attempt.read_text())["result"]
        fixed.extend((attempt, REPO / result["metrics_path"]))
    for metadata in sorted(E5_ROOT.glob("*.json")):
        output = REPO / json.loads(metadata.read_text())["output"]
        fixed.extend((metadata, output))
    unique = {path.resolve(): path for path in fixed}
    return [unique[key] for key in sorted(unique, key=lambda item: unique[item].relative_to(REPO).as_posix())]


def freeze() -> None:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_ONLY_REQUIRES_EMPTY_CUDA_VISIBLE_DEVICES")
    require(not ROOT.exists() and not REGISTRY_ROOT.exists(), "FRESH_V8_ROOTS_REQUIRED")
    v7 = json.loads((V7_ROOT / "terminal_receipt.json").read_text())
    require(v7["status"] == "REVISION_V7_REANALYSIS_PASS_WITH_ALL_STRATA_REPORTED", "V7_SOURCE_NOT_PASS")
    ROOT.mkdir(parents=True); REGISTRY_ROOT.mkdir(parents=True)
    bindings = [record(path) for path in source_files()]
    write_json(ROOT / "source_bindings.json", {"schema_version": 1, "status": "BOUND_BEFORE_V8_RESULTS", "files": bindings, "files_digest": canonical_digest(bindings)})
    protocol = {
        "schema_version": 1, "namespace": "NNS_VIT_REVISION_V8_R1_20260923", "status": "FROZEN_BEFORE_V8_RESULTS",
        "execution": {"cpu_only": True, "cuda_visible_devices": "", "jpeg_decode": False, "model_load": False, "checkpoint_read": False, "gpu_inference": False},
        "immutability": {"revision_v7_roots": "read_only", "original_h4": "unchanged", "all_family_decision": "NOT_SUPPORTED_UNCHANGED", "random_region_baseline": "validate_existing_S9_only"},
        "structural_graph": {"models": ["vit_b16", "deit_s16"], "blocks": list(BLOCKS), "target_cut_pairs": "all scalar targets x all 12 blocks x 8 cut declarations", "criterion": "declared edges lie in target ancestor subgraph and every path crosses exactly once", "branch_scope": "branch partitions are not relabeled as global cuts"},
        "registry_split": ["scalar_target_registry", "decomposition_registry", "intervention_registry"],
        "entropy_sensitivity": {"quality": "unit_map_total_L1_absolute_mass", "threshold_scope": "within_model_and_block", "mass_quantiles_percent": list(MASS_QUANTILES), "trim_lowest_percent": list(TRIMS), "quartiles": ["Q1", "Q2", "Q3", "Q4"], "continuous": "within_image_block_Spearman_log1p_mass_vs_entropy_then_mean_12_blocks", "statistical_unit": "image", "identity": "post_hoc_sensitivity"},
        "h4_sensitivity": {"original_path_score": "sum_i_abs(L_i)", "matched_path_score": "abs(sum_i_L_i)", "activation_gradient_score": "abs(dot(z,grad_z_s))", "effect": "abs(clean_minus_normalized_zero_head_replacement)", "blocks": list(H4_BLOCKS), "aggregation": "within_block_head_Spearman_difference_then_mean_blocks", "statistical_unit": "image", "identity": "post_hoc_sensitivity"},
        "statistics": {"bootstrap_replicates": N_BOOTSTRAP, "interval": "pointwise_95_percentile_median", "new_confirmatory_tests": 0},
        "live_plan_at_freeze": record(PLAN),
        "sources_sha256": sha256(ROOT / "source_bindings.json"),
        "code": {"analysis": record(SCRIPT), "validator": record(VALIDATOR), "tests": record(TEST_FILE)},
    }
    write_json(ROOT / "protocol.json", protocol)
    write_json(ROOT / "stage0_receipt.json", {"schema_version": 1, "status": "REVISION_V8_STAGE0_PASS_AND_ANALYSIS_FROZEN", "protocol": record(ROOT / "protocol.json"), "source_bindings": record(ROOT / "source_bindings.json"), "new_results_accessed": False, "next_authorization": "REVISION_V8_CPU_STATIC_AND_ARRAY_ANALYSIS"})
    print(json.dumps({"status": "REVISION_V8_STAGE0_PASS_AND_ANALYSIS_FROZEN", "source_files": len(bindings)}))


def verify_frozen() -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_ONLY_REQUIRES_EMPTY_CUDA_VISIBLE_DEVICES")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    require(protocol["status"] == "FROZEN_BEFORE_V8_RESULTS", "V8_NOT_FROZEN")
    for key, path in (("analysis", SCRIPT), ("validator", VALIDATOR), ("tests", TEST_FILE)):
        require(protocol["code"][key]["sha256"] == sha256(path), "FROZEN_CODE_DRIFT:" + key)
    source = json.loads((ROOT / "source_bindings.json").read_text())
    for item in source["files"]:
        path = REPO / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], "SOURCE_DRIFT:" + item["path"])
    return protocol


def classify_target(row: Mapping[str, Any]) -> str:
    kind = row["kind"]; module = row["module"]
    if kind in {"ground_truth_logit", "frozen_clean_top1_logit"}: return "class_logit"
    if kind == "head_channel": return "attention_head_coordinate"
    if kind == "token_channel" and module == "norm": return "final_cls_coordinate"
    if kind == "token_channel": return "post_block_token_coordinate"
    if module == "patch_embed": return "patch_embedding_aggregate"
    if kind == "patch_channel_energy": return "post_block_patch_energy"
    if kind == "patch_channel_sum" and module.startswith("blocks.") and module.count(".") == 1: return "post_block_patch_sum"
    if kind == "patch_channel_sum" and module.startswith("blocks."): return "module_patch_aggregate"
    raise ValueError("UNCLASSIFIED_TARGET:" + json.dumps(row, sort_keys=True))


def add_edge(edges: list[tuple[str, str, str]], source: str, target: str, edge_id: str) -> None:
    edges.append((source, target, edge_id))


def build_graph(heads: int) -> tuple[list[tuple[str, str, str]], dict[tuple[int, str], set[str]]]:
    edges: list[tuple[str, str, str]] = []
    add_edge(edges, "input", "patch_embed", "input_to_patch_embed")
    previous = "patch_embed"; cuts: dict[tuple[int, str], set[str]] = {}
    for block in BLOCKS:
        prefix = f"b{block:02d}"; block_input = f"{prefix}.input"
        add_edge(edges, previous, block_input, f"{prefix}.entry")
        add_edge(edges, block_input, f"{prefix}.norm1", f"{prefix}.to_norm1")
        head_edges=set(); qkv_edges=set(); rv_edges=set()
        for head in range(heads):
            for letter in ("q", "k", "v"):
                edge=f"{prefix}.qkv.{letter}.{head}"; add_edge(edges, f"{prefix}.norm1", f"{prefix}.{letter}{head}", edge); qkv_edges.add(edge)
            add_edge(edges, f"{prefix}.q{head}", f"{prefix}.route{head}", f"{prefix}.q_to_route.{head}")
            add_edge(edges, f"{prefix}.k{head}", f"{prefix}.route{head}", f"{prefix}.k_to_route.{head}")
            route_edge=f"{prefix}.routing.{head}"; value_edge=f"{prefix}.value.{head}"
            add_edge(edges, f"{prefix}.route{head}", f"{prefix}.mix{head}", route_edge)
            add_edge(edges, f"{prefix}.v{head}", f"{prefix}.mix{head}", value_edge); rv_edges.update((route_edge,value_edge))
            add_edge(edges, f"{prefix}.mix{head}", f"{prefix}.head{head}", f"{prefix}.mix_to_head.{head}")
            head_edge=f"{prefix}.head_use.{head}"; add_edge(edges, f"{prefix}.head{head}", f"{prefix}.slice{head}", head_edge); head_edges.add(head_edge)
            add_edge(edges, f"{prefix}.slice{head}", f"{prefix}.concat", f"{prefix}.concat.{head}")
        add_edge(edges, f"{prefix}.concat", f"{prefix}.attn_proj", f"{prefix}.projection")
        add_edge(edges, f"{prefix}.attn_proj", f"{prefix}.attn_branch", f"{prefix}.drop_path1")
        identity=f"{prefix}.residual1.identity"; attention=f"{prefix}.residual1.attention"
        add_edge(edges, block_input, f"{prefix}.r1", identity); add_edge(edges, f"{prefix}.attn_branch", f"{prefix}.r1", attention)
        add_edge(edges, f"{prefix}.r1", f"{prefix}.norm2", f"{prefix}.to_norm2")
        add_edge(edges, f"{prefix}.norm2", f"{prefix}.mlp_fc1", f"{prefix}.fc1")
        add_edge(edges, f"{prefix}.mlp_fc1", f"{prefix}.mlp_act", f"{prefix}.act")
        add_edge(edges, f"{prefix}.mlp_act", f"{prefix}.mlp_fc2", f"{prefix}.fc2")
        add_edge(edges, f"{prefix}.mlp_fc2", f"{prefix}.mlp_branch", f"{prefix}.drop_path2")
        carry=f"{prefix}.residual2.carry"; mlp=f"{prefix}.residual2.mlp"
        add_edge(edges, f"{prefix}.r1", f"{prefix}.output", carry); add_edge(edges, f"{prefix}.mlp_branch", f"{prefix}.output", mlp)
        cuts[block,"heads_branch"]=head_edges; cuts[block,"heads_total"]=head_edges|{identity}
        cuts[block,"qkv_branch"]=qkv_edges; cuts[block,"qkv_total"]=qkv_edges|{identity}
        cuts[block,"routing_value_branch"]=rv_edges; cuts[block,"routing_value_total"]=rv_edges|{identity}
        cuts[block,"residual1"]={identity,attention}; cuts[block,"residual2"]={carry,mlp}
        previous=f"{prefix}.output"
    add_edge(edges, previous, "final_norm", "final_norm"); add_edge(edges, "final_norm", "classifier", "classifier")
    return edges,cuts


def target_node(row: Mapping[str, Any]) -> str:
    kind=row["kind"]; module=row["module"]
    if kind in {"ground_truth_logit","frozen_clean_top1_logit"}: return "classifier"
    if kind=="token_channel" and module=="norm": return "final_norm"
    if module=="patch_embed": return "patch_embed"
    parts=module.split("."); require(len(parts)>=2 and parts[0]=="blocks", "UNKNOWN_TARGET_MODULE:"+module)
    block=int(parts[1]); prefix=f"b{block:02d}"
    if kind=="head_channel": return f"{prefix}.slice{int(row['head'])}"
    if len(parts)==2: return f"{prefix}.output"
    suffix=".".join(parts[2:]); mapping={"norm1":"norm1","attn.proj":"attn_proj","norm2":"norm2","mlp.fc1":"mlp_fc1","mlp.act":"mlp_act","mlp.fc2":"mlp_fc2"}
    require(suffix in mapping, "UNKNOWN_TARGET_LOCATION:"+module)
    return f"{prefix}.{mapping[suffix]}"


def reachable(edges: Sequence[tuple[str,str,str]], source: str, target: str) -> bool:
    outgoing: dict[str,list[str]]=defaultdict(list)
    for left,right,_ in edges:outgoing[left].append(right)
    seen={source};queue=deque([source])
    while queue:
        node=queue.popleft()
        if node==target:return True
        for nxt in outgoing[node]:
            if nxt not in seen:seen.add(nxt);queue.append(nxt)
    return False


def path_cut_counts(edges: Sequence[tuple[str,str,str]], target: str, cut: set[str], start: str="input") -> tuple[set[int], bool]:
    nodes={start,target};outgoing:dict[str,list[tuple[str,str]]]=defaultdict(list);indegree=Counter()
    edge_rows={edge:(source,dest) for source,dest,edge in edges}
    for source,dest,edge in edges:nodes.update((source,dest));outgoing[source].append((dest,edge));indegree[dest]+=1;indegree.setdefault(source,0)
    queue=deque(sorted(node for node in nodes if indegree[node]==0));order=[]
    while queue:
        node=queue.popleft();order.append(node)
        for dest,_ in outgoing[node]:
            indegree[dest]-=1
            if indegree[dest]==0:queue.append(dest)
    require(len(order)==len(nodes),"ARCHITECTURE_GRAPH_NOT_DAG")
    counts:dict[str,set[int]]=defaultdict(set);counts[start]={0}
    for node in order:
        for dest,edge in outgoing[node]:
            for value in counts[node]:counts[dest].add(min(2,value+(edge in cut)))
    all_edges_in_ancestor=all(edge in edge_rows and reachable(edges,edge_rows[edge][1],target) for edge in cut)
    return counts[target],all_edges_in_ancestor


class GraphAnalyzer:
    """Cache target-ancestor subgraphs and cut counts for one frozen architecture."""
    def __init__(self, edges: Sequence[tuple[str,str,str]], start: str="input") -> None:
        self.edges=tuple(edges);self.start=start;self.outgoing=defaultdict(list);self.incoming=defaultdict(list);indegree=Counter();nodes={start}
        for source,dest,edge in self.edges:
            nodes.update((source,dest));self.outgoing[source].append((dest,edge));self.incoming[dest].append((source,edge));indegree[dest]+=1;indegree.setdefault(source,0)
        queue=deque(sorted(node for node in nodes if indegree[node]==0));order=[]
        while queue:
            node=queue.popleft();order.append(node)
            for dest,_ in self.outgoing[node]:
                indegree[dest]-=1
                if indegree[dest]==0:queue.append(dest)
        require(len(order)==len(nodes),"ARCHITECTURE_GRAPH_NOT_DAG");self.order=tuple(order);self._ancestor_cache={};self._count_cache={}

    def ancestors(self,target: str) -> set[str]:
        if target not in self._ancestor_cache:
            found={target};queue=deque([target])
            while queue:
                node=queue.popleft()
                for source,_ in self.incoming[node]:
                    if source not in found:found.add(source);queue.append(source)
            self._ancestor_cache[target]=found
        return self._ancestor_cache[target]

    def cut_counts(self,target: str,cut: set[str]) -> tuple[set[int],bool]:
        key=(target,tuple(sorted(cut)))
        if key in self._count_cache:return self._count_cache[key]
        ancestors=self.ancestors(target);counts=defaultdict(set);counts[self.start]={0}
        ancestor_edges=set()
        for node in self.order:
            if node not in ancestors:continue
            for dest,edge in self.outgoing[node]:
                if dest not in ancestors:continue
                ancestor_edges.add(edge)
                for value in counts[node]:counts[dest].add(min(2,value+(edge in cut)))
        result=(counts[target],cut<=ancestor_edges);self._count_cache[key]=result;return result


def build_registries_and_graph_audit() -> dict[str, Any]:
    original=json.loads(SOURCE_TARGETS.read_text());models=json.loads(SOURCE_MODELS.read_text());interventions=json.loads(SOURCE_INTERVENTIONS.read_text())
    model_heads={row["model_id"]:int(row["heads"]) for row in models["models"]}
    target_counts={model:Counter(classify_target(row) for row in rows) for model,rows in original["targets_by_model"].items()}
    for model,rows in original["targets_by_model"].items():require(len(rows)==original["target_count_by_model"][model],"TARGET_COUNT_DRIFT")
    scalar={"schema_version":1,"registry_type":"scalar_target_readout","status":"DERIVED_EXACTLY_FROM_EP0_TARGET_ROWS","source":record(SOURCE_TARGETS),"meaning":"fixed scalar readouts per image; not exhaustive neurons, decomposition labels, or interventions","target_count_by_model":original["target_count_by_model"],"target_class_count_by_model":{m:dict(sorted(c.items())) for m,c in target_counts.items()},"targets_by_model":original["targets_by_model"]}
    decomposition={"schema_version":1,"registry_type":"local_differential_decomposition","status":"DERIVED_FROM_EP0_CUT_CONTRACT","source":record(SOURCE_TARGETS),"object_classes":{"native_consumer_use":["head-to-concat","Q projection slot","K projection slot","V projection slot","residual identity input","residual branch input"],"analytic_differential_partition":["Q-induced","K-induced","routing-induced","value-induced"],"aggregate_score":["map absolute mass","signed sum","spatial entropy"]},"cuts":original["cuts"],"cut_blocks":original["cut_blocks"],"cut_scopes":original["cut_scopes"],"hook_edges":original["hook_edges"],"terminology":{"path_map":"local differential term indexed by a concrete consumer use, not a unique causal path","branch_partition":"reconstructs a branch contribution only","global_cut":"includes every bypass and reconstructs the same-target total differential"}}
    intervention={"schema_version":1,"registry_type":"finite_replacement_intervention","status":"DERIVED_EXACTLY_FROM_EP0_INTERVENTION_CONTRACT","source":record(SOURCE_INTERVENTIONS),"target":interventions["target"],"references":interventions["references"],"operations":interventions["operations"],"component_experiment":interventions["C4a"],"regional_experiment":interventions["C4b"],"boundary":"finite replacement variables and effects; not scalar target readouts or local differential partitions"}
    write_json(REGISTRY_ROOT/"scalar_target_registry.json",scalar);write_json(REGISTRY_ROOT/"decomposition_registry.json",decomposition);write_json(REGISTRY_ROOT/"intervention_registry.json",intervention)
    matrix=[];summary=Counter();graph_stats={};branch_ids={"heads_branch","qkv_branch","routing_value_branch"}
    cut_order=("heads_branch","heads_total","qkv_branch","routing_value_branch","qkv_total","routing_value_total","residual1","residual2")
    for model,targets in sorted(original["targets_by_model"].items()):
        edges,cuts=build_graph(model_heads[model]);analyzer=GraphAnalyzer(edges);graph_stats[model]={"heads":model_heads[model],"nodes":len({n for e in edges for n in e[:2]}),"edges":len(edges)}
        for target in targets:
            node=target_node(target);target_class=classify_target(target)
            for block in BLOCKS:
                branch_endpoint=f"b{block:02d}.attn_branch"
                for cut_id in cut_order:
                    counts,all_ancestor=analyzer.cut_counts(node,cuts[block,cut_id])
                    if cut_id in branch_ids:
                        branch_counts,branch_ancestor=analyzer.cut_counts(branch_endpoint,cuts[block,cut_id])
                        if counts=={1} and all_ancestor:status="VALID_FULL_TARGET_CUT_WITHIN_ATTENTION_BRANCH"
                        elif branch_counts=={1} and branch_ancestor and reachable(edges,branch_endpoint,node):status="VALID_ATTENTION_BRANCH_PARTITION_ONLY"
                        else:status="NOT_APPLICABLE_TARGET_UPSTREAM_OR_OUTSIDE_BRANCH"
                    else:
                        status="VALID_FULL_TARGET_CUT" if counts=={1} and all_ancestor else "NOT_APPLICABLE_DECLARED_RECONSTRUCTION"
                    row={"model":model,"target_id":target["target_id"],"target_class":target_class,"target_module":target["module"],"target_location":target["location"],"target_node":node,"cut_block":block,"cut_id":cut_id,"path_cut_counts":";".join(map(str,sorted(counts))),"all_cut_edges_in_target_ancestor":all_ancestor,"status":status}
                    matrix.append(row);summary[model,target_class,cut_id,status]+=1
    fields=["model","target_id","target_class","target_module","target_location","target_node","cut_block","cut_id","path_cut_counts","all_cut_edges_in_target_ancestor","status"]
    write_csv(ROOT/"target_cut_applicability.csv",matrix,fields)
    summary_rows=[{"model":k[0],"target_class":k[1],"cut_id":k[2],"status":k[3],"pair_count":v} for k,v in sorted(summary.items())]
    write_csv(ROOT/"target_cut_summary.csv",summary_rows,["model","target_class","cut_id","status","pair_count"])
    report={"schema_version":1,"status":"PASS_ALL_TARGET_CUT_PAIRS_CLASSIFIED","architecture":"registered timm 0.9.16 standard pre-LN Block/Attention unfused eval","graph_stats":graph_stats,"scalar_target_counts":original["target_count_by_model"],"target_class_counts":{m:dict(sorted(c.items())) for m,c in target_counts.items()},"target_cut_pair_count":len(matrix),"status_counts":dict(sorted(Counter(row["status"] for row in matrix).items())),"proof":{"full_cut":"all declared cut edges lie in the target ancestor graph and all input-to-target paths cross exactly one cut edge","branch_partition":"all partition edges lie in the attention-branch ancestor graph, all input-to-branch paths cross exactly once, and the branch lies upstream of target","method":"topological dynamic programming over possible cut-edge counts, capped at two","extra_branches":"outside the registered architecture contract are not certified"},"registries":[record(REGISTRY_ROOT/name) for name in ("scalar_target_registry.json","decomposition_registry.json","intervention_registry.json")]}
    write_json(ROOT/"architecture_graph_audit.json",report);return report


def m3_records() -> list[dict[str,Any]]:
    records=[]
    for task in json.loads((M3_ROOT/"task_manifest.json").read_text())["tasks"]:
        attempt=json.loads((M3_ROOT/"results/attempts"/task["task_id"]/"attempt-01.json").read_text())["result"]
        with np.load(REPO/attempt["metrics_path"],allow_pickle=False) as arrays:
            records.append({"model":attempt["model_id"],"image":attempt["sample_id"],"block":arrays["block"].astype(np.int16),"mass":arrays["absolute_mass"].astype(np.float64),"entropy":arrays["normalized_spatial_entropy"].astype(np.float64),"zero":arrays["zero_gradient"].astype(bool)})
    return records


def entropy_analysis() -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]],float]:
    records=m3_records();thresholds={};threshold_rows=[]
    for model in sorted({r["model"] for r in records}):
        for block in BLOCKS:
            values=np.concatenate([r["mass"][(r["block"]==block)&~r["zero"]&np.isfinite(r["mass"])] for r in records if r["model"]==model])
            quant=np.quantile(values,np.asarray(MASS_QUANTILES)/100.0,method="linear")
            thresholds[model,block]={q:float(v) for q,v in zip(MASS_QUANTILES,quant)}
            threshold_rows.append({"model":model,"block":block,"n_units":len(values),**{f"p{q:02d}":thresholds[model,block][q] for q in MASS_QUANTILES}})
    old={(row["model_id"],row["sample_id"]):float(row["late_minus_early_entropy"]) for row in csv.DictReader((M3_ROOT/"image_values.csv").open())}
    image_rows=[];max_error=0.0
    for r in records:
        valid=~r["zero"]&np.isfinite(r["mass"])&np.isfinite(r["entropy"]);values={};retained={}
        for trim in TRIMS:
            mask=valid.copy()
            if trim:
                for block in BLOCKS:mask[(r["block"]==block)&(r["mass"]<=thresholds[r["model"],block][trim])]=False
            early=r["entropy"][mask&np.isin(r["block"],range(4))];late=r["entropy"][mask&np.isin(r["block"],range(8,12))]
            require(len(early)>0 and len(late)>0,"ENTROPY_TRIM_EMPTY")
            values[f"trim_p{trim:02d}_late_minus_early"]=float(np.median(late)-np.median(early));retained[f"trim_p{trim:02d}_retained_units"]=int(mask.sum())
        for quartile,(low,high) in enumerate(((None,25),(25,50),(50,75),(75,None)),1):
            mask=valid.copy()
            for block in BLOCKS:
                bm=r["block"]==block
                if low is not None:mask[bm&(r["mass"]<=thresholds[r["model"],block][low])]=False
                if high is not None:mask[bm&(r["mass"]>thresholds[r["model"],block][high])]=False
            early=r["entropy"][mask&np.isin(r["block"],range(4))];late=r["entropy"][mask&np.isin(r["block"],range(8,12))]
            require(len(early)>0 and len(late)>0,"ENTROPY_QUARTILE_EMPTY")
            values[f"q{quartile}_late_minus_early"]=float(np.median(late)-np.median(early));retained[f"q{quartile}_units"]=int(mask.sum())
        rhos=[]
        for block in BLOCKS:
            mask=valid&(r["block"]==block);rho_value,reason=spearman(np.log1p(r["mass"][mask]),r["entropy"][mask]);require(rho_value is not None,"ENTROPY_RHO_"+reason);rhos.append(rho_value)
        values["mean_within_block_mass_entropy_spearman"]=float(np.mean(rhos))
        values["late_minus_early_log1p_mass"]=float(np.median(np.log1p(r["mass"][valid&np.isin(r["block"],range(8,12))]))-np.median(np.log1p(r["mass"][valid&np.isin(r["block"],range(4))])))
        max_error=max(max_error,abs(values["trim_p00_late_minus_early"]-old[r["model"],r["image"]]))
        image_rows.append({"model":r["model"],"image":r["image"],**values,**retained})
    summary=[];metrics=[f"trim_p{x:02d}_late_minus_early" for x in TRIMS]+[f"q{x}_late_minus_early" for x in range(1,5)]+["mean_within_block_mass_entropy_spearman","late_minus_early_log1p_mass"]
    for model in sorted({r["model"] for r in image_rows}):
        for metric in metrics:
            data=[float(r[metric]) for r in image_rows if r["model"]==model];median,low,high=bootstrap_median(data,f"entropy|{model}|{metric}")
            summary.append({"model":model,"metric":metric,"n_images":len(data),"median":median,"ci95_low":low,"ci95_high":high,**sign_counts(data),"analysis_identity":"post_hoc_mass_sensitivity"})
    return threshold_rows,image_rows,summary,max_error


def h4_analysis() -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]],float]:
    old={(r["model"],r["image"]):float(r["h4_image_mean"]) for r in csv.DictReader((V7_ROOT/"h4_recomputed_image_values.csv").open())}
    block_rows=[];image_values=defaultdict(lambda:defaultdict(list));max_error=0.0
    for metadata in sorted(E5_ROOT.glob("*.json")):
        item=json.loads(metadata.read_text());require(item["status"]=="VALID","E5_NOT_VALID")
        with np.load(REPO/item["output"],allow_pickle=False) as arrays:
            for block in H4_BLOCKS:
                l1=arrays[f"b{block:02d}_heads_absolute_mass"].astype(np.float64)
                signed=np.abs(arrays[f"b{block:02d}_heads_signed_sum"].astype(np.float64))
                ag=np.abs(arrays[f"b{block:02d}_head_activation_gradient"].astype(np.float64))
                effect=np.abs(arrays[f"b{block:02d}_normalized_zero_head_effect"].astype(np.float64))
                l1_rho,r1=spearman(l1,effect);signed_rho,r2=spearman(signed,effect);ag_rho,r3=spearman(ag,effect);agreement,r4=spearman(signed,ag)
                reason=r1 or r2 or r3 or r4;require(not reason,"H4_MATCHED_"+reason)
                original=float(l1_rho-ag_rho);matched=float(signed_rho-ag_rho)
                block_rows.append({"model":item["model_id"],"image":item["image"],"block":block,"path_l1_rho":l1_rho,"path_signed_sum_abs_rho":signed_rho,"activation_gradient_rho":ag_rho,"original_difference":original,"matched_difference":matched,"matched_path_vs_activation_gradient_rho":agreement,"heads":len(l1)})
                image_values[item["model_id"],item["image"]]["original"].append(original);image_values[item["model_id"],item["image"]]["matched"].append(matched);image_values[item["model_id"],item["image"]]["agreement"].append(float(agreement))
    image_rows=[]
    for (model,image),values in sorted(image_values.items()):
        row={"model":model,"image":image,"original_h4":float(np.mean(values["original"])),"matched_aggregation_h4":float(np.mean(values["matched"])),"matched_path_vs_activation_gradient_rho":float(np.mean(values["agreement"]))}
        max_error=max(max_error,abs(row["original_h4"]-old[model,image]));image_rows.append(row)
    summary=[]
    for model in sorted({r["model"] for r in image_rows}):
        for metric in ("original_h4","matched_aggregation_h4","matched_path_vs_activation_gradient_rho"):
            data=[float(r[metric]) for r in image_rows if r["model"]==model];median,low,high=bootstrap_median(data,f"h4matched|{model}|{metric}")
            summary.append({"model":model,"metric":metric,"n_images":len(data),"median":median,"ci95_low":low,"ci95_high":high,**sign_counts(data),"analysis_identity":"original_reproduction" if metric=="original_h4" else "post_hoc_aggregation_sensitivity"})
    return block_rows,image_rows,summary,max_error


def validate_random_baseline() -> dict[str,Any]:
    analysis=json.loads((BASELINE_ROOT/"baseline_analysis.json").read_text())
    image_rows=list(csv.DictReader((BASELINE_ROOT/"baseline_image_values.csv").open()))
    repeat_rows=list(csv.DictReader((BASELINE_ROOT/"baseline_repeat_image_values.csv").open()))
    cohort_rows=list(csv.DictReader((BASELINE_ROOT/"baseline_repeat_cohort_values.csv").open()))
    require(analysis["status"]=="COMPLETE_CORRECTED_ESTIMAND" and analysis["repeat_rows"]==32000,"RANDOM_BASELINE_STATUS")
    require(len(image_rows)==1600 and len(repeat_rows)==32000 and len(cohort_rows)==80,"RANDOM_BASELINE_COVERAGE")
    groups=Counter((r["model"],r["reference"]) for r in image_rows);require(set(groups.values())=={400},"RANDOM_BASELINE_IMAGE_GROUPS")
    repeats=defaultdict(set)
    for row in repeat_rows:repeats[row["model"],row["reference"],row["sample_id"]].add(int(row["repeat"]))
    require(all(value==set(range(20)) for value in repeats.values()),"RANDOM_BASELINE_REPEAT_IDS")
    stability=[]
    for row in analysis["repeat_stability"]:
        require(set(row["resampling"])=={"5","10","20"},"RANDOM_BASELINE_RESAMPLING")
        stability.append({"model":row["model"],"reference":row["reference"],"relative_mc_se_median":row["relative_mc_se_median"],"relative_mc_se_q90":row["relative_mc_se_q90"],"r5_error":row["resampling"]["5"]["median_image_absolute_mc_error"],"r10_error":row["resampling"]["10"]["median_image_absolute_mc_error"],"r20_error":row["resampling"]["20"]["median_image_absolute_mc_error"]})
    return {"schema_version":1,"status":"ALREADY_COMPLETE_AND_SOURCE_VALIDATED","estimand":"absolute effect before within-repeat AUC, then mean over 20 repeats and heads within image","image_rows":len(image_rows),"repeat_rows":len(repeat_rows),"cohort_repeat_rows":len(cohort_rows),"checks":["20 repeats for every model-reference-image","relative MC SE","5/10/20 repeat resampling","split-half","leave-one-repeat-out"],"stability":stability}


def analyze() -> None:
    verify_frozen();architecture=build_registries_and_graph_audit()
    threshold_rows,entropy_rows,entropy_summary,entropy_error=entropy_analysis();require(entropy_error<=5e-7,"M3_ENTROPY_REPRODUCTION")
    h4_blocks,h4_images,h4_summary,h4_error=h4_analysis();require(h4_error<=1e-12,"H4_REPRODUCTION")
    write_csv(ROOT/"entropy_mass_thresholds.csv",threshold_rows,list(threshold_rows[0]))
    write_csv(ROOT/"entropy_mass_image_values.csv",entropy_rows,list(entropy_rows[0]))
    write_csv(ROOT/"entropy_mass_sensitivity.csv",entropy_summary,list(entropy_summary[0]))
    write_csv(ROOT/"h4_matched_block_values.csv",h4_blocks,list(h4_blocks[0]))
    write_csv(ROOT/"h4_matched_image_values.csv",h4_images,list(h4_images[0]))
    write_csv(ROOT/"h4_matched_summary.csv",h4_summary,list(h4_summary[0]))
    random_report=validate_random_baseline();write_json(ROOT/"random_region_precision_audit.json",random_report)
    receipt={"schema_version":1,"status":"REVISION_V8_ANALYSIS_PASS","cpu_only":True,"gpu_inference_runs":0,"architecture_status":architecture["status"],"scalar_target_counts":architecture["scalar_target_counts"],"target_cut_pair_count":architecture["target_cut_pair_count"],"m3_entropy_reproduction_max_error":entropy_error,"h4_reproduction_max_error":h4_error,"entropy_image_count":len(entropy_rows),"h4_image_count":len(h4_images),"random_region_precision":random_report["status"],"original_decisions_unchanged":True,"new_confirmatory_tests":0,"next_authorization":"REVISION_V8_MANUSCRIPT_INTEGRATION"}
    write_json(ROOT/"terminal_receipt.json",receipt);print(json.dumps(receipt,ensure_ascii=False))


def main() -> int:
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument("--freeze",action="store_true");group.add_argument("--analyze",action="store_true");args=parser.parse_args();freeze() if args.freeze else analyze();return 0


if __name__=="__main__":raise SystemExit(main())
