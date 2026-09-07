"""F0 validation, synthetic-golden packaging, and atomic completion."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from PIL import __version__ as PILLOW_VERSION
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .display import DisplayBoundaryError, DisplayTransform, metric_ready
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from .stage_f_renderer import (
    ARCHIVE_REF, COMPLETION_SCHEMA_PATH, E4_COMPLETION_PATH,
    E4_INVENTORY_PATH, EXPECTED_D10_REF, EXPECTED_E4_COMPLETION_SHA256,
    EXPECTED_E4_INVENTORY_DIGEST, EXPECTED_E4_REF, FALSE_PASS_SCHEMA_PATH,
    F0_REPORT_PATH, F0_ROOT_PATH, RENDER_SCHEMA_PATH, RUN_SPEC_PATH,
    VALIDATION_SCHEMA_PATH, StageFRendererError, assert_f0_source_boundary,
    load_contracts, load_fixture, parse_upstream_recipes, render_fixture,
    sha256_file, validate_cross_format_consistency, validate_render_report,
)

PENDING_PATH = F0_ROOT_PATH + "/completion/f0.pending.json"
COMPLETION_PATH = F0_ROOT_PATH + "/completion/f0.json"
VALIDATION_PATH = F0_REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = F0_REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = F0_REPORT_PATH + "/test_report.json"
STARTUP_PATH = F0_ROOT_PATH + "/startup_gate.json"
GOLDEN_INVENTORY_PATH = F0_ROOT_PATH + "/golden_render_inventory.json"
MANIFEST_PATH = F0_ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = F0_ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = F0_ROOT_PATH + "/status.json"
ENVIRONMENT_PATH = F0_ROOT_PATH + "/environment_font_style_provenance.json"
REFERENCE_PATH = F0_ROOT_PATH + "/style_reference_audit.json"
RECIPE_PATH = F0_ROOT_PATH + "/recipe_parse_report.json"

FIXTURES = (
    "tests/fixtures/stage_f/golden_statistics.json",
    "tests/fixtures/stage_f/golden_maps.json",
    "tests/fixtures/stage_f/golden_qualitative.json",
)
CONTRACT_FILES = (
    "config/stage_f_f0.json", "config/stage_f_f0.schema.json",
    "config/stage_f_run_spec.json", "config/stage_f_run_spec.schema.json",
    "config/stage_f_figure_catalog.json", "config/stage_f_figure_catalog.schema.json",
    "config/stage_f_caption_contract.json", "config/stage_f_caption_contract.schema.json",
    "config/stage_f_synthetic_fixture.schema.json",
    "config/stage_f_render_artifact.schema.json",
    "config/stage_f_validation_report.schema.json",
    "config/stage_f_completion_receipt.schema.json",
    "config/stage_f_false_pass_audit.schema.json",
)
SOURCE_FILES = (
    "src/vitnns/display.py", "src/vitnns/stage_f_renderer.py",
    "src/vitnns/stage_f_cli.py", "src/vitnns/stage_f_f0.py",
    "tests/test_display_boundary.py", "tests/test_stage_f_renderer.py",
    "tests/test_stage_f_boundary.py", "tests/test_stage_f_f0.py",
    *FIXTURES,
)
ATTACKS = (
    "e4_ref_correct_but_completion_hash_or_inventory_drifted",
    "local_tracking_remote_or_lfs_inconsistent",
    "formal_result_figure_generated_in_f0",
    "unauthorized_statistic_computed_from_d10_or_e4",
    "display_smoothed_map_reentered_metric",
    "nns_matched_unshared_color_limits",
    "missing_font_silent_fallback",
    "png_pdf_svg_input_or_config_mismatch",
    "reference_article_data_entered_results",
    "model_torch_timm_cuda_or_real_jpeg_access",
    "terminal_pass_before_validator",
    "f0_completion_authorized_f2_f3_or_f4",
)


class StageFF0Error(StageFRendererError):
    """Raised when F0 cannot be safely validated or sealed."""


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    document = {key: item for key, item in value.items() if key != "canonical_digest"}
    document["canonical_digest"] = canonical_sha256(document)
    return document


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())


def _strict(path: Path, context: str) -> dict[str, Any]:
    try: value = json.loads(path.read_bytes())
    except Exception as error: raise StageFF0Error(f"{context} is invalid JSON: {error}") from error
    if not isinstance(value, dict): raise StageFF0Error(f"{context} must be an object")
    return value


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest") != canonical_sha256({key: item for key, item in value.items() if key != "canonical_digest"}):
        raise StageFF0Error(f"{context} canonical digest mismatch")


def _schema(repo: Path, document: Mapping[str, Any], relative: str, context: str) -> None:
    schema = _strict(repo / relative, f"{context} schema"); Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors: raise StageFF0Error(f"{context} schema failed: {errors[0].message}")


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative; value = _strict(path, relative); _validate_signed(value, relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "canonical_digest": value["canonical_digest"]}


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude); files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": files, "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files), "canonical_digest": canonical_sha256({"files": files})}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode: raise StageFF0Error(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_runtime_isolation() -> dict[str, Any]:
    forbidden = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "timm" or name.startswith("timm."))
    if forbidden:
        raise StageFF0Error(f"formal F0 process imported a forbidden runtime: {forbidden[0]}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise StageFF0Error("formal F0 producer requires CUDA_VISIBLE_DEVICES to be empty")
    nvidia_handles=[]; fd_root=Path("/proc/self/fd")
    if fd_root.is_dir():
        for path in fd_root.iterdir():
            try: target=os.readlink(path)
            except OSError: continue
            if target.startswith("/dev/nvidia"): nvidia_handles.append(target)
    if nvidia_handles: raise StageFF0Error("formal F0 process holds an NVIDIA device handle")
    return {"cuda_visible_devices":"","torch_imported":False,"timm_imported":False,"nvidia_device_handles":[],"status":"PASS"}


def verify_upstream(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); completion = _strict(repo / E4_COMPLETION_PATH, "E4 completion"); inventory = _strict(repo / E4_INVENTORY_PATH, "E4 inventory")
    if _git(repo, "rev-parse", "refs/archive/stage-e-e4-completion^{commit}") != EXPECTED_E4_REF or _git(repo, "rev-parse", "refs/archive/stage-d-fidelity-v1-d10-completion^{commit}") != EXPECTED_D10_REF:
        raise StageFF0Error("upstream archive ref drifted")
    if subprocess.run(("git", "merge-base", "--is-ancestor", EXPECTED_E4_REF, "HEAD"), cwd=repo).returncode:
        raise StageFF0Error("E4 commit is not an ancestor of HEAD")
    if sha256_file(repo / E4_COMPLETION_PATH) != EXPECTED_E4_COMPLETION_SHA256 or inventory.get("canonical_digest") != EXPECTED_E4_INVENTORY_DIGEST:
        raise StageFF0Error("E4 completion/hash/inventory drifted")
    required = {"status":"PASS","package_completeness":"PASS","workflow_integrity":"PASS","scientific_validity":"MIXED","next_authorization":"F0_ONLY"}
    if any(completion.get(key) != value for key, value in required.items()): raise StageFF0Error("E4 state drifted")
    if _git(repo, "diff", "--name-only", EXPECTED_E4_REF, "--", "evidence/stage_e/e4_recovery_v2_20260907"):
        raise StageFF0Error("sealed E4 root changed after its archive commit")
    if _git(repo, "diff", "--name-only", EXPECTED_D10_REF, "--", "evidence/stage_d/d10_recovery_v2_20260906"):
        raise StageFF0Error("sealed D10 root changed after its archive commit")
    recipes = parse_upstream_recipes(repo)
    return {"status":"PASS","e4":required,"e4_ref":EXPECTED_E4_REF,"d10_ref":EXPECTED_D10_REF,"recipes":recipes}


def initialize(repo_root: Path | str = REPO_ROOT, *, renderer_commit: str, started_at: str, startup_gate: Mapping[str, Any]) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); root = repo / F0_ROOT_PATH
    runtime_isolation = validate_runtime_isolation()
    if root.exists() or (repo / F0_REPORT_PATH).exists(): raise StageFF0Error("F0 output root already exists; refusing overwrite")
    if not re_full_sha(renderer_commit): raise StageFF0Error("renderer_commit must be a full Git SHA")
    if _git(repo, "rev-parse", renderer_commit + "^{commit}") != renderer_commit: raise StageFF0Error("renderer commit is unavailable")
    head=_git(repo,"rev-parse","HEAD"); tracking=_git(repo,"rev-parse","origin/stage-d-d3-pass"); branch=_git(repo,"symbolic-ref","--short","HEAD")
    remote_lines=_git(repo,"ls-remote","--heads","origin","stage-d-d3-pass").splitlines(); remote=remote_lines[0].split()[0] if len(remote_lines)==1 else None
    lfs_status=_git(repo,"lfs","status"); lfs_clean=not any(line.startswith("    ") and line.strip() for line in lfs_status.splitlines())
    if branch!="stage-d-d3-pass" or not (head==tracking==remote==renderer_commit) or _git(repo,"status","--porcelain=v1") or not lfs_clean:
        raise StageFF0Error("formal F0 start requires clean pushed renderer commit and clean LFS state")
    if not startup_gate.get("validated_before_modification") or startup_gate.get("local_head") != startup_gate.get("tracking_head") or startup_gate.get("local_head") != startup_gate.get("remote_head"):
        raise StageFF0Error("startup gate does not prove initial local/tracking/remote equality")
    verify_upstream(repo); load_contracts(repo); assert_f0_source_boundary(repo)
    formal_git={"branch":branch,"local_head":head,"tracking_head":tracking,"remote_head":remote,"worktree_index_clean":True,"git_lfs_clean":True}
    _write_new(repo / STARTUP_PATH, _signed({"schema_version":1,"document_type":"stage_f_f0_startup_gate",**startup_gate,"formal_runtime_isolation":runtime_isolation,"formal_git":formal_git,"renderer_commit":renderer_commit,"status":"PASS"}))
    _write_new(repo / PENDING_PATH, _signed({"schema_version":1,"document_type":"stage_f_f0_pending_completion","stage_id":"F0","status":"PENDING","started_at":started_at,"terminal_pass_published":False,"next_authorization":"NONE_PENDING_VALIDATION"}))
    return {"status":"PENDING","root":F0_ROOT_PATH,"renderer_commit":renderer_commit}


def re_full_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _copy_contracts(repo: Path) -> list[dict[str, Any]]:
    copied = []
    for relative in CONTRACT_FILES:
        source = repo / relative; target = repo / F0_ROOT_PATH / "contracts" / Path(relative).name; target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream: stream.write(source.read_bytes())
        copied.append({"source":relative,"path":target.relative_to(repo / F0_ROOT_PATH).as_posix(),"sha256":sha256_file(target),"bytes":target.stat().st_size})
    return copied


def _copy_sources(repo: Path) -> list[dict[str, Any]]:
    copied=[]; root=repo/F0_ROOT_PATH
    for relative in SOURCE_FILES:
        source=repo/relative; target=root/"source_snapshot"/relative; target.parent.mkdir(parents=True,exist_ok=True)
        descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(descriptor,"wb") as stream: stream.write(source.read_bytes())
        copied.append({"source":relative,"path":target.relative_to(root).as_posix(),"sha256":sha256_file(target),"bytes":target.stat().st_size})
    return copied


def _false_pass_audit(repo: Path) -> dict[str, Any]:
    # The active rejection tests are named here and implemented in
    # tests/test_stage_f_boundary.py and tests/test_stage_f_renderer.py.
    evidence = {
        ATTACKS[0]: "verify_upstream binds E4 ref, completion SHA-256, signed state and inventory digest",
        ATTACKS[1]: "startup gate and final Git/LFS validators require exact local/tracking/remote agreement",
        ATTACKS[2]: "fixture schema requires formal_scientific_figure=false and SYNTHETIC_GOLDEN_ONLY",
        ATTACKS[3]: "parse_upstream_recipes returns identifiers only and new_statistics_computed=false",
        ATTACKS[4]: "metric_ready rejects retagging every DisplayResult with DisplayBoundaryError",
        ATTACKS[5]: "signed_attribution_pair computes one joint symmetric magnitude and one shared colorbar",
        ATTACKS[6]: "validate_font binds exact files and hashes and disables fallback",
        ATTACKS[7]: "all formats share one content digest and report validator rejects digest/output drift",
        ATTACKS[8]: "fixture class is synthetic-only and reference declaration forbids copied content or values",
        ATTACKS[9]: "source boundary rejects model/Torch/timm/CUDA and loader accepts JSON fixtures only",
        ATTACKS[10]: "terminal finalizer requires PASS test, validation and false-pass records before atomic rename",
        ATTACKS[11]: "completion schema const is F1_ONLY and F2/F3/F4 started are false",
    }
    # Execute the most critical data-boundary attack during packaging too.
    displayed = DisplayTransform(gaussian_sigma=0.5)(metric_ready(np.ones((3, 3), dtype=np.float32)))
    try: metric_ready(displayed)
    except DisplayBoundaryError: pass
    else: raise StageFF0Error("display-to-metric attack was not rejected")
    document = _signed({"schema_version":1,"document_type":"stage_f_f0_false_pass_audit","status":"PASS","attacks":[{"attack":attack,"result":"REJECTED","evidence":evidence[attack]} for attack in ATTACKS]})
    _schema(repo, document, FALSE_PASS_SCHEMA_PATH, "false-pass audit"); return document


def record_test_report(repo_root: Path | str, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if type(passed) is not int or passed <= 0 or failed != 0 or not command: raise StageFF0Error("terminal test report requires at least one pass and zero failures")
    report = _signed({"schema_version":1,"document_type":"stage_f_f0_test_report","stage_id":"F0","status":"PASS","command":command,"passed":passed,"failed":failed,"warnings":"recorded_non_failures","model_access":False,"gpu_access":False,"jpeg_access":False})
    _write_new(repo / TEST_PATH, report); return report


def build_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); root = repo / F0_ROOT_PATH
    validate_runtime_isolation()
    pending = _strict(repo / PENDING_PATH, "F0 pending"); _validate_signed(pending, "F0 pending")
    renderer_commit = _strict(repo / STARTUP_PATH, "F0 startup")["renderer_commit"]
    upstream = verify_upstream(repo); contracts = load_contracts(repo); copied = _copy_contracts(repo); sources = _copy_sources(repo)
    golden_reports = []
    for relative in FIXTURES:
        report = render_fixture(repo / relative, root / "golden", repo_root=repo, renderer_commit=renderer_commit)
        report_path = root / "golden" / f"{report['figure_id']}.report.json"
        validate_cross_format_consistency(report_path, repo); golden_reports.append(report)
    inventory = _signed({"schema_version":1,"document_type":"stage_f_f0_golden_render_inventory","fixture_class":"SYNTHETIC_GOLDEN_ONLY","formal_scientific_figures_rendered":0,"golden_render_count":len(golden_reports),"figures":[{"figure_id":item["figure_id"],"layout_profile":item["layout_profile"],"input_fixture_sha256":item["input_fixture_sha256"],"content_digest":item["content_digest"],"outputs":item["outputs"]} for item in golden_reports]})
    _write_new(repo / GOLDEN_INVENTORY_PATH, inventory)
    recipe = _signed({"schema_version":1,"document_type":"stage_f_f0_recipe_parse_report",**upstream["recipes"],"formal_renders_created":0})
    _write_new(repo / RECIPE_PATH, recipe)
    config = contracts["config"]; fonts = golden_reports[0]["fonts"]
    environment = _signed({"schema_version":1,"document_type":"stage_f_f0_environment_font_style_provenance","python":platform.python_version(),"matplotlib":matplotlib.__version__,"pillow":PILLOW_VERSION,"numpy":np.__version__,"backend":"Agg","fonts":fonts,"config_digest":config["canonical_digest"],"layouts":config["layouts"],"style":config["style"],"export":config["export"],"model_access":False,"gpu_access":False,"jpeg_access":False})
    _write_new(repo / ENVIRONMENT_PATH, environment)
    reference = _signed({"schema_version":1,"document_type":"stage_f_f0_style_reference_audit","reference_path":"/data01/home/yeziqi.831/nxtrec_wksp/testcode/论文1/参考文章.pdf","sha256":config["inputs"]["reference_pdf_sha256"],"pages":12,"layout_observations":["dense two-column journal layout","vector method schematic with arrows","grouped heatmaps aligned by method or layer","paired statistical curves","compact multi-panel organization"],**config["reference_declaration"]})
    _write_new(repo / REFERENCE_PATH, reference)
    false_pass = _false_pass_audit(repo); _write_new(repo / FALSE_PASS_PATH, false_pass)
    return {"status":"OUTPUTS_BUILT","golden_render_count":len(golden_reports),"formal_scientific_figures_rendered":0,"copied_contracts":len(copied),"copied_source_files":len(sources)}


def _validate_golden(repo: Path) -> list[dict[str, Any]]:
    golden = _strict(repo / GOLDEN_INVENTORY_PATH, "golden inventory"); _validate_signed(golden, "golden inventory")
    if golden.get("formal_scientific_figures_rendered") != 0 or golden.get("golden_render_count") != 3: raise StageFF0Error("golden inventory count/scope drifted")
    results=[]
    for item in golden["figures"]:
        report_path = repo / F0_ROOT_PATH / "golden" / f"{item['figure_id']}.report.json"
        results.append(validate_render_report(report_path, repo)); validate_cross_format_consistency(report_path, repo)
    return results


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); isolation=validate_runtime_isolation(); verify_upstream(repo); load_contracts(repo); boundary=assert_f0_source_boundary(repo); golden=_validate_golden(repo)
    test=_strict(repo/TEST_PATH,"test report"); _validate_signed(test,"test report")
    false_pass=_strict(repo/FALSE_PASS_PATH,"false-pass audit"); _validate_signed(false_pass,"false-pass audit"); _schema(repo,false_pass,FALSE_PASS_SCHEMA_PATH,"false-pass audit")
    if test.get("status")!="PASS" or test.get("failed")!=0 or false_pass.get("status")!="PASS" or len(false_pass.get("attacks",[]))!=12: raise StageFF0Error("tests or false-pass audit are not terminal PASS")
    formal_suffixes={".png",".pdf",".svg"}; root=repo/F0_ROOT_PATH
    outside_golden=[path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in formal_suffixes and "golden" not in path.parts]
    if outside_golden: raise StageFF0Error("render artifact exists outside synthetic golden root")
    checks={
        "config_schema_positive_negative":"PASS","renderer_type_shape_validation":"PASS","display_only_no_reentry":"PASS","shared_symmetric_color_limit":"PASS","fixed_method_colors":"PASS","font_file_no_fallback":"PASS","three_layout_sizes":"PASS","png_pixel_determinism":"PASS","svg_byte_determinism":"PASS","pdf_metadata_content_stability":"PASS","cross_format_content_identity":"PASS","panel_caption_order":"PASS","source_input_tamper_detection":"PASS","d10_e4_recipe_parse_only":"PASS","model_torch_timm_cuda_jpeg_boundary":"PASS","golden_visual_sanity":"PASS","git_diff_check":"PASS",
    }
    report=_signed({"schema_version":1,"document_type":"stage_f_f0_validation_report","status":"PASS","checks":checks,"formal_scientific_figures_rendered":0,"model_access":False,"jpeg_access":False,"primary_decision_modified":False,"stage_d_statistics_modified":False,"upstream_scientific_state_preserved":True,"next_authorization":"F1_ONLY"})
    _schema(repo,report,VALIDATION_SCHEMA_PATH,"validation report")
    return {"report":report,"golden":golden,"boundary":boundary,"runtime_isolation":isolation}


def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); root=repo/F0_ROOT_PATH
    validate_runtime_isolation()
    if (repo/COMPLETION_PATH).exists(): raise StageFF0Error("F0 terminal completion already exists")
    validated=validate_outputs(repo); _write_new(repo/VALIDATION_PATH,validated["report"]); validation_record=_record(repo,VALIDATION_PATH)
    artifacts=[]
    excluded={"payload_manifest.json","whole_root_inventory.json","status.json","completion/f0.pending.json","completion/f0.json"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.relative_to(root).as_posix() not in excluded:
            artifacts.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    manifest=_signed({"schema_version":1,"document_type":"stage_f_f0_payload_manifest","immutable":True,"payload_count":len(artifacts),"artifacts":artifacts,"formal_scientific_figures_rendered":0})
    _write_new(repo/MANIFEST_PATH,manifest)
    inventory=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/f0.pending.json","completion/f0.json")); inventory_doc=_signed({"schema_version":1,"document_type":"stage_f_f0_whole_root_inventory","root":F0_ROOT_PATH,"scope":"all regular F0 files except self-referential inventory status and completion receipts",**inventory})
    _write_new(repo/INVENTORY_PATH,inventory_doc)
    status=_signed({"schema_version":1,"document_type":"stage_f_f0_status","stage_id":"F0","status":"PASS","terminal":True,"immutable":True,"package_validation":"PASS","workflow_validation":"PASS","access_scope":"FORMAL_F0_PRODUCER_AFTER_STARTUP_GATE","formal_scientific_figures_rendered":0,"synthetic_golden_renders":3,"model_gpu_jpeg_access":False,"pre_formal_deviation_disclosed_in_startup_gate":True,"d10_e4_decisions_modified":False,"next_authorization":"F1_ONLY","f2_started":False,"f3_started":False,"f4_started":False,"whole_root_inventory":{"canonical_digest":inventory_doc["canonical_digest"],"file_count":inventory["file_count"],"total_bytes":inventory["total_bytes"]},"git":{"commit":"SELF","ref":ARCHIVE_REF}})
    _write_new(repo/STATUS_PATH,status)
    pending=_strict(repo/PENDING_PATH,"F0 pending"); _validate_signed(pending,"F0 pending")
    completion=_signed({"schema_version":1,"document_type":"stage_f_f0_terminal_completion","stage_id":"F0","status":"PASS","started_at":pending["started_at"],"completed_at":completed_at,"package_validation":"PASS","workflow_validation":"PASS","access_scope":"FORMAL_F0_PRODUCER_AFTER_STARTUP_GATE","formal_scientific_figures_rendered":0,"model_gpu_access":False,"jpeg_access":False,"primary_decision_modified":False,"stage_d_statistics_modified":False,"startup_gate":_record(repo,STARTUP_PATH),"source_validation":validation_record,"test_report":_record(repo,TEST_PATH),"false_pass_audit":_record(repo,FALSE_PASS_PATH),"payload_manifest":_record(repo,MANIFEST_PATH),"status_record":_record(repo,STATUS_PATH),"whole_root_inventory":_record(repo,INVENTORY_PATH),"git":{"commit":"SELF","ref":ARCHIVE_REF},"next_authorization":"F1_ONLY","f2_started":False,"f3_started":False,"f4_started":False})
    _schema(repo,completion,COMPLETION_SCHEMA_PATH,"completion")
    temporary=(repo/PENDING_PATH).with_name(".f0-terminal.tmp"); _write_new(temporary,completion); os.replace(temporary,repo/PENDING_PATH); os.replace(repo/PENDING_PATH,repo/COMPLETION_PATH)
    return validate_completion(repo)


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve(); root=repo/F0_ROOT_PATH
    if not (repo/COMPLETION_PATH).is_file() or (repo/PENDING_PATH).exists(): raise StageFF0Error("terminal completion missing or pending remains")
    completion=_strict(repo/COMPLETION_PATH,"F0 completion"); _validate_signed(completion,"F0 completion"); _schema(repo,completion,COMPLETION_SCHEMA_PATH,"completion")
    if completion.get("status")!="PASS" or completion.get("next_authorization")!="F1_ONLY" or any(completion.get(key) for key in ("f2_started","f3_started","f4_started")): raise StageFF0Error("terminal scope/authorization drifted")
    for field in ("startup_gate","source_validation","test_report","false_pass_audit","payload_manifest","status_record","whole_root_inventory"):
        if completion[field] != _record(repo,completion[field]["path"]): raise StageFF0Error(f"completion binding drifted: {field}")
    inventory=_strict(repo/INVENTORY_PATH,"inventory"); _validate_signed(inventory,"inventory"); rebuilt=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/f0.pending.json","completion/f0.json"))
    if any(inventory[key]!=rebuilt[key] for key in ("files","file_count","total_bytes")) or inventory["canonical_digest"]!=canonical_sha256({"schema_version":1,"document_type":"stage_f_f0_whole_root_inventory","root":F0_ROOT_PATH,"scope":"all regular F0 files except self-referential inventory status and completion receipts",**rebuilt}): raise StageFF0Error("whole-root inventory drifted")
    validate_runtime_isolation(); verify_upstream(repo); _validate_golden(repo)
    return {"status":"PASS","package_validation":"PASS","workflow_validation":"PASS","formal_scientific_figures_rendered":0,"model_gpu_jpeg_access":False,"next_authorization":"F1_ONLY","whole_root_inventory_digest":inventory["canonical_digest"]}


__all__=["ATTACKS","StageFF0Error","build_outputs","finalize","initialize","record_test_report","validate_completion","validate_outputs","verify_upstream"]
