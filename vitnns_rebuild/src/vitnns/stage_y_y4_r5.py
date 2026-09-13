"""Prospective Y4-R5 output-field contract recovery over frozen Y4-R4."""
from pathlib import Path

from . import stage_y_y4_r1 as _core
from . import stage_y_y4_r4 as _r4

STAGE_ID = "Y4-R5"
STARTUP_COMMIT = "9cb65a3677cac2daec432c6a0108ce97a0a36e27"
ROOT = Path("evidence/stage_y/y4_r5_mechanism_analysis_20260914/formal")
REPORT_ROOT = Path("reports/stage_y_y4_r5_20260914")
REPORT = REPORT_ROOT / "report.md"
DOC = Path("docs/stage_y_y4_r5_field_contract_recovery_20260914.md")
SPEC = Path("config/stage_y/y4_r5_run_spec.json")
SPEC_SCHEMA = Path("config/stage_y/y4_r5_run_spec.schema.json")
ARTIFACT_SCHEMA = Path("config/stage_y/y4_r5_artifact.schema.json")
PRODUCER_REF = "refs/archive/stage-y-y4-r5-mechanism-analysis-producer"
PASS_REF = "refs/archive/stage-y-y4-mechanism-analysis-and-decision"
BLOCKED_REF = "refs/archive/stage-y-y4-r5-mechanism-analysis-and-decision-blocked"
PRODUCER_FILES = (
    DOC,
    SPEC,
    SPEC_SCHEMA,
    ARTIFACT_SCHEMA,
    Path("src/vitnns/stage_y_y4_r5.py"),
    Path("scripts/run_stage_y_y4_r5.py"),
    Path("scripts/validate_stage_y_y4_r5.py"),
    Path("tests/test_stage_y_y4_r5.py"),
)
DEPENDENCIES = _r4.PRODUCER_FILES + _r4.DEPENDENCIES

for _module in (_core, _r4):
    for _name in (
        "STAGE_ID", "STARTUP_COMMIT", "ROOT", "REPORT_ROOT", "REPORT",
        "DOC", "SPEC", "SPEC_SCHEMA", "ARTIFACT_SCHEMA", "PRODUCER_REF",
        "PASS_REF", "BLOCKED_REF", "PRODUCER_FILES", "DEPENDENCIES",
    ):
        _module.__dict__[_name] = globals()[_name]

_core.EXPECTED_REFS.update({
    "refs/archive/stage-y-y4-r4-mechanism-analysis-producer":
        "277a3fa42f99009cc6da8cc51ce8f2fe0cb68200",
    "refs/archive/stage-y-y4-r4-mechanism-analysis-and-decision-blocked":
        STARTUP_COMMIT,
})

for _name in dir(_r4):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(_r4, _name)

_inherited_startup = startup

def startup(repo, allow_sources=False):
    result = _inherited_startup(repo, allow_sources)
    r4_root = repo / "evidence/stage_y/y4_r4_mechanism_analysis_20260914/formal"
    require(
        sha(r4_root / "completion/y4_r4.json") ==
        "a59ca82fd1af20a0c0c769c9229d4bda0a0c61617afd574911ac10920d067d72",
        "Y4-R4 completion drift", StartupError,
    )
    require(
        sha(r4_root / "failure.json") ==
        "7dc1f01134c9a8058916dc0ebee58724296fc8e03c039aa4692922c659161b71",
        "Y4-R4 failure receipt drift", StartupError,
    )
    require(
        sha(r4_root / "whole_root_inventory.json") ==
        "1ba3b5539196e22404cf93a60b9be9d73c672b515d064340e0023fea3c759a13",
        "Y4-R4 inventory drift", StartupError,
    )
    return result

def seal_producer(repo, preflight):
    runtime_guard()
    head = git(repo, "rev-parse", "HEAD")
    rem = remote_refs(repo)
    require(
        head == git(repo, "rev-parse", "@{upstream}") ==
        rem.get("refs/heads/" + BRANCH) == git(repo, "rev-parse", PRODUCER_REF) ==
        rem.get(PRODUCER_REF),
        "producer publication incomplete", StartupError,
    )
    require(not git(repo, "status", "--porcelain"), "producer worktree dirty", StartupError)
    validate_schema_sources(repo)
    history = verify_expected_refs(repo)
    require(
        canonical([r for r in history if r["ref"] != PRODUCER_REF]) ==
        preflight["startup_refs_digest"],
        "historical ref snapshot changed", StartupError,
    )
    root = repo / ROOT
    require(not root.exists(), "Y4-R5 root exists", StartupError)
    write(root / "completion/y4_r5.pending.json", "pending", {
        "started_unix_ns": __import__("time").time_ns(),
        "next_authorization": "NONE_WHILE_PENDING",
    }, "PENDING")
    files = producer_files(repo)
    write(root / "producer_implementation_seal.json", "implementation_seal", {
        "producer_commit": head,
        "producer_ref": PRODUCER_REF,
        "source_files": files,
        "files_digest": canonical({"files": files}),
    })
    write(root / "producer_preflight.json", "producer_preflight", preflight)
    write(root / "startup_ref_snapshot.json", "binding", {
        "name": "startup_stage_y_refs", "files": [],
        "files_digest": preflight["startup_refs_digest"], "commit": STARTUP_COMMIT,
    })
    contracts = contract_bindings(repo)
    write(root / "y0_contract_bundle_binding.json", "binding", {
        "name": "Y0 contract bundle", "files": contracts,
        "files_digest": canonical(contracts),
        "commit": EXPECTED_REFS["refs/archive/stage-y-y0-protocol-freeze"],
    })
    write(root / "y2_completion_publication_binding.json", "binding", publication_binding(repo, "Y2"))
    write(root / "y3_completion_publication_binding.json", "binding", publication_binding(repo, "Y3"))
    protocol = [file_record(repo / DOC, repo), file_record(repo / SPEC, repo)]
    write(root / "prospective_protocol_binding.json", "binding", {
        "name": "Y4-R5 output-field contract recovery", "files": protocol,
        "files_digest": canonical(protocol), "commit": head,
    })
    verify_implementation(repo)
    return {"status": "PASS", "producer_commit": head}
