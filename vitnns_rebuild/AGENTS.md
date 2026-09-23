# ViT-NNS Rebuild Architecture Guide

This file applies to the repository root. The paper trees have additional rules in
`papers/nns_vit_path/AGENTS.md` and `papers/nns_vit_endpoint/AGENTS.md`; follow those
files as well when editing their subtrees.

## Project Overview

`vitnns-rebuild` is a Python 3.10 research codebase for auditable Vision Transformer
attribution and mechanism studies. It reconstructs a canonical timm ViT forward path,
implements conditional Value-path and Endpoint (`x * grad`) attribution, evaluates
fidelity and interventions, and produces hash-bound evidence, figures, and manuscripts.
It is a staged experiment pipeline, not an online service.

The project deliberately separates scientific intent from execution:

1. `docs/`, `config/`, and `manifests/` freeze the protocol, schemas, run specs,
   cohorts, targets, transformations, gates, and authorization boundaries.
2. `src/vitnns/` contains reusable model, attribution, execution, analysis, validation,
   and rendering code. Declared console entry points cover Stages A, B, C, D, and F.
3. `scripts/` provides study-specific orchestration for later Endpoint, patch-depth,
   revision, Stage X, and Stage Y work. Most studies have distinct run/worker, analyze,
   validate, render, and seal scripts rather than one global CLI.
4. `evidence/`, `reports/`, and `artifacts/` hold execution products and receipts. Large
   data, checkpoints, and generated roots are generally local and excluded from the
   lightweight Git checkout; a fresh clone is not a complete scientific archive.
5. `papers/` consumes sealed evidence through evidence matrices and figure manifests.
   It must not derive numerical claims from display PNGs.

The common execution shape is:

`frozen protocol/run spec -> preflight/authorization -> isolated worker(s) -> source-recomputing validator -> immutable terminal receipt -> analysis/figure/paper package`

### Important architectural boundaries

- `modeling.py`, `dataset.py`, `extractor.py`, `attribution.py`, and `provenance.py` form
  the canonical model/input foundation. Model loading verifies the state-dict identity,
  uses `weights_only=True`, loads strictly, and checks ViT geometry.
- Stage families are named modules such as `stage_c_*`, `stage_d_*`, `stage_y_*`,
  `endpoint_*`, and `nns_patch_depth_*`. Keep stage-specific policy in its family; do
  not add hidden cross-stage defaults.
- Newer studies split a lightweight control plane, NumPy/CPU analysis, and a heavy
  worker with lazy Torch/timm imports. Preserve that split when a protocol forbids
  model, image, or outcome access before authorization.
- A failed or sealed root is evidence. Recoveries use a new root and a prospective
  amendment; they do not overwrite terminal records or silently relax thresholds.
- Stage completion does not authorize the next expensive stage. Read the relevant live
  plan and its terminal receipt before executing anything. For Endpoint work, the single
  route entry is `docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md`.
- This repository is the active source of truth. The removed legacy `49_vit` tree is
  history/reference only and must not be treated as current implementation.

## Build and Commands

There is no compilation or service deployment step. The build is an editable Python
package; a research "release" is a validated/sealed evidence, report, figure, or paper
package produced by its stage-specific command.

### Environment setup

The checked-in bootstrap expects project-local Conda under the parent `testcode/.conda`:

```bash
./scripts/bootstrap_conda.sh
```

For an existing environment:

```bash
source ../.conda/etc/profile.d/conda.sh
conda activate vitnns
python -m pip install -e . --no-deps
python -m pip check
```

`environment.yml` creates the Python 3.10 environment. `requirements.lock.txt` pins the
full workspace, including Torch 2.2.2 CUDA 12.1 wheels; `pyproject.toml` declares the
smaller package runtime and pytest extra. Do not casually regenerate either lock surface.

### Main command surfaces

```bash
vitnns-stage-a --help   # deterministic manifests and canonical forward verification
vitnns-stage-b --help   # fixed-clean-attention conditional IG
vitnns-stage-c --help   # plans, runs, validation, aggregation, C7-C10 workflows
vitnns-stage-d --help   # contracts, authorization, production, sealing, paper data
vitnns-stage-f --help   # figure-stage initialization, rendering, tests, completion
```

Examples documented by the root README include:

```bash
vitnns-stage-b estimate --config config/stage_b.json
vitnns-stage-c estimate --config config/stage_c.json --run-spec PATH_TO_RUN_SPEC.json
vitnns-stage-d validate-contracts
vitnns-stage-d plan
vitnns-stage-d estimate
```

Do not copy historical README commands containing absent `/data00/...` roots. Recovery
bindings and repository-local evidence paths supersede them. For later studies, use the
exact `python scripts/run_*.py`, `validate_*.py`, or `seal_*.py` command recorded in that
study's live plan/run spec; similar script names often represent immutable revisions, not
interchangeable entry points.

Useful recovery commands are:

```bash
./scripts/verify_recovery_assets.sh
./scripts/rerun_stage_d_d10_lightweight.sh
```

The D10 rerun is CPU-only and requires the locally restored, hash-matching sealed D8/D9
inputs. It creates a timestamped fresh root and supports the documented `D8_ROOT`,
`D9_REPORT`, `D9_FORMAL_BUDGET`, `D10_RERUN_ROOT`, and `PYTHON` overrides.

## Code Style

No Black, Ruff, isort, mypy, or pre-commit configuration is declared. Do not claim a
formatter gate or reformat unrelated frozen code. Follow the surrounding module and these
observed conventions:

- Use Python 3.10 syntax, `from __future__ import annotations` in modern modules, type
  hints, `pathlib.Path`, and stage-specific exception classes.
- Use `snake_case` for functions/variables/modules, `PascalCase` for classes, and
  `UPPER_SNAKE_CASE` for frozen constants and authorization tokens. Stage/revision IDs
  remain explicit in filenames and document fields because they carry provenance.
- Prefer frozen dataclasses or immutable mappings for loaded contracts. Parse JSON as
  UTF-8, reject duplicate keys and non-finite values, and validate semantic relationships
  after JSON Schema validation.
- Keep schemas closed (`additionalProperties: false`) and outputs explicit. Adding a field
  requires coordinated updates to its schema, producer, validator, tests, and any canonical
  digest or inventory. Never add an open-ended payload to bypass that work.
- Use the stage's canonical serialization/signing helpers. JSON records commonly use
  sorted keys plus a trailing newline; signed records exclude then recompute
  `canonical_digest`. Hand-editing a sealed or signed JSON file invalidates its identity.
- Preserve write-once and atomic publication behavior: create fresh paths, use exclusive
  creation for unit artifacts, and publish terminal records only after validation.
- Keep heavy imports inside authorized workers when the control plane must stay metadata-
  only or outcome-blind. Pure statistical/operator modules should remain independently
  testable without model initialization.
- Scientific arrays have explicit dtype, shape, axes, and hash contracts. Load NumPy files
  with `allow_pickle=False`; do not substitute rendered images for raw arrays.

## Testing

The test framework is pytest 8.4.2, configured in `pyproject.toml` with `tests/` as the
test root and `-ra` enabled. The suite combines unit tests, JSON Schema/contract tests,
numeric regression tests, subprocess boundary tests, tamper/false-pass tests, and tests
that use real local checkpoints/images. There is no checked-in CI workflow; verification
is performed locally and formal stages record their test command and counts in receipts.

Documented normal run:

```bash
python -m pytest -m 'not unsafe_pickle'
```

During development, run the file matching the changed family first, then the normal run:

```bash
python -m pytest tests/test_stage_c_config.py -q
python -m pytest -m integration
```

`integration` tests require the real local ViT-B checkpoint and images. Session fixtures
in `tests/conftest.py` assert that those assets exist. The unsafe compatibility test is
separate and must only be run for the known trusted pickle:

```bash
VITNNS_ALLOW_UNSAFE_PICKLE=1 python -m pytest -m unsafe_pickle
```

Stage D boundary tests use `tests/d0_guard` through `tests/d7_guard` `sitecustomize.py`
modules. Their exact `PYTHONPATH`, `VITNNS_D*_STRICT=1`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and empty `CUDA_VISIBLE_DEVICES` commands are part of
the corresponding stage contract; do not run those tests without their guard environment.

For a contract or validator change, tests should cover the valid record and the relevant
fail-closed cases: missing/extra fields, digest or byte drift, forbidden access, partial
coverage, and terminal publication order. A producer passing its own test is not enough;
run the independent validator named by the live stage document.

## Security and Data Protection

- Treat model files and data as untrusted until their bound SHA-256 values are recomputed.
  Canonical models use safe state-dict loading and strict architecture checks.
- Full-model `.pth` loading executes pickle semantics. It is compatibility-only, requires
  both an explicit code opt-in and the `VITNNS_ALLOW_UNSAFE_PICKLE=1` test gate, and is not
  a production dependency. A matching hash does not make pickle safe.
- Raw ImageNet data, official devkit material, checkpoints, and large evidence remain
  local. Do not commit payloads that `.gitignore` intentionally excludes; track manifests,
  checksums, source bindings, and restoration instructions instead. ImageNet labels come
  from the canonical WNID mapping/devkit, never directory enumeration order.
- Respect declared read whitelists. CPU-only/metadata-only phases may explicitly forbid
  JPEG decoding, checkpoint access, Torch/timm imports, or CUDA initialization. The guard
  modules enforce these restrictions process-wide.
- Set deterministic CUDA controls before importing Torch in formal workers. The recurring
  contract is `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one explicitly selected GPU through
  `CUDA_VISIBLE_DEVICES`, deterministic algorithms, cuDNN benchmark disabled, and TF32
  disabled. Use the current run spec for the complete stage-specific gate.
- CPU-only validation/analysis normally requires `CUDA_VISIBLE_DEVICES=''`. A leaked GPU
  environment is an access-boundary failure, not a scientific result.
- Preserve failed, partial, and superseded outputs. Do not interpret incomplete roots as
  scientific evidence, delete failure receipts, reuse a terminal root, or convert an
  infrastructure/access error into a scientific conclusion.

## Configuration Management

- `pyproject.toml`: package metadata, console entry points, core dependencies, pytest
  markers, and test root.
- `environment.yml` and `requirements.lock.txt`: Python/Conda bootstrap and exact workspace
  dependency snapshot.
- `config/*.json` and `config/**/*.json`: frozen protocols, run specs, budgets, recovery
  amendments, authorization records, and their Draft 2020-12 schemas. Most stages bind
  config content by SHA-256 and/or canonical digest.
- `manifests/`: immutable sample identities, cohorts/splits, target and transform
  registries, execution schedules, and access ledgers. Change selection logic by creating
  a new revision, not by editing a used manifest in place.
- `config/data_roots.example.json`, recovery bindings, and CLI flags describe local paths.
  Do not embed a developer's new absolute data/checkpoint path into reusable code.
- Environment variables are part of provenance, not informal shell state. Record the
  resolved Python executable, CUDA visibility, cuBLAS setting, dependency versions, model
  identity, Git commit/dirty state, seeds, and input/config hashes when the stage contract
  calls for them.

## Development and Debugging Workflow

Before modifying a stage, read its live plan, run spec, schema, last terminal receipt, and
matching source/tests. Then keep the change inside the appropriate layer:

- Contract/preflight failures: inspect config/schema, manifest bindings, authorization,
  Git cleanliness, asset hashes, and environment identity before touching model code.
- Forward or attribution mismatches: start in `modeling.py`, `extractor.py`, the relevant
  operator module, and its numeric parity tests. Do not bypass the canonical timm forward
  with a replayed approximation unless the protocol explicitly defines one.
- Missing/invalid evidence: inspect the worker terminal, artifact inventory, byte/SHA and
  canonical digests, then run the independent validator. Do not patch the aggregate.
- Guard failures: treat the reported forbidden import/path/environment access as the bug;
  moving the access earlier or disabling the guard violates the architecture.
- Scientific gate failures: preserve the terminal result. Any revised estimator, threshold,
  cohort, or access rule requires a prospective amendment and a fresh root.

Keep changes scoped to the requested stage. Validate the prior stage first, execute only the
authorized stage, and stop at explicit `next_authorization`/user-request boundaries.
