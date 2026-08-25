# Stage D D0：post-C10 fidelity-only 范围决议

> 决议日期：2026-08-25
>
> D0 状态：`PASS`
>
> 准备授权：`APPROVED_TO_PREPARE_STAGE_D`
>
> 执行授权：`execution_authorization=NOT_ISSUED`
>
> 分析授权：`analysis_authorization=NOT_EVALUATED`
> attempt 状态：首次尝试 `BLOCKED`；严格 model-free/no-JPEG 重试 `PASS`，且只有重试是当前授权依据。

## 1. 权威性和作用

本文件是机器记录 `config/stage_d_scope_decision.json` 的人类可读展开。机器记录及其 closed schema `config/stage_d_scope_decision.schema.json` 是 D0 scope、状态、claim ID、披露义务和授权边界的单一权威；本文不能扩张机器记录。任何不一致必须 fail closed，并在版本化修订中处理。

D0 只冻结 post-C10 研究范围，授权 D1–D7 继续准备工作；它不授权正式 Stage D execution，不表示已经形成 Stage D validity/effect/cost 结果，不允许读取 reserve 图像字节或 outcome，也不改变任何 sealed C9/C10 evidence。只有 D7 在全部前置合同、实现、smoke 和新预算通过后，才可能另行签发 `AUTHORIZED_STAGE_D_FIDELITY_SCOPE`。

## 2. C10 历史状态保持不变

D0 从 sealed C10 root 只读复核了文件 bytes、canonical digests、manifest/status 绑定、archive ref 和 whole-root inventory；没有重建 C10 科学统计，也没有写入 C9/C10。必须永久保留：

- `C10_PRIMARY_DECISION_NO_GO_NOVELTY`：C10 v1 primary decision 仍为 `NO_GO_NOVELTY`；
- `C10_ACCEPTANCE_BLOCKED`：Stage C acceptance 仍为 `BLOCKED`；
- `C10_HOLM_BLOCKER_UNDERSPECIFIED`：workflow blocker 仍为 `HOLM_BLOCKED_UNDERSPECIFIED`；
- `C10_SPATIAL_OVERALL_FAIL`：C10 overall spatial status 仍为 `FAIL_SPATIAL`；
- `C10_SPATIAL_COMPONENTS_MIXED`：clean joint spatial 与 Gaussian stability 通过，指定的 4-pixel translation 和 center-crop/resize stability 失败；
- `C10_DOWNSTREAM_NOT_AUTHORIZED`：历史字段仍为 `downstream_authorized=false` 和 `stage_d_started=false`。

这份新范围决议不是 C10 勘误或重判。它不得把 C10 改名为 GO，不得把 Stage C acceptance 改成 PASS，也不得宣称历史 Holm blocker 已被修复或取消。

### 2.1 Sealed evidence 绑定

- formal root：`/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824`；
- completion commit：`69e1fa887862e00da4441842c6de149fe2424225`；
- aggregation producer：`92ad81342ff22e2710df495a9961a3ab8c015a3a`；
- archive ref：`refs/archive/c10-aggregation-producer`；
- producer tree：`3551096e358391fe04952e8ced19ea60884fb32f`；
- producer `src/vitnns` tree：`6874e71aec0e592e110a540a1638ea2cfc697c8d`；
- sealed root：12 个普通、single-link 文件，`524796892` bytes；
- whole-root inventory digest：`382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4`。

关键文件的 SHA-256 / canonical digest：

| 文件 | SHA-256 | canonical digest |
|---|---|---|
| `status.json` | `192d383d5c4788f612018532074c06d48511513edd7565dbb99934706cd96db2` | `d08bcb62a0fa95fef81c75d955b88082a8e6788a76b1a0bfffe1aa4e0181eb87` |
| `stage_c_report.json` | `9a16034d29bc87eaba852aa0e618928ac0e4ecee9bf745d765f42311dd4563b4` | `90232df2010f6a8fc771baeac3aaac44fee93d436597cf0cbdebe3cf8d2142e2` |
| `decision.json` | `07fc5875e46018adf5dc0cd09967e273df3b443c1deeb1653d672d05103e513e` | `60e112d50f22229474a634039ba71c7eed1adba081d8e1097f850a8a4b4d8dc0` |
| `holm.json` | `be37ffbfd2ba60b5549ba791076383d9ba53a3b437178273a608fbf527b91781` | `271ba3625de6a91912728a32880cf1e1861aecb9f12903cf759e83e3332ab2fd` |
| `provenance.json` | `4dcca54fdc83ba7401070ef843636b5e67b68a8ce720fa70f729e407fc5717f0` | `4b26452041685cbb6c12d93139736e65fc4330fd44a64bf854848e7cff66c742` |
| `manifest.json` | `507ca3c4ac210882a5918cc602cf7a4656aa398af76ff9f3638a77f3af42a0d8` | `c71ae7c19941eab556a0ed9fdde553cb1a3fae914e149652df9b31cf3538720a` |

## 3. 新研究的科学依据与边界

`C10_PRIMARY_FIDELITY_COMPONENT_VALID` 和 `C10_PRIMARY_FIDELITY_COMPONENT_EFFECT_PASS` 是新 fidelity-only 研究的依据，但只能带完整限定语使用：

> 在 Stage C 的 15 张固定、历史文件名等间隔抽取且非类别均衡的 holdout 上，对冻结的 ViT-B/16、固定 clean-attention head-direction Value-path 标量、primary `(block=0, head=9)`、同目标 matched patch-token baseline 和既定 fine-4×4-cell intervention endpoint，C10 为 `PASS_VALIDITY` 与 `PASS_EFFECT`。NNS 的 fine-4×4-cell fidelity 高于 matched patch baseline；median `delta_fid=1.7534972022995547`，15/15 为正，95% paired image-bootstrap CI `[1.3317595184568727, 3.3813212483296478]`。

这项 component-level 证据不能脱离 15 图、抽样方式、模型、目标、primary、baseline 和 endpoint 表述。它不等于 C10 GO，也不支持普遍 ImageNet 优越。

`STAGE_D_IS_PROSPECTIVE_REPLICATION_EXTENSION`：Stage D 是在查看 C10 后新建的 prospective replication/extension。它检验相同的有限 fidelity 问题能否在另一个预先冻结的 500 图 cohort 上复制与外推。Stage D 若失败，必须限制更广泛的论文主张；它不会抹掉上述固定 15 图的历史 component 结果。

`NO_INTERMEDIATE_C_PLUS_REQUIRED`：C10 已经给出有效的 fidelity 单项证据，因此不增加 C+ 或另一轮小样本 fidelity 准入实验。Stage D 本身仍是新 cohort 的正式检验，不是把 fidelity 从此视为无需检验。

C10 的 `PASS_COST` 仅是原 8,000-unit workload 的历史可行性证据；它不授权新 Stage D。D7 必须按新的 1,000-unit typed plan 独立重算并通过预算。

## 4. 冻结的 Stage D primary scope

`STAGE_D_PRIMARY_SCOPE_DEFINED` 和 `FINE_ENDPOINT_IS_4X4_CELL` 的精确含义是：

- 模型：`vit_base_patch16_224.augreg2_in21k_ft_in1k`，state-dict SHA-256 `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`；
- 预处理：RGB、短边 248、bicubic、center crop 224×224、mean/std `(0.5,0.5,0.5)`；
- primary：`(block=0, head=9)`，不得事后重选；
- target：`fixed_clean_attention_cls_head_direction`；
- NNS：`conditional_fixed_clean_attention_head_direction_value_path_ig`；
- matched baseline：`matched_fixed_clean_attention_patch_token_contribution`；
- primary endpoint：`fine_4x4_cell_delta_fid`；
- statistical unit：image；C10 `n=15` 与 Stage D planned `n=500` 分开报告，不池化为 `n=515`；
- fine-4×4-cell 指每个 `16×16` patch 内 16 个非重叠 `4×4` cell，不是原生单像素 attribution；
- fidelity 角色是 `PRIMARY_HARD_SCIENTIFIC_GATE`。

D0 没有正式化 500 图 reserve。D1 才能在 metadata-only 边界内生成正式 registry；正式 Stage D cohort 还必须与 C10 cohort 互斥。任何“未见 reserve”表述只能限定为“未出现在已声明且封闭的 prior-access ledger 中”，不能声称绝对没有历史人工接触。

## 5. Spatial 和 Holm 的新旧命名空间

`STAGE_D_SPATIAL_NOT_IN_SCOPE`：Stage D 的 `run_stability=false`，不运行、不分析、不产生新的 spatial/stability endpoint；spatial 不进入 Stage D primary truth table。C10 的 spatial 结果仍必须在 Study Design、Results 和 Limitations 中披露，不能仅藏在讨论附注中。指定 translation/crop 扰动失败足以否定对这两类扰动的鲁棒性主张，但不应扩张成“所有 spatial robustness 都失败”。

`STAGE_D_HOLM_NOT_APPLICABLE_CONDITIONALLY`：Stage D 只有一个 primary，且在不输出 secondary p-values、不做 secondary winner selection 或正式 secondary inference 的条件下，新研究记录 `holm_status=NOT_APPLICABLE_NO_SECONDARY_INFERENCE`。这个 Stage D 状态绝不改变 C10 的 `HOLM_BLOCKED_UNDERSPECIFIED`。若未来增加 secondary inference，必须先另行冻结 exact family、假设、方向、检验统计、raw p-value 算法与 multiplicity correction。

## 6. Claim allowlist

D0 当前允许机器记录发出的 claim IDs 只有：

- `C10_PRIMARY_DECISION_NO_GO_NOVELTY`
- `C10_ACCEPTANCE_BLOCKED`
- `C10_HOLM_BLOCKER_UNDERSPECIFIED`
- `C10_SPATIAL_OVERALL_FAIL`
- `C10_SPATIAL_COMPONENTS_MIXED`
- `C10_DOWNSTREAM_NOT_AUTHORIZED`
- `C10_PRIMARY_FIDELITY_COMPONENT_VALID`
- `C10_PRIMARY_FIDELITY_COMPONENT_EFFECT_PASS`
- `STAGE_D_IS_PROSPECTIVE_REPLICATION_EXTENSION`
- `NO_INTERMEDIATE_C_PLUS_REQUIRED`
- `STAGE_D_PRIMARY_SCOPE_DEFINED`
- `FINE_ENDPOINT_IS_4X4_CELL`
- `STAGE_D_SPATIAL_NOT_IN_SCOPE`
- `STAGE_D_HOLM_NOT_APPLICABLE_CONDITIONALLY`
- `APPROVED_TO_PREPARE_STAGE_D`

任何 Stage D pass/fail/inconclusive、validity 或 cost 结果都属于未来 conditional claim，D0 没有发出。

## 7. Claim denylist

D0 及后续论文必须拒绝下列 claim IDs；未来若新协议真的产生不同范围证据，也不能借它们回写 C10：

- `C10_GO`、`C10_ACCEPTANCE_PASS`、`C10_DOWNSTREAM_AUTHORIZED`、`C10_STAGE_D_STARTED`；
- `C10_HOLM_BLOCKER_RESOLVED_OR_CANCELLED`、`C10_REANALYZED_AS_FIDELITY_ONLY`；
- `STAGE_D_EXECUTION_AUTHORIZED_AT_D0`、`STAGE_D_ANALYSIS_AUTHORIZED_AT_D0`；
- `STAGE_D_EFFECT_RESULT_AVAILABLE_AT_D0`、`STAGE_D_VALIDITY_RESULT_AVAILABLE_AT_D0`、`STAGE_D_COST_RESULT_AVAILABLE_AT_D0`；
- `RESERVE_ABSOLUTELY_NEVER_VIEWED`；
- `SPATIAL_ROBUSTNESS_PROVEN`、`TRANSLATION_ROBUSTNESS_PROVEN`、`CROP_SCALE_ROBUSTNESS_PROVEN`、`ALL_SPATIAL_ROBUSTNESS_FAILED`；
- `NATIVE_SINGLE_PIXEL_ATTRIBUTION`；
- `COMPLETE_QKV_ATTRIBUTION`、`COMPLETE_ATTENTION_EXPLANATION`、`CLASS_LOGIT_EXPLANATION`；
- `STATISTICAL_EQUIVALENCE`、`UNIVERSAL_IMAGENET_SUPERIORITY`；
- `C10_STAGE_D_POOLED_N515`；
- `MISCLASSIFIED_SAMPLES_EXCLUDED`、`COMPLETE_CASE_EFFECT_ANALYSIS`、`FAILED_SAMPLES_IMPUTED_OR_REPLACED`；
- `POSTHOC_PRIMARY_HEAD_SELECTION`、`SECONDARY_INFERENCE_WITHOUT_MULTIPLICITY_CONTRACT`；
- `DISPLAY_ARTIFACT_USED_AS_QUANTITATIVE_INPUT`；
- `C10_PASS_COST_AUTHORIZES_STAGE_D_COST`；
- `STAGE_D_SPATIAL_RESULT`、`PATCH_SCALE_OR_OTHER_METHOD_AS_STAGE_D_PRIMARY`。

不得把“无显著差异”写成等价；Stage C/C10 没有注册 equivalence margin/TOST。不得把 fixed-attention Value-path 结果写成完整 Q/K/V、完整 attention 或 class-logit attribution。

## 8. Mandatory disclosures

机器合同冻结以下披露 ID；论文文本可编辑措辞，但不能遗漏事实或要求位置：

- `DISCLOSE_POST_C10_SCOPE_CHANGE`：Study Design、Methods；
- `DISCLOSE_C10_COMPOSITE_STATUS`：Study Design、Results；
- `DISCLOSE_C10_FIDELITY_BOUNDARY`：Methods、Results；
- `DISCLOSE_C10_SPATIAL_MIXED_RESULT`：Study Design、Results、Limitations；
- `DISCLOSE_NO_GENERAL_SPATIAL_CLAIM`：Limitations，并在结论提 fidelity 时同步限定；
- `DISCLOSE_4X4_NOT_NATIVE_PIXEL`：Methods、Results 或图注；
- `DISCLOSE_VALUE_PATH_TARGET_LIMIT`：Methods、Limitations；
- `DISCLOSE_COHORT_SEPARATION`：Methods、Results、tables；
- `DISCLOSE_RESERVE_LEDGER_SCOPE`：registry、Study Design；
- `DISCLOSE_STAGE_D_NO_SPATIAL_ENDPOINT`：Methods；
- `DISCLOSE_HOLM_SEPARATION`：Methods、Results；
- `DISCLOSE_MISCLASSIFICATION_POLICY`：Methods；
- `DISCLOSE_FAILURE_POLICY`：Methods；
- `DISCLOSE_COST_EVIDENCE_BOUNDARY`：Methods、Supplement。

误分类样本必须保留在 fidelity 分母，只可作描述性分层。required-primary 的 missing、failed、non-finite、direction degeneration 或 invalidating NA 触发 validity failure；禁止 complete-case analysis、补零、换样或按结果选择性重跑。

## 9. D0 数据访问与产出声明

### 9.1 不可删除的 attempt history

历史失败没有被改写。首次尝试 `D0_INITIAL_ATTEMPT_20260825` 为 `BLOCKED`：它运行了超出 D0 门槛的广泛回归测试，实际发生 `model_loaded=true`、`model_inference_performed=true`、`model_or_gpu_used=true`、`non_reserve_jpeg_bytes_accessed=true` 和 `jpeg_decode_performed=true`，对应 blocker 为 `D0_NO_MODEL_OR_INFERENCE_GATE_VIOLATED` 与 `D0_NO_JPEG_DECODE_GATE_VIOLATED`。该尝试未用 GPU，且没有读取 reserve metadata 内容、reserve 图像字节或 reserve outcome；它不签发任何授权。

当前尝试 `D0_MODEL_FREE_RERUN_20260825` 是独立重试，并且是当前 D0 状态和准备授权的唯一依据。它采用 closed node-id allowlist、禁用 pytest plugin autoload、清空 `CUDA_VISIBLE_DEVICES`，并在 collection 前加载 `tests/d0_guard/sitecustomize.py`；guard 拒绝模型/图像栈导入、图像与模型文件访问、reserve tree 访问、GPU device 访问和非 Git 子进程。全量 regression suite 明确不在本次允许范围。

### 9.2 当前合规重试的访问声明

以下字段只描述 `D0_MODEL_FREE_RERUN_20260825`，是 task owner 对当前重试行为的审计声明。closed schema、canonical digest、Git diff 和测试只能保证这份声明的内容被固定并可检测漂移，不能独立证明当前或历史进程从未执行某个动作；后续 reserve 历史访问范围仍必须由 D1 封闭的 prior-access ledger/attestation 支持。

- `reserve_access_state=METADATA_EXISTENCE_ONLY`；仅确认路线文档已声明的候选 metadata 文件路径存在，没有打开候选 reserve 或 access-log 内容；
- `reserve_image_bytes_accessed=false`；
- `reserve_result_accessed=false`；
- `model_loaded=false`、`model_inference_performed=false`、`model_or_gpu_used=false`、`gpu_used=false`；
- `non_reserve_jpeg_bytes_accessed=false`、`jpeg_decode_performed=false`；
- `stage_d_outcome_available=false`；
- `c9_evidence_read=false`、`c9_evidence_modified=false`；
- `c10_sealed_package_read_for_integrity=true`、`c10_scientific_results_recomputed=false`、`c10_evidence_modified=false`；
- 没有生成 paper data 或 figure。

本决议的唯一授权结果是 `APPROVED_TO_PREPARE_STAGE_D`。下一任务只允许 D1 metadata-only registry/ledger/audit-ID formalization；不得读取 JPEG bytes、解码图像、运行模型/GPU、访问 outcome、实现 D2+，或签发 execution authorization。

## 10. 机器记录的结构化文档投影

下面的受控 JSON 块覆盖 authorization、全部 C10 历史状态、C10 binding、scientific basis、Stage D scope、claim sets、mandatory disclosures 和 access attestation。测试要求块内投影与权威机器记录确定性生成的投影完全相等，因此块内漂移会被拒绝；块外叙述由 required-fragment 检查和人工审阅共同约束，不能把它当成任意自然语言矛盾的完整语义证明。

<!-- STAGE_D_SCOPE_PROJECTION_BEGIN -->
```json
{
  "access_attestation": {
    "attestation_scope": "CURRENT_D0_MODEL_FREE_RERUN_ONLY",
    "c10_evidence_modified": false,
    "c10_scientific_results_recomputed": false,
    "c10_sealed_package_read_for_integrity": true,
    "c9_evidence_modified": false,
    "c9_evidence_read": false,
    "gpu_used": false,
    "jpeg_decode_performed": false,
    "model_inference_performed": false,
    "model_loaded": false,
    "model_or_gpu_used": false,
    "non_reserve_jpeg_bytes_accessed": false,
    "paper_data_or_figures_generated": false,
    "reserve_access_state": "METADATA_EXISTENCE_ONLY",
    "reserve_image_bytes_accessed": false,
    "reserve_metadata_contents_accessed": false,
    "reserve_result_accessed": false,
    "stage_d_outcome_available": false
  },
  "allowed_claim_ids_digest": "610c88e0ab6428aa366841e057844c8f0e6ff5035c11a1565f5e269670a17102",
  "attempt_history": {
    "current_attempt_id": "D0_MODEL_FREE_RERUN_20260825",
    "records": [
      {
        "access_facts": {
          "gpu_used": false,
          "jpeg_decode_performed": true,
          "model_inference_performed": true,
          "model_loaded": true,
          "model_or_gpu_used": true,
          "non_reserve_jpeg_bytes_accessed": true,
          "reserve_image_bytes_accessed": false,
          "reserve_metadata_contents_accessed": false,
          "reserve_result_accessed": false
        },
        "attempt_id": "D0_INITIAL_ATTEMPT_20260825",
        "authoritative_for_current_authorization": false,
        "blocking_reasons": [
          "D0_NO_MODEL_OR_INFERENCE_GATE_VIOLATED",
          "D0_NO_JPEG_DECODE_GATE_VIOLATED"
        ],
        "status": "BLOCKED"
      },
      {
        "access_facts": {
          "gpu_used": false,
          "jpeg_decode_performed": false,
          "model_inference_performed": false,
          "model_loaded": false,
          "model_or_gpu_used": false,
          "non_reserve_jpeg_bytes_accessed": false,
          "reserve_image_bytes_accessed": false,
          "reserve_metadata_contents_accessed": false,
          "reserve_result_accessed": false
        },
        "attempt_id": "D0_MODEL_FREE_RERUN_20260825",
        "authoritative_for_current_authorization": true,
        "blocking_reasons": [],
        "status": "PASS"
      }
    ]
  },
  "authority": {
    "analysis_authorization": "NOT_EVALUATED",
    "authorized_next_task": "D1",
    "does_not_override_c10": true,
    "execution_authorization": "NOT_ISSUED",
    "preparation_authorization": "APPROVED_TO_PREPARE_STAGE_D",
    "preparation_scope": [
      "D1_METADATA_ONLY_REGISTRY_PREPARATION",
      "D2_PROTOCOL_AND_STATISTICS_FREEZE",
      "D3_TO_D6_IMPLEMENTATION_WITH_SYNTHETIC_OR_METADATA_INPUTS",
      "D7_NON_RESERVE_SMOKE_BUDGET_AND_EXECUTION_AUTHORIZATION_GATE"
    ]
  },
  "blocking_reasons": [],
  "denied_claim_ids_digest": "aefd17158558e9e4b79a4cdb07f6e7f264a890923857ea6bb4a43685487c74bc",
  "document_type": "stage_d_scope_markdown_projection",
  "future_conditional_claim_ids_digest": "34f4682738c018cd0d7287cabf7c9f08d198284989a7e16816477993ff54012c",
  "machine_record_canonical_digest": "7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f",
  "machine_record_path": "config/stage_d_scope_decision.json",
  "mandatory_disclosures_digest": "234a4fb39a00d4c6726b7d207d29dce6238d1fb3fee1d926e1851de89b33af0c",
  "schema_version": 1,
  "scientific_basis_digest": "4d78fdb177dadad4eca5ee803ab03f17648f8ac89dd693c08abe71ddb8f3d807",
  "source_c10_binding_digest": "168f39d5a004100b597d42218098036f8762ea5623d177157e40ee275277901f",
  "source_c10_historical_status": {
    "cost_status": "PASS_COST",
    "downstream_authorized": false,
    "effect_status": "PASS_EFFECT",
    "equivalence_test_performed": false,
    "holm_performed": false,
    "primary_decision": "NO_GO_NOVELTY",
    "spatial_status": "FAIL_SPATIAL",
    "stage_c_acceptance": "BLOCKED",
    "stage_d_started": false,
    "validity_status": "PASS_VALIDITY",
    "workflow_blocker": "HOLM_BLOCKED_UNDERSPECIFIED"
  },
  "stage_d_scope_digest": "f59eebee428f770c1e7e6fc33ace0e28647ed8d2a5028e066ff3ae62edbd7738",
  "task_status": "PASS",
  "verification_policy": {
    "allowed_pytest_node_ids": [
      "tests/test_stage_d_scope_contract.py::test_scope_schema_is_closed_draft_2020_12",
      "tests/test_stage_d_scope_contract.py::test_checked_in_scope_decision_matches_schema_and_semantics",
      "tests/test_stage_d_scope_contract.py::test_scope_import_does_not_load_model_or_torch_stack",
      "tests/test_stage_d_scope_contract.py::test_scope_document_and_machine_record_agree",
      "tests/test_stage_d_scope_contract.py::test_scope_markdown_projection_tamper_is_rejected",
      "tests/test_stage_d_scope_contract.py::test_scope_rejects_schema_and_authorization_drift",
      "tests/test_stage_d_scope_contract.py::test_scope_rejects_attempt_history_or_allowlist_tamper",
      "tests/test_stage_d_scope_contract.py::test_scope_rejects_unsigned_tamper",
      "tests/test_stage_d_scope_contract.py::test_scope_rejects_resigned_semantic_tamper",
      "tests/test_stage_d_scope_contract.py::test_strict_loader_rejects_duplicate_and_nonfinite_json",
      "tests/test_stage_d_scope_contract.py::test_claim_sets_and_access_attestation_are_exact",
      "tests/test_stage_d_scope_contract.py::test_d0_static_files_keep_execution_and_reserve_closed"
    ],
    "allowlist_closed": true,
    "cuda_visible_devices": "",
    "full_regression_suite_allowed": false,
    "model_and_image_stack_imports_forbidden": [
      "PIL",
      "cv2",
      "timm",
      "torch",
      "torchvision",
      "vitnns.modeling"
    ],
    "profile_id": "D0_MODEL_FREE_NO_JPEG_V1",
    "pytest_plugin_autoload": false,
    "runner_command_template": "VITNNS_D0_STRICT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests/d0_guard:src /data00/home/yeziqi.831/miniforge3/envs/vitnns/bin/python -m pytest -ra <allowed_pytest_node_ids>",
    "safety_guard_path": "tests/d0_guard/sitecustomize.py",
    "sealed_c10_validation_mode": "BYTE_HASH_CANONICAL_MANIFEST_STATUS_INVENTORY_AND_GIT_REF_ONLY",
    "source_scientific_recomputation_allowed": false
  }
}
```
<!-- STAGE_D_SCOPE_PROJECTION_END -->
