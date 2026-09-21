# Endpoint–PCIA–PEFA 步骤 1 资产清单

- 状态：`FROZEN_STEP1_20260921`
- 核验时间：2026-09-21（PRC）
- 仓库根：`/data01/home/yeziqi.831/nxtrec_wksp/testcode`
- 核验时分支：`stage-d-d3-pass`
- 核验时 HEAD：`944f54c99c8d83803d461224565b93a779e75915`

本清单只建立步骤 1 的来源边界，不把历史 Endpoint 结果改名为 PCIA/PEFA 结果，也不授权步骤 2 计算。哈希均在本轮从文件字节重新计算，而非复制报告字段。
任务源 `vit_pcia/执行规划.md` 在冻结时为 14925 bytes，SHA-256 为 `03ae681d9ddcadd7ab07b13592e92463775cfa55f81a914df68164689132809d`；本轮未修改该文件。

## 1. 可继承内容与禁止继承内容

| 类别 | 冻结决定 |
|---|---|
| Endpoint 数学身份 | 可继承：参考点为零时是同一内部标量目标上的 Gradient × Input |
| 一阶真实 Jacobian 与 ViT 结构接口 | 可继承为实现起点，但须在新的 PCIA/PEFA source binding 中重新登记 |
| residual、heads、Q/K/V、routing/value 的旧切面与负控制 | 可继承为一阶候选 registry；步骤 4 仍须按 consumer/slot/invocation 重新证明合法 |
| 模型、预处理、cohort 和已物化 payload | 可按本清单哈希继承；不得替换或根据旧 outcome 筛选 |
| 旧 Endpoint 数值图、C4a/C4b、Path IG 结果 | 仅是历史暴露和边界证据；**禁止**标记为 PCIA、$T$、$R$ 或 $\Omega$ 结果 |
| 旧 EP0 source binding | 保留为当时快照；当前源码以后续授权 EP1 的哈希为准，不能静默沿用 EP0 旧哈希 |
| 旧 Path IG 校准 | 可继承风险事实：ViT-B 在冻结最大 $M=512$ 未过旧门；不可改写为 PEFA 数值结论 |
| 旧 Endpoint utility | C4a/C4b 最终均 `NOT_SUPPORTED`；不可用来预选 PEFA 样本或主张其效用 |

## 2. 来源提交与血缘

| 角色 | commit | 本轮验证 |
|---|---|---|
| Endpoint 核心 scanner 初始实现 | `4d3a0a4bab5d72e9ef0766ce2d035d4489d3e792` | 是当前 HEAD 祖先 |
| EP1 权威实现/回归 | `db342d6015b971f9015e0302030f4547ddfa4200` | 是当前 HEAD 祖先；当前核心源码与 EP1 report 哈希一致 |
| E3 正确性生产 | `7d930595d5d3956383c7ead821c08ca3206aa597` | 是当前 HEAD 祖先 |
| E3 R2 handoff | `e00a38c5cdc455b9797dd71f751c3fd40f27563c` | terminal 内绑定；本轮未把它当当前源码版本 |
| E6-R3 统计/论文发布 | `fa670bdc86f1b6344fd193f9c4117e6f55edb9ac` | 是当前 HEAD 祖先 |
| PCIA/PEFA 步骤 1 工作树 | `944f54c99c8d83803d461224565b93a779e75915` + `vit_pcia/` 未跟踪文件 | 尚未提交；这是合同冻结时的精确状态 |

说明：EP0 `source_bindings.json` 中 `endpoint_nns.py` 和 `endpoint_vit.py` 的旧 SHA-256 分别为 `b885...df8` 与 `eb61...ab6`。当前文件哈希变化可由后续 EP1 提交与 EP1 validation report 解释，当前值逐字节匹配 EP1 的 implementation bindings，故判为 `AUTHORIZED_PROVENANCE_ADVANCE`，不是 silent drift。

## 3. 代码与合同字节绑定

| 路径（相对 `vitnns_rebuild/`） | bytes | SHA-256 | 角色 |
|---|---:|---|---|
| `src/vitnns/endpoint_nns.py` | 10399 | `143226aba89d9b975b34ba014ee7dbba1adaf651c3cdaae0d43f4fd08c20cd4e` | Endpoint 数学与 CNN fixture |
| `src/vitnns/endpoint_vit.py` | 26224 | `66a32cb7a236dcca25c916eaddc775afae516367f00ea0da91aa6585eb1ca4c7` | ViT target 与一阶结构切面 |
| `tests/test_endpoint_nns.py` | 9814 | `f6f7660da30234a43210954465375ae5721d8249ff4cb427951be9ac136a6fd8` | 一阶回归 |
| `tests/test_endpoint_vit.py` | 14657 | `0b6ff2c7139aa04b1ea7f36e788e544795d70791cb8af892afdd25e734805f9d` | ViT/切面回归 |
| `tests/test_endpoint_ep1_queue.py` | 4706 | `89fdf610d2625fc8f7dd201754fed32e2db8660fcf980d9559ad023cbf691935` | 队列与恢复回归 |
| `manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json` | 446993 | `4d1e88d946c11af3fce9ca05c139007dc0991768ea224e2067ca33b337a7e0f2` | 528 行冻结 cohort |
| `manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json` | 185629 | `c2f8a5075d1bae28ae122ad9190731a675cc21774f3eb0f5b6a37c9019c2b626` | 目标注册 |
| `manifests/nns_vit_endpoint/ep0_r1_20260918/implementation_contract.json` | 2127 | `dde628560d1d453e11bff3b09c48a2321db5354301928d7bcde7066a89821bef` | 历史精度/回归合同 |
| `manifests/nns_vit_endpoint/ep0_r1_20260918/resource_contract.json` | 3727 | `c38d30d91e822c424fbc8e18a35595680133966414478d4c4605e7f074da9476` | 历史资源合同 |
| `manifests/nns_vit_endpoint/ep0_r1_20260918/source_bindings.json` | 5564 | `adf261134c0cf72035a12f3c66be5aebbfb446a102232a4beacb03205d69732b` | EP0 历史来源快照 |
| `manifests/nns_vit_endpoint/ep0_r1_20260918/literature_registry.json` | 33785 | `c2551b5204de027693ec0bc24308d5010cbc2a1d52a9178c7d52b4a6486353ec` | EP0 文献来源快照与本轮项目内差集复核入口 |

步骤 2/4 不得直接修改这些历史文件。新实现须使用 `vit_pcia/` 下新路径或明确的新 namespace，并建立新的绑定。

## 4. 模型、数据与 payload 复核

| 资产 | bytes | 本轮重算 SHA-256 / digest | 结果 |
|---|---:|---|---|
| ViT-B checkpoint `assets/models/vit_base_patch16_224_weights.semantic_candidate.pth` | 346329934 | `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d` | 与 EP0 registry 一致 |
| DeiT-S checkpoint `assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth` | 88251143 | `cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f` | 与 EP0 registry 一致 |
| ImageNet validation tar `assets/imagenet/ILSVRC2012_img_val.tar` | 6744924160 | `c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0` | 与历史校准记录一致 |
| 64 calibration cache | 11445059 total | canonical path/hash binding `2be0ba262f2ac569f9548d4c2ecaaaaba82eeb4c341bb78d0b1398f4a56996e8` | 64/64 present，0 mismatch |
| 400 evaluation cache | 53952422 total | canonical path/hash binding `20a1b32d01cf84ee9a33368aadfa1355e33c32f2c739aee629f958c6340594fa` | 400/400 present，0 mismatch |

canonical binding 的计算是按 UTF-8 路径排序后连接 `path + NUL + expected_sha256 + LF` 再取 SHA-256。calibration 期望值来自 `nns_vit_endpoint_ep2_r1_20260918/task_manifest.json`；evaluation 期望值来自 `nns_vit_endpoint_e5_spatial_r1_20260918/task_manifest.json`。两份 manifest 本轮哈希分别为：

- EP2 task manifest：`3621a06a7dbd6fc921181ebcdb64eec27f0eeafeb491e09e9b5a41a1eb6c46e9`；
- E5 task manifest：`5ed2cdcfd9dee2fa5e206706a0864b91aaa2e77c7e7331a3cb10fb5ee7634d28`。

discovery split 的 64 行在 EP0 registry 中已冻结，但未发现专属 Endpoint materialization manifest。它们只能在步骤 4 启动时从上述官方 tar 按 member name 抽取并逐文件绑定；缺失或重复 payload 时失败，不自动替换。部分 discovery ID 后来出现在其他研究计划中，因此步骤 4 不可声称其 outcome-naive，只能把它们用于实现/成本诊断。

## 5. 历史证据锚点

| 证据 | bytes | SHA-256 | 可支持的陈述 | 不可支持的陈述 |
|---|---:|---|---|---|
| `docs/NNS_ViT_Endpoint_Scanner_EP0协议冻结_20260918.md` | 15079 | `4c133a5a5b0c3dfcf86dbbf1d1bb908d5bf78aaea89375feaf6944be9279d7b9` | Endpoint 定义、模型/数据/目标/旧切面合同 | PEFA 已验证 |
| `reports/nns_vit_endpoint_ep1_r11_20260918/terminal_receipt.json` | 459 | `cc81f3ac26943ff742cc4230f30fbc15f027f9f5057db71abedb5944649528bb` | EP1 实现与回归 PASS | 二阶图正确 |
| `reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json` | 6840 | `c855d2302a90b1b419b0e040a6ca13a3ceeae27bbe42c4caf5605b74af5c1b1a` | 当前核心源码哈希、44 CPU tests、一阶切面 closure | $T/R/\Omega$ closure |
| `reports/nns_vit_endpoint_e3_closure_r2_20260918/artifact_manifest.json` | 267 | `9868bbbe0b96e7508b771fd9ff3e580ce0ac4b12538a355f6c9ba689ae1c410f` | E3-R2 封存入口 | 新框架资产 manifest |
| `reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json` | 1518 | `6261e12a6837ef8e931ca32bf3faab2689c2e4aa8b26d9fc7e20b947b2e7d082` | 旧一阶结构正确性验证 | 二阶实现正确 |
| `reports/nns_vit_endpoint_e3_closure_r2_20260918/terminal_receipt.json` | 682 | `675cf4c6bec7dd681497f57bbd31b638dece7a5dfe0abc2b99e921bdbfbfe8ac` | `CORRECTNESS_ONLY` PASS | utility 或 PEFA PASS |
| `reports/nns_vit_endpoint_e6_release_r3_20260918/validation_report.json` | 2403 | `53925d742b6a070a1bb4dff253237b64299a015a9fb131199ea60d997f4a515f` | Endpoint 主轨最终统计包验证 | PEFA 新颖性 |
| `reports/nns_vit_endpoint_e6_release_r3_20260918/terminal_receipt.json` | 550 | `50342a81e2acfa98e155671858d3fa675a7d8eca8868b94ab7890e9bd5287f9e` | Endpoint 主轨完成，C4a/C4b 均未获支持 | 新框架效用 |
| `../创新思想v1.md` | 9529 | `4252a0fa9c26229e875a36a72ee540d867550b0bc23da1dc8cfdc7f6734166d2` | 候选定义和初步碰撞审计 | 完成后的查新或冻结合同 |

## 6. 当前实现覆盖缺口

现有 `endpoint_vit.py` 的 `EndpointTarget` 以模块路径和 input/output 位置标识 hook，并对同一模块重复执行直接报错；它已能捕获 residual、heads、Q/K/V 等一阶分量，但尚未实现本合同所需的通用 `(producer tensor, consumer operator, input-slot, invocation)` use-port registry。现有 `decompose_endpoint_vit` 明确把下游 cotangent `.detach()` 后做一阶 VJP；这对历史 Endpoint 切面是正确语义，却会截断 $R/\Omega$ 所需的二阶依赖。

因此步骤 2/4 必须新建二阶实现，不能把现有一阶函数原样复用并声称 PEFA 已覆盖。旧测试验证的是 Endpoint 一阶合同，不覆盖新对象。

## 7. 工作树隔离

冻结时仓库存在与本目标无关的未跟踪文件：`vitnns_rebuild/config/nns_vit_cut_intervention/p0_c.json`、对应 manifest/source/worker/validator/test，以及根目录 `创新思想v1.md`。这些文件均未修改、未暂存。`vit_pcia/` 本身为新未跟踪目录；步骤 1 只在该目录创建交付物。

## 8. 资产门结论

```text
ENDPOINT_PROVENANCE=PASS_WITH_AUTHORIZED_EP1_ADVANCE
CHECKPOINT_BYTES=2_OF_2_MATCH
IMAGENET_CONTAINER=PASS_SHA256_RECOMPUTED
CALIBRATION_PAYLOADS=64_OF_64_MATCH
EVALUATION_PAYLOADS=400_OF_400_MATCH
DISCOVERY_PAYLOADS=DEFERRED_TO_STEP4_MATERIALIZATION
OLD_RESULTS_RELABELED_AS_PEFA=false
STEP2_ASSET_ELIGIBILITY=PASS_CPU_FIXTURES_ONLY
STEP4_ASSET_ELIGIBILITY=CONDITIONAL_ON_NEW_SOURCE_BINDING_AND_DISCOVERY_MATERIALIZATION
```
