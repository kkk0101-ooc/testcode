# NNS-ViT P0-A/B STATIC PACKAGE 执行提示词

将下面提示词完整交给执行者。

---

你现在位于：

```text
/data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild
```

执行 `NNS_VIT_CUT_INTERVENTION_PREDICTION_V1` 的 **P0-A/B STATIC PACKAGE**。这是静态协议、schema 和验证器实现任务，不是模型实验。必须完成、验证、提交并推送本阶段，然后停止。

## 1. 权威输入与目标

权威规划：

```text
docs/NNS_ViT完整切面归因与匹配干预研究规划_20260917.md
```

绑定身份：

```text
planning_authority_commit = 7f169c7ee8d4acf5a6274f93a8340310dd1a53de
planning_document_sha256 = fa1fd7884c9fe64a6dd49450202a6135f66b1c999b3c5d2620df50b7881ca15c
branch = stage-d-d3-pass
protocol_id = NNS_VIT_CUT_INTERVENTION_PREDICTION_V1
stage_id = P0-STATIC
authorization = P0_A_B_STATIC_PACKAGE_ONLY
```

HEAD 可以晚于 `planning_authority_commit`，但该 commit 必须是 HEAD 祖先；从该 commit 读取的权威规划文件 SHA-256 必须等于上述 digest。当前工作树中的同名规划文件也必须与该版本一致，否则停止为 `BLOCKED_PROTOCOL_DRIFT`。

本阶段目标是创建一套可由 Python 标准库独立验证的静态合同包，并在全部检查通过后签发：

```text
P0_STATIC_PASS_AUTHORIZES_P0_C_ENGINEERING_RESOURCE_PILOT_ONLY
```

该 token 只授权 P0-C 使用 synthetic fixtures 和一张已登记 prior-access 图像进行工程/资源 pilot。它不是完整 `P0 PASS`，不授权 cohort seal、独立图像读取、正式归因或正式 intervention。

## 2. 启动门禁

开始写文件前完成并记录：

1. `git rev-parse HEAD`；
2. `git rev-parse @{upstream}`；
3. `git ls-remote origin refs/heads/stage-d-d3-pass`；
4. `git status --branch --porcelain=v2`；
5. `git lfs status`；
6. `git merge-base --is-ancestor 7f169c7ee8d4acf5a6274f93a8340310dd1a53de HEAD`；
7. 从 commit `7f169c7ee...` 重算规划文档 SHA-256；
8. 检查目标输出根尚未存在。

要求：

- HEAD、tracking ref 和远端 branch 必须相同；
- index 必须干净；
- 用户现有未跟踪文件允许存在，但不得纳入本次提交；
- LFS 必须无待提交或待推送对象；
- 规划 commit 祖先检查和文档 digest 必须通过。

任一门禁失败，生成最小 `BLOCKED_STARTUP_STATE` 记录并停止，不得继续创建看似 PASS 的静态包。

## 3. P0-A/B 访问边界

本阶段明确采用：

```text
source-code access != empirical-outcome access
```

允许读取：

- 本仓库源代码、测试、配置、schema、协议和文档；
- Git 历史、refs、LFS metadata；
- 已有 JSON/CSV 的 schema、字段名、路径、hash、shape、dtype、计数和 provenance metadata；
- 已公开的模型名称、checkpoint hash、preprocessing metadata；
- 已封存完成状态和资源 telemetry；
- synthetic fixtures。

禁止读取或执行：

- JPEG/PNG 像素解码；
- `.npy`、`.npz`、tensor、map、activation 或 intervention 数值；
- 新 cohort 的 prediction、logit、attribution、effect 或任何 outcome；
- checkpoint tensor 反序列化和模型加载；
- `torch`、`torchvision`、`timm`、`numpy`、`PIL`、`cv2`、`cupy` 导入；
- CUDA、GPU、`nvidia-smi` 或 accelerator probing；
- P0-C、P1、P2、P3、P4；
- 新的 cohort membership、$y^*$ 数值或 20 个 source-object ID；
- 对 R9、Stage X/Y/S 或其他 sealed root 的写入。

不要用 `rg`、`find`、脚本或测试读取禁止范围中的 numeric payload。访问审计必须逐项证明零次违规访问。

## 4. 必须创建的文件

### 4.1 配置与 closed schemas

```text
config/nns_vit_cut_intervention/p0_static.json
config/nns_vit_cut_intervention/p0_static.schema.json
config/nns_vit_cut_intervention/artifact.schema.json
```

所有 object，包括所有 nested object，必须 `additionalProperties: false`。禁止使用开放的 `payload: {"type": "object"}`。可以使用 `$defs`，但每个 document type 的 payload 都必须有 closed schema。

### 4.2 机器注册表

```text
manifests/nns_vit_cut_intervention/target_registry.json
manifests/nns_vit_cut_intervention/preprocessing_registry.json
manifests/nns_vit_cut_intervention/cut_intervention_registry.json
manifests/nns_vit_cut_intervention/baseline_registry.json
manifests/nns_vit_cut_intervention/statistical_analysis_plan.json
manifests/nns_vit_cut_intervention/failure_registry.json
manifests/nns_vit_cut_intervention/artifact_contract.json
manifests/nns_vit_cut_intervention/p0_c_cost_pilot_plan.json
manifests/nns_vit_cut_intervention/schema_registry.json
```

### 4.3 验证器与测试

```text
scripts/validate_nns_vit_cut_intervention_p0_static.py
tests/test_nns_vit_cut_intervention_p0_static.py
```

验证器必须只使用 Python 标准库。源文件 AST 中出现任何禁止 import 都必须失败。验证器必须：

- 严格 UTF-8 JSON 解析；
- 拒绝 duplicate keys、NaN 和 Infinity；
- 重算 canonical digest；
- 重算文件 SHA-256 和字节数；
- 拒绝 symlink 和 root escape；
- 验证 closed schemas；
- 验证所有跨文件 ID、枚举、计数、公式字符串、门槛和 next authorization；
- 验证 source bundle；
- 支持 `draft`、`final` 和 `seal`；
- seal 时使用 pending → atomic replace；
- 对已有 terminal root 禁止原地重写。

### 4.4 Evidence 与报告根

```text
evidence/nns_vit_cut_intervention/p0_static_20260917/formal/
reports/nns_vit_cut_intervention_p0_static_20260917/
```

Evidence 根至少包含：

```text
source_binding.json
access_audit.json
contract_bundle.json
false_pass_audit.json
validation_report.json
whole_root_inventory.json
completion/p0_static.pending.json
completion/p0_static.json
```

报告根至少包含：

```text
report.md
validation_report.json
```

### 4.5 Handoff 文档

```text
docs/nns_vit_cut_intervention_p0_static_handoff_20260917.md
```

## 5. P0-A 数学合同

### 5.1 向量 component conductance

注册并验证：

$$
K^C_{k,i}
=\Delta x_i
\int_0^1
\sum_{j\in\mathcal J_k}
\frac{\partial F(x_\alpha)}{\partial z_{k,j}}
\frac{\partial z_{k,j}(x_\alpha)}{\partial x_i}
d\alpha.
$$

注册表必须声明 component axes、坐标集合、聚合 dtype 和 retained map level。

### 5.2 Head computational edge cut

切点固定为：

```text
post-attention-dropout
pre-head-concat
pre-output-projection
all 197 query-token outputs
```

必须登记 12 heads、匹配的 $W_O$ 列、attention aggregate、pre-attention identity edge 和 first-residual reconstruction。CLS-only replacement 不是主干预。

### 5.3 Residual computational edge cuts

第一次 residual：

$$
z_{l,\mathrm{id}}=h_l,\qquad
z_{l,\mathrm{attn}}=a_l(h_l),\qquad
h_l'=z_{l,\mathrm{id}}+z_{l,\mathrm{attn}}.
$$

两条输入边使用同一个 $\nabla_{h_l'}F$。

第二次 residual：

$$
z_{l,\mathrm{carry}}=h_l',\qquad
z_{l,\mathrm{mlp}}=m_l(h_l'),\qquad
h_{l+1}=z_{l,\mathrm{carry}}+z_{l,\mathrm{mlp}}.
$$

两条输入边使用同一个 $\nabla_{h_{l+1}}F$。必须明确禁止用原始共享节点的总 downstream gradient 作为 identity/carry 后再叠加 branch。

### 5.4 Attention differential-path decomposition

Routing/value 不能注册为普通 complete edge cut。应注册为 attention branch 内的 differential paths：

$$
dO=(dA)V+A(dV),
$$

$$
K^{\mathrm{attention}}\approx K^R+K^V,
\qquad
K^R\approx K^Q+K^K+K^B.
$$

若 attention bias 为输入无关常数，$K^B=0$，但 bias 仍保留在 forward score。

### 5.5 Standard interaction

固定：

$$
I_{RV}=e_R+e_V-e_{R+V}.
$$

相反符号必须被 validator/false-pass test 拒绝。

### 5.6 切面—归因—干预注册表

机器注册表至少覆盖：

| Family | Component | Mathematical type | Main intervention | Role |
|---|---|---|---|---|
| Head | all-token head output | edge cut | reference replacement | primary |
| First residual | identity | add-input edge | none | reconstruction |
| First residual | attention | add-input edge | branch reference replacement | primary |
| Second residual | carry | add-input edge | none | reconstruction |
| Second residual | MLP | add-input edge | branch reference replacement | primary |
| Value | $A\,dV$ | differential path | $V\leftarrow V(x_0)$ | primary |
| Routing | $(dA)V$ | differential path | $A\leftarrow A(x_0)$ | primary |
| Q/K | routing subpaths | differential path | Q/K replacement | secondary |

## 6. P0-B 目标、reference 与统计合同

### 6.1 固定目标算法

主目标是共享 raw class logit：

$$
F(x)=\ell_{y^*}(x).
$$

静态包只冻结 $y^*$ 的 metadata-only 确定算法，不计算 $y^*$。正确分类、prediction、logit、margin 和 confidence 不得进入 cohort membership。

### 6.2 Preprocessing-derived reference

主 reference 是 normalized input space 中的零张量。注册表必须绑定：

```text
color_mode = RGB
raw_range = [0,1]
resize_shorter_side = 248
interpolation = bicubic
antialias = true
center_crop = [224,224]
channel_order = RGB
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]
```

并声明一般关系：

$$
x_{\mathrm{norm},c}=
\frac{x_{\mathrm{raw},c}-\mu_c}{\sigma_c},
\qquad
x_0^{\mathrm{norm}}=0Longleftrightarrow x_0^{\mathrm{raw}}=\mu.
$$

0.5 必须来自当前模型配置绑定，不能作为通用 ViT 常数。

### 6.3 Cohort 算法

静态包冻结但不执行以下算法：

1. 建立所有历史 source-object 排除集合；
2. 在剩余候选中筛出 metadata 上至少有 20 张 true-label 图像的 classes；
3. 按 `SHA256(protocol_id|class_id)` 选择第一类；
4. 按 `SHA256(protocol_id|source_object_id)` 选择前 20 图；
5. 保存候选、排除和 membership digests；
6. 不足 20 张时签发 `BLOCKED_INSUFFICIENT_DISJOINT_COHORT`；
7. 失败后不补样。

### 6.4 Attribution 与 effect 对应

$$
a_k=\sum_iK_{k,i},\qquad
m_k=\sum_i|K_{k,i}|,\qquad
q_k=\frac{m_k}{\sum_{j\in C}m_j}.
$$

- $a_k$ 对 signed effect；
- $m_k/q_k$ 对 absolute effect；
- zero cut mass 记 `NA_ZERO_CUT_MASS`；
- activation replacement effect 为
  $$
  e_k=F(x)-F(x;z_k\leftarrow z_k^{\mathrm{ref}}).
  $$

### 6.5 图像级统计

- 图像是唯一独立单位；
- head Spearman 先在每个 block 的 12 heads 内计算，再取 12 blocks 的中位数；
- shallow/middle/deep 固定为 0–3/4–7/8–11；
- routing/value 每图在 144 个 block-head pairs 上计算一个 $\rho_n^R$；
- attention-only 和 MLP-only 各自在 12 blocks 上计算；
- ties 使用 average ranks；
- 常数向量记 `NA_CONSTANT_VECTOR`；
- head family 每图少于 8 个有效 blocks 时记 `NA_INSUFFICIENT_VALID_BLOCKS`；
- family 至少需要 18/20 个有效图像 summary；
- 18–20 有效时按有效图估计并报告 completion rate；
- 少于 18 时 `INVALID_INSUFFICIENT_VALID_IMAGES`。

### 6.6 Primary hypotheses 与门槛

机器化登记 H1–H5、B1–B10、R1–R3，完全复现权威规划第 10–11 节：

- effect thresholds：Spearman $\ge0.30$、direction agreement $\ge0.60$、replacement/zeroing rank consistency $\ge0.50$；
- bootstrap：20,000 次 image-level paired PCG64，linear percentile 95% interval；
- exact sign flip：最多 $2^{20}$ 个图像级符号组合；
- Holm：Family H、B、R 分开；
- practical threshold、bootstrap lower bound 和 Holm-adjusted $p<0.05$ 同时满足；
- 至少两个 families 为 `SUPPORTED_PREDICTIVE_RELATION`，且至少一个为 Head 或 Routing/value；
- supported families 的 baseline gain 必须为正；
- `INVALID` 或 `BLOCKED_INFRASTRUCTURE` 阻止聚合 GO。

### 6.7 Mandatory baselines

分别冻结：

- Head：output norm、output-change norm、gradient norm、Gradient × activation、mean CLS attention、CLS attention entropy、raw CLS attention mass；
- Branch：output norm、output-change norm、gradient norm、Gradient × activation；
- Routing/value：$\|A-A_0\|$、$\|V-V_0\|$、Q/K/V activation-change norm、对应 gradient norm、mean CLS attention、entropy。

Random ranking 只作 descriptive sanity control。最强 mandatory baseline 在每个同步 image-bootstrap replicate 中从整个 cohort statistic 取最大值，不能逐图选择。

## 7. 三类数值合同

必须分别注册，禁止合并成一个 `convergence_pass`：

1. `ALGEBRAIC_CUT_RECONSTRUCTION`：同一 quadrature order 下 component sum 对 total attribution；
2. `IG_COMPLETENESS_TO_TARGET_DELTA`：total attribution 对 $F(x)-F(x_0)$；
3. `QUADRATURE_SUCCESSIVE_ORDER_CONVERGENCE`：相邻阶数 map、absolute mass 和 signed total 的变化。

固定 Gauss–Legendre schedule：

```text
[128,256,512,1024,2048,4096,8192]
```

静态配置必须逐字段复现权威规划第 12 节的容差和 small-target 处理。8192 仍失败时只能签发对应 typed failure。

## 8. 资源与 P0-C 合同

本阶段只冻结 P0-C 计划，不执行它。Owner ceilings 固定为：

```text
aggregate_h20_gpu_hours_max = 80
eight_gpu_wall_hours_max = 24
cpu_hours_max = 500
peak_reserved_gpu_gib_per_process_max = 32
fleet_concurrent_reserved_gpu_gib_max = 256
retained_storage_gib_max = 32
temporary_storage_gib_max = 64
devices = 8
```

P0-C 必须在一张 prior-access 图和 synthetic fixtures 上测量每种操作的 calls、P95、显存和 bytes，以 `P95 × planned calls × 20 × 1.5` 投影 P2。任一上限超出则 `NO_GO_RESOURCE`。静态包不得填写伪造的实测 projection。

## 9. False-pass attacks

至少实现以下 mutation attacks，并证明 validator 全部拒绝：

1. residual identity 使用原节点总 downstream gradient，造成 double counting；
2. first residual 缺 identity 或 attention edge；
3. second residual 缺 carry 或 MLP edge；
4. head cut 使用 CLS-only replacement；
5. head hook 位于 dropout 前、concat 后或 output projection 后；
6. routing/value 被标成普通 complete edge cut；
7. routing 或 Value 漏项仍标 reconstruction PASS；
8. Q 或 K 漏项仍标 routing closure PASS；
9. interaction 使用 `e_R+V-e_R-e_V`；
10. normalized-zero 被无条件写成 RGB 0.5；
11. preprocessing mean/std、crop、interpolation 或 antialias 漂移；
12. algebraic reconstruction 冒充 quadrature convergence；
13. low-order map 与高阶 map 漂移但仍 PASS；
14. small target delta 被除法相对误差掩盖；
15. 低于 18/20 仍签发 supported；
16. 144 heads 或 block-head rows 被当作独立 $n$；
17. undefined Spearman 被静默删除或补零；
18. direction agreement 用 absolute mass 预测符号；
19. 每图挑选最强 baseline；
20. random ranking 进入 mandatory baseline set；
21. Family B 只对表现更好的 branch 做 Holm；
22. supported family baseline gain 非正仍聚合 GO；
23. 只一个 family PASS 仍聚合 GO；
24. resource projection 缺失或超 cap 仍授权 P0-C/P1；
25. open nested schema 或额外未知字段通过；
26. duplicate JSON key、NaN、Infinity、symlink 或 root escape 通过；
27. completion 缺 source binding、access audit、false-pass audit 或 inventory 仍 PASS；
28. P0 static completion 错误授权 P1 cohort seal 或 GPU production。

False-pass 测试使用临时目录和 synthetic JSON，不修改正式静态包。

## 10. 验收条件

P0-A/B STATIC PACKAGE 仅在以下条件全部满足时 PASS：

- startup gate 全部通过；
- 所有配置和 registries 通过 closed schemas；
- canonical digests、SHA-256、bundle digest 和 cross-file references 重算一致；
- 权威规划 commit/hash 绑定一致；
- 数学、edge-cut、differential-path、interaction、preprocessing、统计和资源合同全部匹配；
- standard-library validator 自检通过；
- focused tests 全部通过；
- 28 类 false-pass attacks 全部被拒绝；
- access audit 确认零 JPEG/PNG decode、零 NPY/NPZ numeric read、零 model/checkpoint load、零 Torch/timm/Numpy/PIL import、零 CUDA/GPU access；
- evidence root inventory 完整且 completion 原子发布；
- Git commit、tracking ref、remote branch 和 LFS 状态一致。

PASS completion 必须写：

```text
status = PASS_STATIC
next_authorization = P0_C_ENGINEERING_RESOURCE_PILOT_ONLY
p0_terminal_pass = false
p1_cohort_seal_authorized = false
independent_cohort_outcome_access_authorized = false
gpu_production_authorized = false
```

任一关键检查失败时签发 typed `BLOCKED_*`，`next_authorization=NONE`。

## 11. 执行顺序

1. 运行启动门禁并保存 `startup_gate.json`。
2. 读取权威规划、必要源代码、schemas 和 metadata contracts。
3. 创建 config、registries 和所有 closed schemas。
4. 实现标准库 validator。
5. 实现 synthetic fixtures 和 mutation-based false-pass tests。
6. 运行 focused tests。
7. 运行 validator `draft`；修复任何失败。
8. 生成 source binding、access audit、contract bundle、false-pass audit 和报告。
9. 运行 validator `final`。
10. 若全部通过，原子生成 `completion/p0_static.json` 和 whole-root inventory。
11. 再次运行 source-recomputing validator，证明 seal 后仍 PASS。
12. 更新 handoff 文档，明确下一 token 仅为 P0-C。
13. 仅暂存本阶段文件；检查不包含用户已有无关改动。
14. commit，建议消息：

```text
Freeze NNS-ViT P0 static contracts
```

15. push 当前 branch。
16. 验证 local HEAD、tracking ref、remote branch、porcelain v2 和 Git LFS 状态。
17. 停止，不执行 P0-C。

## 12. 最终报告格式

最终回复必须报告：

- P0-A/B 状态；
- 生成的 config、registries、schemas、validator、tests 和 evidence 根；
- false-pass attacks 通过数；
- 访问审计；
- completion token 与唯一 next authorization；
- 未执行的 P0-C/P1–P4；
- commit SHA 与远端包含性；
- 任何仍存在的 blocker。

不要报告尚未计算的科学结果，不要把 `PASS_STATIC` 写成完整 `P0 PASS`。

---
