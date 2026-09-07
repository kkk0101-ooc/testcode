# Stage F F3C1 implementation and synthetic preflight 执行提示词（2026-09-07）

请在以下仓库中完整执行 Stage F 的 F3C1，不要只做规划、预检说明或部分实现：

`/data01/home/yeziqi.831/nxtrec_wksp/testcode/vitnns_rebuild`

当前有效项目只能是 `vitnns_rebuild`。本次唯一授权是 `F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`：完成实现、synthetic/engineering preflight、验证、false-pass audit、completion、commit、push 和 archive ref。不得启动 F3C2 五图 pilot、F3C3 50 图 formal、F3C4 outcome analysis 或 F4。

## 一、开始前必须完整阅读

1. `docs/阶段F规划.md`，尤其第 21–23 节；
2. `docs/stage_f_f3c_class_targeted_protocol_20260907.md`；
3. `config/stage_f_f3c.json`；
4. `config/stage_f_f3c.schema.json`；
5. `manifests/stage_f_f3c_v1/sample_split.json`；
6. `evidence/stage_f/f3c0_protocol_v1_20260907/formal/completion/f3c0.json`；
7. `evidence/stage_f/f3h_full_head_r3_terminal_seal_20260907/formal/completion/f3h.json`；
8. `reports/stage_f_f3h_full_head_r2_cpu_reconcile_20260907/class_conditioned_basis_assessment.json`；
9. `docs/stage_f_f3h_full_head_gradcam_protocol_20260907.md`；
10. `src/vitnns/stage_c_class_logit.py`；
11. `src/vitnns/stage_c_attention.py`；
12. `src/vitnns/attribution.py`；
13. 与 recovery-v2 模型、样本、preprocessing 和 typed-terminal writer 直接相关的现有实现与测试。

## 二、启动门禁

正式修改或加载模型前必须验证并记录：

1. 分支为 `stage-d-d3-pass`；
2. local HEAD、`origin/stage-d-d3-pass` 和远端 branch 一致，worktree/index clean，Git LFS clean；
3. `refs/archive/stage-f-v1-f3h-completion` 精确指向 `56afedd2e73a760dc5127ab4ee9c38d72623dbbc`，且为当前 HEAD 祖先；
4. `refs/archive/stage-f-v1-f3c0-protocol-freeze` 精确指向 `e2c104bd6653ce62c8c67f7c0ed55a771d505837`，且为当前 HEAD 祖先；
5. F3H completion SHA-256 为 `26c47acc7729f20b3b22ab66f9a50ca67568fc67ecb9a939bee57c19ed6538c2`，status `PASS`，288/288 Value-path terminals VALID，Grad-CAM VALID；
6. F3C0 completion SHA-256 为 `6c049c445799c44a39dff2a19ff7a159e9e573baca47c3fe8b17a8c4ac52f901`，canonical digest 为 `09d5dd1ae704eb8d5d06c6a44a12bfe7182e6161b9ace2e7ac8fef8aeea08a48`，status `PASS`，`next_authorization=F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`，`empirical_f3c_started=false`，`f4_started=false`；
7. 冻结文件保持：
   - `docs/stage_f_f3c_class_targeted_protocol_20260907.md` SHA-256 `6b0bc933b635209e1c04e2ba1d4a11a15e9827b445edbc3eb32681fccef22bf1`；
   - `config/stage_f_f3c.json` SHA-256 `abb17c68c0f5961b75510604de486a2aa30509bb2c5dabc3a6688a1e3a11b0df`；
   - `config/stage_f_f3c.schema.json` SHA-256 `65191984978bb9191ca0474458f50949403e55741bcf8a8381b8e4beeecb055b`；
   - `manifests/stage_f_f3c_v1/sample_split.json` SHA-256 `b1dd0bb74f9ee0a2381eaa797c27a0365c13757b0f72ea253924fded3d8a2b09`；
8. F3H basis status 仍为 `INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS`；
9. F3 r7、blue-display r4、D10、E4、F0、F1、F2、F3H、F3C0 的 sealed evidence、completion 和 archive refs 无漂移。

若门禁失败，只允许诊断和最小的 F3C1 启动修复。不得 reset、stash、覆盖、删除或改写已有 evidence，不得把失败 root 改写成 PASS。

## 三、必须保持的方法定义

实现方法 ID：

`conditional_all_blocks_fixed_clean_attention_class_logit_ig`

每张允许的工程图像只解析一次 clean predicted class：

```text
y* = argmax_k logit_k(x)
```

在同一次 clean forward 中捕获模型实际消费的 12 层完整 post-softmax multi-head attention tensors：

```text
A*_l = A_l(x), l=0,...,11
t_NNS(z) = logit_y*(F(z; A_0=A*_0, ..., A_11=A*_11))
```

必须满足：

- 12 层 × 12 heads 全部保留，不选择、学习或人工设置 head 权重；
- `A*_l` detach、clone、hash 并冻结；
- Q/K/softmax variation 被切断；
- Value、head concatenation、attention output projection、residual、MLP、final norm、classifier 保持 live；
- 在 clean endpoint `x`，conditional forward 必须复现 clean raw logits，全部 logits 的 max absolute error `<1e-6`；
- baseline 为 normalized zero，即 canonical RGB `(0.5,0.5,0.5)`，不是黑图；
- IG 使用 float32 Gauss–Legendre schedule `[512,1024,2048,4096,8192]`、chunk 8、`atol=1e-4`、`rtol=0.01`，首个 conservation PASS 后停止并保留全部 attempts；
- native attribution 为 signed RGB `3×224×224`；必须先按符号求和 RGB，再做空间投影；
- 不能把该方法称为 raw attention、F3H maps 加权和、完整 Q/K/V attribution 或 unmodified full-model IG。

## 四、F3C1 必须完成的实现

1. all-block clean-attention capture、shape/dtype/device validation、canonical hash 和 frozen injection；
2. conditional class-logit forward 与 endpoint parity validator；
3. conditional input IG、adaptive schedule、conservation、attempt journal 和 native signed-map retention；
4. 五方法统一 runner，方法集合必须精确为：
   - class-targeted NNS；
   - ViT Grad-CAM；
   - full-model class-logit IG；
   - attention rollout；
   - block-11 raw CLS attention head mean；
5. 五方法共享 model、preprocessing、sample 和冻结的 `y*`；不得把后两种 class-agnostic 方法改称 class-targeted；
6. 14×14 common patch-mass projection、21 个 insertion/deletion fractions、5 个 hash-derived tie permutations和 `0.5*(InsertionAUC-DeletionAUC)` evaluator；display-only resize/normalization 不得回流 metric；
7. immutable sample×method typed terminal、pending→terminal、temp-dir→rename、fsync、resume 和冲突 payload fail-closed；
8. source/model/sample/clean-input/attention/target/attempt/resource/access/payload manifests 及 canonical digests；
9. 在任何 pilot outcome access 前冻结精确 paired statistical implementation：primary contrast 单独报告；三个 secondary contrasts 构成固定 Holm family；实现必须确定性处理 ties、zero differences、nonfinite/invalid units、seed、resample count、two-sided p-value 和 multiplicity adjustment，不得从 pilot/formal outcome 选择算法；
10. validator、CLI、测试、false-pass audit 和 atomic completion publisher。

优先复用已有 Stage C/F3H kernels，但必须验证语义和 hash；不得仅因函数名相似就假定兼容。

## 五、允许的 preflight 与严格数据边界

F3C1 允许：

- 任意不使用正式样本的 synthetic/unit tests；
- 使用已查看、已排除出统计的 continuity anchor order `44` 做恰好一次 numerical-only、无渲染、无人工 map 检视的一图五方法 engineering/profile check；
- 为上述检查加载冻结 recovery-v2 模型、解码 order-44 JPEG、使用必要 GPU，并记录真实资源；
- 验证模型、预处理、class index、attention capture/injection、endpoint parity、conservation、five-method runner、fidelity 和 retention plumbing。

F3C1 禁止：

- 解码、推理或读取 pilot orders `416,350,438,411,337` 的 JPEG、maps、logits、terminal outcomes 或 display；
- 读取或计算冻结 50-image formal cohort 的任何 JPEG、map、logit、outcome 或 display；
- 创建或启动 F3C2/F3C3 formal workload；
- 查看 order-44 新 map 后调整方法、IG schedule、阈值、统计、显示或选择规则；
- 渲染/发布 empirical comparison figure，选择最好看的方法/head/image，或产生 winner/GO/NO-GO；
- 修改 F3H/F3C0 frozen protocol、config、schema、split、completion 或 archive refs；
- 修改任何 sealed D10/E4/F0/F1/F2/F3/F3H evidence；
- 启动 F4。

如果 order-44 engineering check 暴露实现问题，保留失败 attempt，并仅在新的 recovery root 中修复和重试；不得覆盖失败 attempt。

## 六、测试与 false-pass audit

至少主动验证并拒绝：

1. 少于或多于 12 个 attention tensors；
2. 任一 tensor 不是 clean forward 实际消费的 post-softmax attention；
3. 只保留部分 heads 或引入人工/学习 head 权重；
4. attention 未 detach/freeze，或 Q/K/softmax variation 泄漏；
5. Value 或下游 projection/residual/MLP/classifier 被错误切断；
6. endpoint parity 不满足 `<1e-6` 仍继续；
7. class index 在方法或 perturbation 间变化；
8. baseline 被实现成黑图或其他输入；
9. IG 未通过 conservation、attempt 被丢弃或失败被伪装为 VALID；
10. RGB/patch projection 在求和前丢符号；
11. 五方法 roster、target 或既有 kernel 语义漂移；
12. common evaluator 的 fractions、ties、AUC 或 target 漂移；
13. display array、resize、gamma、normalization 回流 metric；
14. statistical test 在 pilot outcome 后才冻结，或根据结果换检验；
15. typed FAILED/NA 被补零、插值、删除或替换；
16. terminal 冲突被覆盖，pending 未完成却发布 PASS；
17. order 44 被计入 pilot/formal/statistics；
18. 读取任何 pilot/formal JPEG、map、logit 或 outcome；
19. engineering check 扩张成 empirical comparison、渲染或 winner 声明；
20. 资源/profile 缺失或预算字段不可复算；
21. 上游 sealed evidence/ref/hash 漂移；
22. F3C2、F3C3、F3C4 或 F4 提前开始；
23. completion、validator、manifest 或 false-pass audit 未完成即宣称 PASS；
24. `git diff --check` 失败。

Synthetic tests 必须覆盖一个可解析的 toy attention model，并证明：all-block capture/injection、endpoint parity、cut/live paths、IG conservation、signed projection、deterministic statistics、atomic terminal 和 tamper rejection。真实 order-44 工程检查只验证 plumbing 和资源，不提供科学结论。

## 七、F3C1 正式产物与完成条件

使用新的不可覆盖版本化 roots，例如：

- `evidence/stage_f/f3c1_implementation_v1_20260907/formal/`；
- `reports/stage_f_f3c1_implementation_v1_20260907/`。

至少生成：startup gate、implementation/config binding、synthetic fixture manifest、engineering-check manifest、attention binding、target binding、attempt journal、resource/access audit、statistical-method freeze、payload manifest、whole-root inventory、validation report、false-pass audit、test report、status 和 pending→terminal `completion/f3c1.json`。

F3C1 terminal PASS 必须明确记录：

- implementation/workflow/package validation `PASS`；
- synthetic tests 和所有 false-pass attacks `PASS/REJECTED`；
- order-44 engineering check 是否执行、模型/Torch/timm/GPU/CUDA 实际访问以及资源数值，不得谎报为零；
- pilot sample access `false`、formal sample access `false`；
- `f3c2_started=false`、`f3c3_started=false`、`f3c4_started=false`、`f4_started=false`；
- manual/learned head weights `false`；
- F3H/F3C0 和其他 sealed decisions/completions modified `false`；
- scientific comparison performed `false`，winner/GO/NO-GO produced `false`；
- `next_authorization=F3C2_PILOT_ONLY`。

若上述任一项不能可靠证明，状态必须为 `BLOCKED`，不得授权 pilot。

## 八、Git 交付

完成后：

1. 在 `docs/阶段F规划.md` 追加 F3C1 completion record；
2. commit 全部 F3C1 代码、测试、配置补充、报告和必要 evidence；
3. push `stage-d-d3-pass`；
4. 创建并核验 `refs/archive/stage-f-v1-f3c1-implementation`；
5. 最终核验 local HEAD、tracking ref、`git ls-remote --heads origin stage-d-d3-pass`、F3C1 archive ref、F3C0/F3H refs 不变、`git status --branch --porcelain=v2` 和 `git lfs status`。

最终回复必须报告：F3C1 PASS/BLOCKED、commit SHA、archive ref、实现文件、测试数、false-pass 数、order-44 engineering profile 结果和资源、endpoint parity/conservation、是否访问 pilot/formal、模型/GPU 实际使用、local/tracking/remote/LFS 状态，以及唯一下一任务是否为 `F3C2_PILOT_ONLY`。不得在同一轮继续执行 F3C2。
