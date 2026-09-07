# Stage F F3H 全 head atlas 与 Grad-CAM 前瞻性补充协议

日期：2026-09-07。协议 ID：`STAGE_F_F3H_FULL_HEAD_GRADCAM_V1_20260907`。本文件在任何 F3H 模型加载、新 attribution、Grad-CAM 计算或新结果查看之前冻结。

## 1. 授权与阶段位置

F3 r7 科学/来源封存和 post-F3 blue-display r4 展示封存保持只读。F3、Stage D、Stage E 的 completion、decision、样本选择、统计和 archive ref 均不得修改。

从本协议冻结起，F4 状态为 `PAUSED_PENDING_F3H`。F3 terminal receipt 中历史 `F4_ONLY` 字段不追溯改写；本协议是 owner 授权的前瞻性插入阶段。F3H terminal completion 通过且 archive ref 核验后，才重新授权 `F4_ONLY`。

F3H 分为四个原子里程碑：

1. `F3H0`：协议、config、schema、CPU-only validator 与测试冻结；不得加载模型、解码目标 JPEG 或计算 attribution。
2. `F3H1`：runner、typed-terminal writer、renderer 和 synthetic tests；只允许 implementation/preflight，不得启动正式 GPU workload。
3. `F3H2`：先执行冻结 pilot，再在预算门内完成正式 GPU attribution；不得渲染或人工选择 head。
4. `F3H3`：CPU-only source validation、固定渲染、人工 QC、false-pass audit、terminal completion、commit/push/archive。

本轮 F3H0 PASS 只能授权 `F3H1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY`，不直接授权 GPU 计算。

## 2. 研究目的和严格边界

F3H 只回答一个描述性问题：对于已经冻结的 Figure 6 `median_effect` 样本，12 个 blocks × 12 个 heads 的 fixed-clean-attention head-direction Value-path attribution 空间模式分别是什么；同一 head 的 NNS 与 matched patch 在共同显示尺度下如何分布；一个不同 target 的 clean predicted-class Grad-CAM 参照呈现什么空间支持。

F3H 不检验新的科学优越性，不产生新的 GO/NO-GO、winner、p-value、置信区间、fidelity、稳定性或定位指标。不得声称：

- NNS 是 class-logit attribution、raw attention、完整 attention/Q/K/V attribution；
- Grad-CAM 与 NNS/matched 解释同一 target；
- 深层比浅层“更语义”或图直接证明网络逐层学习内容；
- 某个 head 是 best/representative/object head；
- 热点与物体主体重合等价于方法忠实性或正确性。

## 3. 冻结样本、模型与预处理

唯一样本固定为 F3 r7 在查看 E3 map 前已封存的 `median_effect`：

- sample ID：`ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG`；
- selection order：`44`；
- image SHA-256：`c597ef7e061560542edbd77d2cbca6107cc44ff5e86de487a015eb47ae4620a8`；
- official ImageNet-1k index：`713`；
- sealed clean top-1 index：`713`，且该样本 clean top-1 correct。

不得根据新 atlas 或 Grad-CAM 外观换样本。正式运行前必须重新核验 F3 selection/cross-layer manifests、D1 registry、JPEG 字节和 clean tensor hash。

模型必须唯一绑定 recovery-v2 artifact：

- `assets/models/vit_base_patch16_224_weights.semantic_candidate.pth`；
- file/LFS SHA-256 `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d`；
- canonical tensor digest `150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd`；
- 152 keys，86,567,656 parameters；
- identity `RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718`。

历史 `2e718...` 不能作为 F3H checkpoint。若文件 SHA、LFS OID、tensor digest、key/parameter count 或 sealed D8/E3 provenance 任一不一致，必须在 inference 前 `BLOCKED`。预处理固定为 RGB、shorter side 248 bicubic、224×224 center crop、mean/std `(0.5,0.5,0.5)`。

## 4. 全 head Value-path workload

坐标全集在结果访问前冻结为 block-major 顺序：

```text
(block, head), block = 0..11, head = 0..11
```

每个坐标固定两种同 target 方法：

1. `conditional_fixed_clean_attention_head_direction_value_path_ig`；
2. `matched_fixed_clean_attention_patch_token_contribution`。

逻辑总量为 144 coordinates × 2 methods = 288 typed terminals。F3 r7 已封存的 `(0,9)`、`(4,2)`、`(6,10)`、`(11,2)` 共 8 个 valid maps 必须按 path/file SHA/signed-map digest/model/image/clean-input/target binding 继承，禁止重算。其余 140 coordinates × 2 methods = 280 个新 terminal units。

NNS 数值合同保持 recovery-v2 E3 最终合同：float32 Gauss-Legendre，steps `[512,1024,2048,4096,8192,16384,32768]`，chunk 8，从小到大取第一个满足 `abs_error <= 1e-4 + 0.01*abs(delta)` 的结果。matched patch 使用同一 clean attention、head direction、baseline、target 和 token identity。每个 unit 必须原子终止为 `VALID`、`FAILED` 或 `NA`；不得补零、插值、复制相邻 head 或换 target。

正式保留的 Value-path map 是 signed float32 56×56 fine-cell map。可额外保留重建所需 native payload，但 display array 不得回流到 attribution、排序或任何 metric。F3H 不运行 insertion/deletion、stability、sanity 或 perturbation。

## 5. 正负贡献拆分与 pair-shared log 显示

对每张 raw signed map `S` 定义：

```text
positive(S) = max(S, 0)
negative_magnitude(S) = max(-S, 0)
```

对同一样本、同一 `(block, head)` 的 NNS/matched pair，固定共同尺度：

```text
M = max(max(abs(S_nns)), max(abs(S_matched)))
D(C) = log1p(999 * C / M) / log(1000),  C >= 0
```

若 `M=0`，四个 component display arrays 全零并标记 `ZERO_PAIR_SCALE`；不得改尺度或换 head。NNS+/matched+/NNS−/matched− 四个 panel 使用相同的 `M`、`kappa=999`、色图和 `[0,1]` 显示范围。该变换是严格单调的共同变换：不允许 method-specific gain、gamma、alpha、阈值或亮度补偿；不得人为“调亮 NNS、调暗 matched”。

色图固定为 deep-blue→blue→cyan→yellow→orange→red：`0.0 #07077F`、`0.2 #0019C6`、`0.42 #00AEE8`、`0.65 #FFF000`、`0.82 #FF6A00`、`1.0 #D90000`。56×56 到 224×224 仅用 bilinear display resize；无 clipping、percentile、smoothing 或 max pooling。

每个 panel 必须记录 raw signed array digest、positive/negative array digest、pair scale `M`、`kappa`、display-array digest、source path/hash、status 和 transform。跨 head、跨 block 的 target 不同，即使显示范围同为 `[0,1]`，也不得作绝对 attribution magnitude 比较。

## 6. Sealed Grad-CAM 参照

仓库当前没有 order 44 的 sealed Grad-CAM map；F3H 前瞻性授权恰好一个新 Grad-CAM terminal unit：

- method：`vit_grad_cam_block11_norm1`；
- exact name：`ViT Grad-CAM (blocks[11].norm1, clean predicted-class raw logit)`；
- sample/order：同一 order 44；
- target：clean predicted-class raw logit，class index 在任何 Grad-CAM map access 前固定为 sealed value `713`；
- target layer：`blocks[11].norm1`；
- CLS excluded，196 patch tokens；channel weight 为 target gradient 在 196 patch tokens 上的算术均值；weighted activation channel sum 后 ReLU；
- raw output：nonnegative float32 14×14；不做算法内 min-max、平滑或 percentile enhancement。

正式运行必须重新解析 clean logits，并要求 argmax 恰为 `713`；否则在写 map 前 `BLOCKED`。Grad-CAM raw map 必须单独封存并哈希；显示时使用自身 raw maximum 和同一 `kappa=999` log transform，只用于看清空间支持。它与 Value-path NNS/matched target 不同，不共享色限、不进入 NNS/matched 的差值、排名、fidelity 或胜负结论。Grad-CAM 必须 `VALID` 才能完成 F3H；all-zero、nonfinite、target drift 或 hook/shape failure 均使 F3H `BLOCKED`。

## 7. 固定图形模块

模块编号为 F3H 内部 ID，最终论文编号仍由恢复后的 F4 决定：

1. `F3H-A`：NNS positive 12×12 atlas，rows=blocks 0..11，columns=heads 0..11；
2. `F3H-B`：NNS negative-magnitude 12×12 atlas，同顺序；
3. `F3H-C`：matched positive 12×12 atlas，同顺序；
4. `F3H-D`：matched negative-magnitude 12×12 atlas，同顺序；
5. `F3H-E00..E11`：每个 block 一页，固定 4×12 rows=`NNS+`,`matched+`,`NNS−`,`matched−`，columns=heads 0..11；
6. `F3H-F`：Original、sealed Grad-CAM heatmap 和 Grad-CAM overlay 参照模块。

共 17 个 render instances，每个必须从同一 panel manifest 导出 SVG/PDF 与 600-DPI PNG，并生成 17 张 report-side PNG review copies。不得生成只包含“最好看 heads”的精选面板；如未来 F4 需要缩减主文，只能引用完整 atlas 并按与 outcome 无关的固定坐标，不得事后挑选。

强制图注必须说明：

> Full-head panels show fixed-clean-attention head-direction Value-path attribution for one pre-existing D10 median-effect sample. They are not raw self-attention or class-logit attribution. Positive and negative contributions are separated. Within each coordinate, NNS and matched patch use one shared monotonic log display scale with no method-specific gain; colors across heads or blocks do not compare absolute attribution magnitude. All 12×12 heads are retained, including failed or unavailable cells. The sealed Grad-CAM panel uses a different clean predicted-class logit target and is a qualitative reference, not a same-target winner comparison.

## 8. 执行、预算与故障策略

正式 root 固定为 `evidence/stage_f/f3h_full_head_v1_20260907/formal/`，report root 为 `reports/stage_f_f3h_full_head_v1_20260907/`。必须 pending→terminal、unit temp-dir→rename、fsync、不可覆盖。中断恢复只能跳过已经完整验证的 terminal unit。

冻结 pilot coordinates 为 `(0,0)`、`(4,0)`、`(8,0)`、`(11,0)`；如果某坐标属于新计算，运行两方法但在 bulk authorization 前不得查看 JPEG、map 或 display。pilot continuation 只根据完整性、数值状态、资源和预算，不得根据热点外观改变坐标、方法、schedule 或显示参数。

历史 E3 1,500 个 NNS 与 1,500 个 matched units 的最大 elapsed 分别为 418.6982 秒和 8.6280 秒。对 140 个新 pair 使用 2× observed-max 安全因子并为 Grad-CAM/控制面预留 0.25 小时，保守投影为 33.5 H20-GPU-hours；硬上限固定 40.0 H20-GPU-hours，最多 8 张 H20，单进程 peak reserved cap 16 GiB。达到硬上限时停止发放新 unit、保留 terminal payload 并将阶段标为 `BLOCKED_BUDGET`；不得自动 waiver。

允许按 block-major coordinate index modulo 8 分 shard；同一坐标的 NNS/matched 必须在同 shard。Grad-CAM 在 Value-path terminal reconciliation 后单独执行。实际 elapsed、GPU-hours、forward/backward calls、peak memory 和 bytes 必须全量记录，包括失败和中断尝试。

## 9. 机器文件与验收

F3H0 protocol freeze 至少交付 config/schema、protocol validation、false-pass audit、test report、status 和 completion。F3H terminal package 至少包含：startup gate、frozen plan、unit manifest、attempt journal、terminal union、source binding、model/sample/clean-input bindings、Grad-CAM target binding、signed-map inventory、positive/negative manifest、pair-shared scale manifest、panel manifest、render/review inventories、access/resource audit、validation、false-pass audit、whole-root inventory 和 completion。

F3H terminal PASS 必须同时满足：

- 144/144 coordinates、288/288 Value-path method terminals；其中 8 个合法继承，280 个新 terminal；
- exactly one VALID sealed Grad-CAM terminal；
- 所有 valid raw maps 与 model/sample/target/source hashes 一致；
- FAILED/NA 原位显示且不补零；
- 四个 12×12 atlas、12 个 block pages 和 1 个 Grad-CAM module 完整；
- 17/17 review copies 逐图检查；
- 无 method-specific display gain，无 best-head selection，无新 fidelity/statistics/decision；
- F3 r7、blue r4、D10、E4 completion/decision 未修改；
- F4 未启动。

Value-path 有 typed FAILED/NA 时允许 workflow/package `PASS` 与 scientific validity `MIXED`，前提是 288 units 全部 terminal 且失败原位披露；Grad-CAM 非 VALID 则 F3H `BLOCKED`。F3H PASS 的唯一 next authorization 为 `F4_ONLY`。

## 10. False-pass audit

validator 必须主动拒绝至少以下攻击：样本或 class index 换成视觉更佳者；只计算/显示部分 heads；挑选 best head；把 4 个已有坐标重算；model/JPEG/clean-input/source hash 漂移；NNS/matched target 不配对；method-specific brightness/gamma/alpha；分别自动缩放 pair；正负混为 abs 主图；clipping/smoothing/max-pooling；FAILED/NA 隐藏或补值；Grad-CAM 与 Value-path 冒充同 target；Grad-CAM 算法内 min-max；运行 fidelity/stability/统计；预算自动 waiver；先看 pilot 热图再改协议；跨格式不同源；F4 提前开始；未完成人工 QC/validator/manifest 就发布 PASS；修改 sealed F3/D10/E4 证据或 archive ref。
