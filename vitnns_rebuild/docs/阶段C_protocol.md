# 阶段 C 冻结实验协议

## 1. 状态、范围与修订规则

- 协议版本：`1.0.0`。
- 冻结阶段：C0，2026-08-21，在任何 Stage C heatmap、候选指标或 holdout 结果产生前。
- 机器可读配置：`config/stage_c.json`；字段合同见 `docs/stage_c_config_contract.md`。
- 本协议只定义阶段 C。C0 不实现比较方法、统一指标、runner 或 selection，也不运行 GPU attribution。
- 阶段 B 的 `conditional_fixed_attention_value_path_ig` 是单 head-channel、固定 attention 的 Value-path 方法；它不是完整 attention attribution，也不是类别解释。阶段 C 的 head-direction 方法使用新名称，不改变阶段 B 产物语义。
- 冻结后若必须修改科学口径，须在读取相关 holdout 结果之前提交带日期、理由、旧/新协议及配置 SHA-256、影响范围的 amendment。若修改发生在查看 holdout 之后，该 15 图不再具有确认性 holdout 身份。失败记录不得删除。

本协议的结论边界是一个历史文件名等间隔抽取的 20 图 pilot；它不是类别均衡样本，也不是阶段 D 的独立论文测试集，不能支持“对 ImageNet 普遍更优”的结论。

## 2. 冻结输入、样本与角色

### 2.1 输入证据链

官方标签链固定为：validation 文件名中的 1-based image number → 官方 devkit validation ground-truth 中的 ILSVRC class ID → `meta.mat` leaf WNID → timm ImageNet-1k canonical 0-based index。禁止用模型预测、目录顺序或人工类别代替真值。

冻结输入如下：

- official devkit SHA-256：`b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953`；
- ImageNet-1k synset order SHA-256：`70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15`；
- Stage A 重建的 500 图 manifest：`manifests/stage_c_flat_val_500/manifest.csv`，SHA-256 `042cfd7845b49f8949de13c78bc40c68192179d6fc004bcdbecf55e502c2eb89`；
- 20 图 registry：`manifests/stage_c_sample_registry.csv`；registry 的 SHA-256 由配置和 completion record 固定。

20 个 ID 来自旧脚本对按文件名排序的 500 图执行 `np.linspace(0, 499, 20, dtype=int)`，对应 zero-based 索引：

```text
0, 26, 52, 78, 105, 131, 157, 183, 210, 236,
262, 288, 315, 341, 367, 393, 420, 446, 472, 499
```

旧目录 `49_vit ` 始终只读。旧清单仅证明 ID 来源；旧 attention、heatmap 和数值结果不得进入 Stage C 指标或选择。

### 2.2 角色

- calibration（5）：`00000079`、`00009888`、`00023229`、`00035680`、`00049970`。只用于 C8 全层/全头校准、selection、工程调试和描述性报告。
- holdout（15）：其余 15 图。它们是阶段 C 唯一的确认性统计样本并决定 Go/No-Go。
- audit（2）：`00000079`、`00049970`。`audit` 是 calibration 上的留存 overlay，不是第三种互斥统计角色，也不增加样本量。
- all-20 只能标为 descriptive total。预测错误样本、退化样本和失败样本均保留，不得剔除后缩小分母。

20/20 文件可解码，hash、字节数、尺寸和官方标签均与 500 图 manifest 一致，因此不调整划分。`00012573` 的原始尺寸仅 `90×90`；它不是损坏文件，仍保留在 holdout，但不得把 resize 后的细节称为原生高分辨率证据。

## 3. 两条比较线和精确方法名

### 3.1 Value-path 线

这一条线解释固定 clean endpoint attention 下的 block/head 内部 Value-path 标量。包含：

1. `NNS fixed-clean-attention head-direction Value-path IG`，ID `conditional_fixed_clean_attention_head_direction_value_path_ig`；
2. `matched fixed-clean-attention patch-token contribution`，ID `matched_fixed_clean_attention_patch_token_contribution`；
3. `direct raw CLS attention (matched block/head)`，ID `direct_raw_cls_attention_per_head`，仅为不同目标的空间结构参考。

只有前两者解释同一标量。raw attention 与 NNS 的 Pearson、Spearman、JS、KL 或 hotspot 只能称为 `spatial structure similarity`，不能称为同目标 fidelity、方法忠实性或优越性。

### 3.2 Class-logit 线

对 clean 输入 `x`，固定 `y*=argmax_k logit_k(x)`，解释/评估 raw logit `t(z)=logit_y*(z)`。所有扰动、deletion 和 insertion 继续使用 `y*`；同时记录官方真值、clean top-1 是否正确、扰动自然预测和类别翻转，但不因预测错误删除样本。

方法精确为：

1. `direct raw CLS attention (block 11 head mean)`：block 11 的 12 heads 先算术平均，再排除 CLS key 并将 196 个 patch 权重归一到和为 1。它是 class-agnostic ranking baseline，不是类别归因；可用 frozen class logit 评价 deletion/insertion。
2. `attention rollout (mean-head residual, blocks 0-11)`：对每层 `B_l=row_norm(I+mean_h(A_lh))`，rollout 为 `B_11 B_10 ... B_0`；取 `[CLS, patch]`、排除 CLS、归一到和 1。它是 class-agnostic attention-flow baseline。
3. `ViT Grad-CAM (blocks[11].norm1, clean predicted-class raw logit)`：目标层固定 `blocks[11].norm1`，排除 CLS，channel weight 是 196 个 patch token 上 gradient 的均值，最后 ReLU；不做 min-max normalization。
4. `full-model class-logit IG`：真实完整模型对 `t` 的 IG，使用第 4 节 baseline/schedule；它不得简称为完整 attention attribution。

这四种方法共享同一输入、模型、预处理和 frozen clean class target。它们可在 class-logit deletion/insertion 下比较，但不能用该结果证明内部 Value-path NNS 的目标守恒。

## 4. NNS head-level target 与 matched token 分解

对 clean normalized input `x`、baseline `x0`、block `l`、head `h`，baseline 固定为 normalized tensor 全零，即 canonical RGB `(0.5,0.5,0.5)`；它不是黑图。query 固定为 CLS（index 0）。

从 clean 真实 timm forward 捕获模型实际使用的 attention `A_lh(x)`。定义：

```text
g_lh(z) = sum(j=0..196) A_lh(x)[0,j] V_lh(z)[j] in R^64
d_lh    = g_lh(x) - g_lh(x0)
r_lh    = ||d_lh||_2
u_lh    = d_lh / r_lh
f_lh(z) = u_lh^T g_lh(z)
```

`V_lh(z)` 来自 z 经真实上游网络到所选 block 的 Value。至关重要的是 `g_lh(x0)=A_lh(x)@V_lh(x0)`，不是 `A_lh(x0)@V_lh(x0)`。`A_lh(x)`、`d_lh` 和 `u_lh` 在积分前 detach、clone 并固定，不能在 quadrature step 中重算。于是 `f_lh(x)-f_lh(x0)=r_lh>0`，方向符号不再任意。

方向阈值为：

```text
tau_dir = 1e-6 * max(1, ||g_lh(x)||_2, ||g_lh(x0)||_2)
```

若任一量非有限或 `r_lh <= tau_dir`，状态为 `DEGENERATE_DIRECTION`；禁止临时换 channel、head、baseline 或翻转方向。primary holdout 发生此状态即 validity 失败。

IG 使用 float32 Gauss-Legendre，steps schedule `[512,1024,2048,4096,8192]`、chunk 8；从小到大取首个满足 `abs_error <= 1e-4 + 0.01*abs(delta)` 的 steps，否则保留全部 attempt 并失败。clean 与 baseline 两个 endpoint 的 intervention parity 都要求最大绝对误差 `<1e-6`。audit 重复性要求 `atol=1e-6, rtol=1e-5`。

同目标的解析 token contribution 为：

```text
c_j = A_lh(x)[0,j] * u_lh^T (V_lh(x)[j] - V_lh(x0)[j]), j=0..196
```

要求 `abs(sum_j c_j-r_lh) <= 1e-6 + 1e-5*abs(r_lh)`。`c_0` 是 CLS token contribution，必须单独保存；只有 `c_1..c_196` reshape 为 `14×14` patch spatial map。该 map 不假装给 CLS contribution 指定空间位置。

定义 endpoint patch support：

```text
s_patch = sum(j=1..196)|c_j| / (|c_0| + sum(j=1..196)|c_j|)
```

分母为零是 invalid；`s_patch<0.01` 标记 `DEGENERATE_PATCH_SUPPORT`。patch 正证据小于全部 token L1 的 1% 标记 `DEGENERATE_POSITIVE_PATCH_EVIDENCE`。二者不改变数学守恒，也不得被删除，但计为 novelty/spatial failure。matched patch baseline 与 NNS 的 fidelity 都评价同一个 `f_lh`；前者的空间 ranking 仅来自 patch tokens，CLS scalar 只单独报告。因此结果只能支持“超越 endpoint patch-token spatialization”的有限主张。

## 5. Map、投影、符号和指标

### 5.1 数值空间和质量守恒投影

- 所有跨方法空间指标的共同网格为 `14×14`。
- signed RGB attribution 先按 RGB channel 做带符号求和，得到 float32 `224×224`；不取绝对值。
- pixel → patch：每个非重叠 `16×16` 区域求和。
- patch → pixel：每个 patch mass 均分到 256 个像素，即每像素 `Q[r,c]/256`。
- fine fidelity：每个 patch 切成 16 个非重叠 `4×4` cell；pixel map 在 cell 内求和，patch baseline 每 cell 为 `c_j/16`。
- 投影先作用于 signed map，再拆 positive/negative；投影和前后总和要求 `atol=1e-6, rtol=1e-6`。
- raw CLS attention 固定 query `0`，删除 self CLS key，196 个非负值再归一；分母非正或非有限是合同失败。
- 禁止用 image resize、bicubic、复制 patch 数值而不除面积、Gaussian 平滑、percentile enhance、max-pool 或 colormap 作为定量投影。display PNG 永远不能回流指标。

### 5.2 signed views

对 signed map `M`：

```text
signed   = M
positive = max(M, 0)
negative = max(-M, 0)   # 负证据幅值
```

不得默认使用 `abs(M)`。Pearson/Spearman 可比较 signed↔signed、positive↔positive，以及双方都具有负证据语义时的 negative↔negative。JS、KL、hotspot 和 entropy 只用于非负 positive/negative view。非负方法的 negative 比较返回 `NA: unsupported_sign_semantics`，不能把人为全零图当成负证据。

### 5.3 指标定义

指标以 float64 累加，输入须同 shape、有限，并使用同一 valid mask：

- Pearson：标准中心化乘积矩相关。
- Spearman：ties 取 average/mid-rank，再做 Pearson。若 `||v-mean(v)||_2 <= 1e-12*max(1,||v||_2)`，该侧为 constant-map NA。
- distribution：先将非负 branch 除以自身总质量。若 branch mass `<=1e-12` 倍原 signed map L1（全零也包括），为 zero-mass NA。
- Jensen-Shannon：报告 divergence 而非平方根，`JSD=0.5 KL(p||m)+0.5 KL(q||m)`，`m=(p+q)/2`，log base 2，采用 `0 log 0=0`，不加 epsilon，范围 `[0,1]`。
- KL：辅助量，双向报告。先得到 mass-one `p,q`，再用 `p_eps=(p+1e-12)/(1+n*1e-12)`（q 同理），报告 base-2 `KL(p_eps||q_eps)` 与 `KL(q_eps||p_eps)`。
- hotspot：top 10% 和 20%，`k=ceil(fraction*n_valid)`；数值降序、相同值按 row-major linear index 升序，记录 cutoff tie count。完整 `14×14` 时 k 分别为 20、40。常量图返回 NA。Jaccard 是集合交集大小除以并集大小。
- normalized entropy：`-sum(p_i log2 p_i)/log2(n_valid)`；零质量或 `n_valid<2` 为 NA。
- 同时记录 conservation abs/relative error、实际阈值/steps、endpoint/intervention parity、fidelity/stability 指标、同步 wall time、CUDA peak allocated/reserved 和 artifact bytes。

每项 NA 都是结构化对象 `{status:"NA", value:null, reason:...}`。允许原因至少为：`constant_left`、`constant_right`、`zero_mass_left`、`zero_mass_right`、`unsupported_sign_semantics`、`degenerate_target_delta`。`empty_valid_region`、`shape_mismatch`、`non_finite`、`degenerate_direction`、`missing_artifact` 在 required primary unit 中是 invalidating NA，必须使 validity 失败。相关或 hotspot 的合法常量 NA 不可伪造为 0；但 primary residual stability 的常量/NA 按 spatial failure 计，不能从分母删除。

## 6. Fidelity 协议

### 6.1 目标响应与 corruption

对被冻结标量 `t`：

```text
Delta = t(x)-t(x0)
sigma = sign(Delta)
S(z)  = sigma * (t(z)-t(x0)) / |Delta|
```

因此 `S(x0)=0`、`S(x)=1`；Value-path 的 sigma 按构造为 +1。若 `|Delta| <= 1e-6*max(1,|t(x)|,|t(x0)|)`，为 `NA: degenerate_target_delta`。Value-path 中它等同 direction validity 失败；class-logit 线中是该样本 class fidelity 的合法 NA，不得换 baseline/class，且不影响 Value-path Go/No-Go 的 15 图分母。

排序分数为 target-oriented signed map `sigma*M` 的正部。replacement 永远是同一个 `x0`；RGB 三通道作为一个空间单位同时替换。deletion 从 clean 按 ranking 依次替换为 baseline；insertion 从 baseline 依次恢复 clean。不能用 blur、随机颜色或重估均值。

固定 21 个 fraction：`0,0.05,...,1.0`。N 个单位下，`k_0=0`，其余 `k=ceil(fraction*N)`，横轴用实际 `k/N`；AUC 用 trapezoid，不截断 S 到 `[0,1]`。Deletion AUC 越低越好，Insertion AUC 越高越好。

所有 score tie group 使用 5 个预注册 hash-seeded permutation，并报告曲线/AUC 的均值与标准差；同一 sample/config/replicate 的方法共享 seed。base seed `20260821`，seed 为字符串 `stage-c-fidelity|20260821|sample_id|block|head|replicate` 的 SHA-256 前 8 bytes 按 big-endian unsigned 解读。五次不是独立统计样本。另报五次全局 random ranking 和按单位中心距图像中心升序（再 row-major）的 center prior；二者仅为 sanity。

### 6.2 两个尺度与边界

- patch fidelity：196 个 `16×16` unit。公平稳健，只说明 patch ranking，不支持像素级新增主张。
- confirmatory fine fidelity：3136 个 `4×4` cell。NNS 分数为 cell 内 signed mass 求和后取正部；matched patch map 均分为每 patch 16 个相同 cell。它只叫 `4×4-cell/subpatch fidelity`，不叫单像素 fidelity。

Value-path confirmatory 增益仅比较 NNS 与 matched patch baseline：

```text
delta_fid = 0.5 * ((InsertionAUC_NNS - InsertionAUC_matched)
                 + (DeletionAUC_matched - DeletionAUC_NNS))
```

Go/No-Go 的 `delta_fid` 固定指 fine 4×4 结果；patch 结果为支持性分析。class-logit 四方法在同一 frozen raw logit 下分别做 patch 和 fine fidelity；原生 patch 方法在 fine scale 的 ties 也按同一五 seed 规则处理。不同 line 的 AUC 不混入 Value-path `delta_fid`。

top-20% sufficiency/comprehensiveness 同样按 `ceil(0.2*N)`：

```text
sufficiency_20       = S(only top-20% kept clean)
sufficiency_gap_20   = 1-sufficiency_20
comprehensiveness_20 = 1-S(top-20% deleted)
```

patch/fine 都报告，但不另设决策门槛，避免对同一曲线重复计票。

## 7. 扰动与稳定性

所有扰动作用于 canonical resize+center-crop 后、normalize 前的 float32 RGB `[0,1]` tensor；同一预生成 tensor/hash 供所有方法和配置共享。class 线固定 clean `y*`。Value-path 固定 clean `A_lh(x)`、`u_lh(x)`、block/head 和 `x0`；不得在扰动端点重选 attention 或方向。raw/rollout/Grad-CAM/full-model IG 在扰动 tensor 上重算 map，但目标 class 不变。

### 7.1 向右下平移 4 像素

- `dx=+4, dy=+4`，整数移动，无插值，reflection padding。
- clean `(r,c)` 移到 perturbed `(r+4,c+4)`。
- 逆变换将 perturbed `[4:224,4:224]` 复制回 clean `[0:220,0:220]`。
- 有效像素域 `[0,220)×[0,220)`；patch 指标仅用完全有效 rows/cols `0..12`，即 169 patches。padding 区不参与指标。

### 7.2 中心裁剪/尺度

- 从 canonical tensor 取 `[4:220,4:220]` 的 `216×216` 中心区域；bicubic、antialias=true resize 回 `224×224`，等效放大约 1.037。
- map 逆变换用 bilinear、antialias=true resize 到 `216×216`，乘质量 Jacobian `(224/216)^2=1.075445816186557`，放入 clean `[4:220,4:220]`。
- 有效像素域 `[4,220)^2`；patch 指标仅用完全有效 rows/cols `1..12`，即 144 patches。

### 7.3 轻 Gaussian 噪声

- RGB `[0,1]` 空间 iid `N(0,0.01^2)`，CPU float32 generator，之后 clamp `[0,1]`。
- `seed=20260821+registry.selection_order`；同一 sample 所有方法共享 tensor。
- inverse 为 identity，有效域是全部 196 patches。
- 这不同于 Stage B normalized-space `std=0.05` sensitivity sanity；后者证明解释非完全不响应输入，不是 Stage C 稳定性上界。

### 7.4 稳定性统计与空间新增信号

clean 与逆变换后的 perturbed map 只在共同有效区域计算 Pearson、Spearman、JS 和 top-10/20 Jaccard。所有样本的 frozen clean target 结果是主报告；prediction unchanged 子集和 flip rate只作描述。

对 NNS native `224×224` map 定义自身 patch 内 residual：

```text
R(M) = M - U(P(M))
rho  = ||R(M)||_1 / ||M||_1
```

P/U 是第 5 节质量守恒投影。几何边界 patch 仅在有效像素上以该 patch 有效区域的均匀 mass 构造 residual，稳定性只比较共同有效像素。零分母/常量 residual 为该图 spatial failure。novel spatial signal 要求至少 12/15 holdout 图同时满足 `s_patch>=0.01`、patch 正证据占全部 token L1 `>=0.01`，以及 clean `rho>=0.05`。

对 translation、scale、noise 各自，clean/aligned perturbed residual signed Spearman 必须：至少 12/15 `>0`；median `>=0.50`；image-bootstrap 95% CI lower `>0`。其他稳定性指标照常报告但不重复设硬门槛。

## 8. C8 selection（只用 calibration）

候选空间固定为 12 blocks × 12 heads，深度带为 `[0,1,2]`、`[3,4,5]`、`[6,7,8]`、`[9,10,11]`。候选 eligibility 要求 5/5 calibration 的 endpoint parity、direction、IG conservation、token parity、required maps/curves、provenance 和 hard budget 全部有效；不得读取 holdout、预测正确性、图片内容、display PNG 或热图美观。空间退化保留在排序中，不作为静默剔除条件。

每候选在 5 calibration 上计算：

```text
n_pos  = count(delta_fid_fine4 > 0)
m_fid  = median(delta_fid_fine4)
m_stab = min over 3 perturbations of median residual Spearman
m_res  = median(rho)
```

NA stability 排序值固定 -1，NA residual ratio 排序值固定 0。按以下 tuple 降序；科学量先 round 到 6 decimals，再以 `(block_index,head_index)` 升序破 tie：

```text
(n_pos, round(m_fid,6), round(m_stab,6), round(m_res,6),
 -block_index, -head_index)
```

每个深度带选 1 个，再以同一规则从四个 winner 选唯一 primary；其余三项仅探索性。任一 band 无 eligible candidate，C8 为 BLOCKED。`selection/stage_c_selection.json` 必须由程序生成，记录 144 个候选与失败，绑定 code/protocol/config/registry/calibration artifact hashes，且 `holdout_accessed=false`；禁止人工改写。探索性三项按 metric family 做 Holm correction（alpha 0.05）。

## 9. 留存、provenance、失败和恢复

- L0：每个预期 sample × method × config execution unit 的显式终态、逐样本标量、预测/target、失败/退化 reason、同步时间、CUDA allocated/reserved、artifact bytes 和 provenance。
- L1：所有预注册配置的 metric-ready 2D 数值图。C0 冻结 float32 存储；以后若要量化，必须在看 holdout 前 amendment 并给出 audit replay 误差门槛。
- L2：仅两个 audit calibration，保存 float32 raw attribution、raw attention、endpoint、CLS/token contributions、收敛 attempts 和足以复核 parity/守恒的中间量。
- L3：仅 audit/显式失败案例的 display PNG，从 L1/L2 派生，永不作为指标输入。

禁止逐 quadrature step、逐像素文件或逐 dimension 文件。每项结果至少绑定 Git commit 与 clean/dirty、model/image/lock/registry/protocol/config/selection SHA-256、角色/audit、精确方法版本和 target、projection/sign/normalization、baseline/schedule/seed/扰动、终态/失败、资源和 artifact inventory。

每个 execution unit 写入新的独立目录，禁止覆盖或向 incomplete 单元续写。只有 `status.complete=true`、status 中 manifest hash 正确且 manifest 中全部 artifact hash/bytes 正确时才可 resume-skip。失败仍写 failure table；修复后用新空目录。batch complete 要求每个 expected unit 有显式终态，不能把 failed/missing 当成功。

GPU wall time 在测量边界前后 `cuda synchronize`；每单元开始前 reset peak memory，分别记录 allocated/reserved。运行前估算 samples × methods × blocks × heads × targets × clean/3 perturbations × repeatability × 21 curve points × 5 tie repeats及全部控制文件。

硬预算：

- 每 L0+L1 execution unit 64 MiB；每 audit sample 全部 L2 1 GiB；
- C8+C9 全部 retained artifacts 32 GiB；Stage D 500 图投影 64 GiB；
- 每 execution unit forward/backward batched calls 各 30,000；peak CUDA reserved 90 GiB；
- C8+C9 合计 1,000 H20-GPU-hours；Stage D 500 图投影 2,000 H20-GPU-hours 且不超过 4 GPU-hours/image；
- C7/C8 后按实测最慢有效 unit × 精确 unit 数 × 1.25 safety factor 估时间；storage 加 25% control/provenance overhead。所有普通文件均计入 bytes。

C7 可用实测收紧预算，不得放宽。超 cap 要 BLOCKED 或在访问 holdout 前提交 amendment，不能静默减层、头、样本、steps 或扰动。最终 Stage D 投影超 cap 且 validity/novelty 已通过时为 `NO_GO_COST`。

## 10. 统计与最终决策

统计单位固定为图像。确认性 `n=15`；calibration、layer、head、pixel、patch、seed、扰动不是额外独立样本。bootstrap 对 15 个 image indices 有放回整簇抽 15 个，10,000 次，base seed `20260821`；metric seed 取 `stage-c-bootstrap|20260821|metric_id` SHA-256 前 8 bytes big-endian unsigned。CI 为 percentile 2.5%/97.5%，linear quantile。报告逐图、mean、median、sample SD（ddof=1）和 95% CI。

### 10.1 Validity

唯一 primary 的 15/15 holdout 必须完成、有限、hash/provenance 完整，并满足：endpoint parity `<1e-6`；IG conservation `<=1e-4+0.01|delta|`；token parity `<=1e-6+1e-5|r|`；无 direction degeneration；两个 audit 重复运行满足 `atol=1e-6,rtol=1e-5`；无静默失败/排除或 display 输入。任一失败优先输出 `NO_GO_VALIDITY`。合法的 secondary class-line NA 不改变 Value-path validity，但必须保留。

### 10.2 Effect

令 15 个 `delta_fid_fine4` 的 median 为 m、正值数为 n+、paired image-bootstrap median CI 为 `[L,U]`：

- PASS_EFFECT：`m>=0.02` 且 `n+>=12` 且 `L>0`；
- FAIL_EFFECT：`n+<12` 或 `U<0.02`；
- 其余为 UNCERTAIN_EFFECT。

### 10.3 Spatial novelty/stability

- support/positive-evidence/residual：至少 12/15 同时满足 `s_patch>=0.01`、patch 正证据占全部 token L1 `>=0.01`、`rho>=0.05`，否则 FAIL_SPATIAL。
- 每个扰动：至少 12/15 residual Spearman `>0`；median `>=0.50`；median bootstrap CI lower `>0`。正方向数不足 12 为 FAIL；median `<0.50` 且 CI upper `<0.50` 为 FAIL；未满足 PASS 又未满足 FAIL则 UNCERTAIN。任一扰动 FAIL 即 FAIL_SPATIAL，任一 uncertain 且无 fail 即 UNCERTAIN_SPATIAL。

低 NNS/raw-attention 相关或较大 JS 本身不是新增信息，也不是 fidelity gate。

### 10.4 唯一枚举优先级

同时保存所有 flag，但 primary decision 严格依次判定：

1. validity fail → `NO_GO_VALIDITY`；
2. validity pass，effect 或 spatial 任一 FAIL → `NO_GO_NOVELTY`；
3. 无 FAIL，但 effect 或 spatial 任一 UNCERTAIN → `INCONCLUSIVE`；
4. novelty 全 PASS，Stage D cost projection fail → `NO_GO_COST`；
5. validity、effect、spatial、cost 全 PASS → `GO`。

CI 恰等于边界属于 `INCONCLUSIVE`。没有显著差异不等于等价；本阶段未预注册 equivalence margin/TOST，因此任何枚举都不能声称 NNS 与 patch baseline 等价。

## 11. C0 对 proposal 的明确补全

C0 保留提案的 `1e-6` endpoint、`1e-4+1%` conservation、median gain `0.02`、12/15 正向、10,000 image bootstrap 等核心值，并在结果前补全了以下歧义：baseline 端使用 clean fixed A；relative direction threshold；token/CLS parity；4×4 而非单像素细粒度 fidelity；21 点 trapezoid AUC；五 seed tie；三类扰动的参数/逆变换/有效域；residual 的 5%/0.50 稳定性门槛；C8 lexicographic selection；decision 优先级；以及 compute/storage hard caps。

这些选择基于守恒、分布偏移控制、几何可逆性、避免任意 tie 和资源可审计性，而非模型表现。预算是治理上限，不是已有 timing 推导；C7 负责测量可行性，只能收紧或走正式 amendment。
