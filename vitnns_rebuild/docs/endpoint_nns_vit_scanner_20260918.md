# NNS-ViT Endpoint Scanner

日期：2026-09-18。当前交付：`endpoint_nns_vit_scan_r3_20260918`，状态 `PASS_ENDPOINT_IMPLEMENTATION_DEMO`。

完整公式推导及与此前 path-integrated NNS-ViT、原始 NNS-CNN 的逐项对照见 [数学分析与方法对比](NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md)。

本实现将原始 NNS 的输入值初始化和终点传播原则适配到 ViT。所有目标使用同一个核心定义：

$$
L_s[i]=x_i\frac{\partial s(x)}{\partial x_i},
\qquad x=\operatorname{Normalize}(\operatorname{Crop}(\operatorname{Resize}(I))).
$$

这里的 $x$ 是实际进入模型的张量。本例重用已裁剪的 RGB 数组，按 checkpoint 的 mean/std 归一化，不重复 resize/crop。`scan_endpoint_vit` 没有 reference 参数，不运行 baseline 前向，不做路径积分。模型参数固定，所有输入相关路径正常求导，包括 LayerNorm 的均值/方差、GELU、softmax、Q/K/V、output projection、MLP 与 residual。

## 1. 传播合同

输入初始化为 $L_{x_p}[i]=x_i\mathbf 1(i=p)$。对多输入算子 $y=f(u_1,\ldots,u_r)$：

$$
L_{y_j}[i]=\sum_{a,k}\left.\frac{\partial y_j}{\partial (u_a)_k}\right|_xL_{(u_a)_k}[i].
$$

自动微分对选定目标求 VJP，得到与该逐算子传播相同的输入 learning image。扫描器只用 hooks 读取真实 timm 前向张量，每个目标做一次反向，共享一次前向。当前实现适合有限目标列表，尚未实现 batched VJP 或全模型所有神经元的密集 Jacobian。

CLS、position embedding 和各层 bias 参与真实前向，并影响 Jacobian；它们固定为模型常数，没有直接像素 seed。保存的 `local_affine_intercept=s(x)-sum(L)` 是当前点一阶 Taylor 截距，不能解释成完整 bias/constant 账本。一般没有 $\sum_iL_s[i]=s(x)$ 或 IG completeness。

`x_norm * grad_norm` 与 `x_RGB * grad_RGB` 是不同输入坐标下的对象。本例使用模型归一化输入；mean/std 均为 0.5。

## 2. 目标注册

| `kind` | 标量目标 | 语义 |
|---|---|---|
| `token_channel` | $h_{t,c}(x)$ | 单个内部单元；也用于 final CLS 固定通道 |
| `patch_channel_sum` | $\sum_{t\in\text{patches}}h_{t,c}(x)$ | 聚合内部响应，不称为单个神经元 |
| `head_channel` | $O_{h,t,c}(x)$，output projection 之前 | 固定方向 $u=e_c$ 的 head 内部目标 |
| `logit` | $\ell_y(x)$ | 类别输出 |
| `margin` | $\ell_y(x)-\ell_r(x)$ | 两类别分数差 |

本例固定 channel 0，扫描 block 0/5/11 的聚合响应和第一个 patch token 单元，block 0 的全部 12 个 head、final CLS、class logit 和 margin，共 40 个目标。各 head 的方向始终是 head 内 channel 0，不按图像重新选择方向。

未指定类别时，单图演示在 clean logits 上选 top-1 和次高类并记录具体编号；跨图同目标比较应显式冻结 `class_id`/`other_class_id`。

## 3. 同一终端目标的结构分解

内部 head 单元图与“某个 head 对最终类别目标的贡献”是不同对象。后者固定同一目标 $F(x)$，对完整输入边切面定义：

$$
E^C_{k,i}=x_i\left\langle\nabla_{z_k}F(x),\frac{\partial z_k(x)}{\partial x_i}\right\rangle,
\qquad \sum_kE^C_{k,i}=x_i\frac{\partial F(x)}{\partial x_i}.
$$

实现将下游 cotangent detach 后作为一次 VJP 的种子，不计算其导数。对第一次 residual $r=u+a(u)$：

$$
g_r=\nabla_rF,\quad
E_{\mathrm{id}}=x\odot J_u^Tg_r,\quad
E_{\mathrm{attn}}=x\odot J_a^Tg_r.
$$

identity 使用 $g_r$，不使用已包含 attention 支路的 $\nabla_uF$。第二次 residual 同理使用该加法器输出处的 cotangent。

`decompose_endpoint_vit` 输出五种独立的完整切面：

| 切面 | 重构总类别 endpoint 图的成分 |
|---|---|
| `residual1` | identity + attention |
| `residual2` | post-attention carry + MLP |
| `heads_with_bypass` | identity + 全部 head 成分 |
| `qkv_with_bypass` | identity + Query + Key + Value |
| `routing_value_with_bypass` | identity + routing + Value，其中 routing = Query + Key |

head 成分使用 concat 后、projection 前张量的非重叠 channel slices，与 concat 前 head 输入边切分等价。其下游梯度包含 output projection 和后续网络。Q/K/V 成分在原生 `attn.qkv` 输出上做非重叠切分，保留真实 softmax routing。各切面分别闭合，不能把不同切面的成分混加。

每个 component 保存 `target_id`、`component_id`、原始 `endpoint_map` 数组索引、signed sum 和 absolute mass。闭合同时报告最大绝对误差与相对 L2 误差。这些量是终点一阶微分分解，不等同于独立干预效应。

当前 edge contract 支持固定版本 timm 的标准 `Block` 和 `Attention`。本 checkpoint 没有额外输入相关 attention bias；含此类额外路径的架构需要显式加入该输入边后重新验证闭合。

## 4. 使用方式

仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .conda/envs/vitnns/bin/python vitnns_rebuild/scripts/run_endpoint_vit_scan.py \
  --output-dir vitnns_rebuild/reports/endpoint_nns_vit_my_scan
```

输出目录必须不存在。默认输入是已验证的 R9 图像裁剪数组，脚本复算其 SHA-256；模型加载同时核对 state-dict 文件 SHA-256 与 canonical tensor digest。

可用 `--input-rgb-npy` 提供自己的 `uint8 [224,224,3]` 裁剪数组，配合 `--expected-input-sha256` 绑定输入。`--blocks 0 5 11`、`--head-block 0`、`--class-id 235` 可固定扫描范围和类别。

单个内部单元与类别分解 API：

```python
from vitnns.endpoint_vit import (
    EndpointTarget, scan_endpoint_vit, decompose_endpoint_vit,
)

# x: [1,3,H,W]，实际送入 ViT 的归一化输入。
target = EndpointTarget(
    name="b5_token1_channel7", module="blocks.5",
    kind="token_channel", token=1, channel=7,
)
scan = scan_endpoint_vit(model, x, [target])
learning_image = scan.units[target.name].attribution  # 原始 signed [1,3,H,W]

terminal = EndpointTarget("class235", "", kind="logit", class_id=235)
parts = decompose_endpoint_vit(model, x, block=5, target=terminal)
```

基于已保存数组复核：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 \
  .conda/envs/vitnns/bin/python vitnns_rebuild/scripts/validate_endpoint_vit_scan.py \
  vitnns_rebuild/reports/endpoint_nns_vit_scan_r3_20260918
```

验证器复算产物/代码/模型/输入哈希、目标 scalar、`x * gradient`、signed sum、absolute mass、local intercept 和各切面的逐像素闭合。

## 5. 本轮实测

模型：`vit_base_patch16_224.augreg2_in21k_ft_in1k`。输入：`ILSVRC2012_val_00007922.JPEG` 的既有 RGB 裁剪；本次直接读取 `.npy`，没有再次解码 JPEG。

| 验证项 | 结果 |
|---|---:|
| 原始 endpoint + 新扫描器测试 | 27 passed |
| 内部目标数 | 40 |
| native logits 与无 hooks 前向最大误差 | 0 |
| 类别 endpoint 与独立 Gradient × Input 最大误差 | 0 |
| head 内部目标的 Q/K/V 闭合误差 | $5.93\times10^{-21}$ |
| 所有聚合目标 residual 闭合最大误差 | $7.06\times10^{-14}$ |
| 同一类别目标全部结构切面闭合最大误差 | $4.44\times10^{-16}$ |
| 类别结构切面最大相对 L2 误差 | $3.14\times10^{-15}$ |
| 40 目标扫描时间，CPU float64、4 threads | 4.32 s |
| 含模型加载、额外验证、三层分解及绘图的总时间 | 28.36 s |

扫描前向数 1，独立验证前向数 2，三个 block 的结构分解各用 1 次前向。baseline 前向数与积分步数均为 0。这里的计时来自单图示例，不是性能 benchmark。

27 项测试包含非退化 Conv–ReLU 原论文权重/门传播回归、非线性算子有限差分、单元/head/class 目标、两层完整切面、训练状态恢复和 hooks 清理。

本例 class 235 的 logit 为 8.08262，而 endpoint 像素和为 1.42146，明确显示它不具备输出 completeness。归因和分解的原始 float64 RGB 张量保存在 `raw_arrays.npz`，显示用 signed RGB sum，不取绝对值后冒充 signed map。

这次状态是实现与单图演示通过。此前 [20 图探索性结果](endpoint_nns_vit_level4_decision_20260917.md) 的 `NO_GO_CURRENT_ENDPOINT_PROTOCOL` 保留；本轮没有提供新的 fidelity 或稳定性优势证据。

## 6. 图像与产物

见 [Image #1–#9 逐图说明](endpoint_nns_vit_scanner_figures_20260918.md)。模块图上排为 14×14 token 数值，下排为 224×224 终点学习图像；classifier 单独采用输入 → full ViT/CLS/logit → signed learning image。

- [实现](../src/vitnns/endpoint_vit.py)
- [运行入口](../scripts/run_endpoint_vit_scan.py)
- [验证入口](../scripts/validate_endpoint_vit_scan.py)
- [实测报告](../reports/endpoint_nns_vit_scan_r3_20260918/scan_report.json)
- [哈希清单](../reports/endpoint_nns_vit_scan_r3_20260918/artifact_manifest.json)
- [原始数组](../reports/endpoint_nns_vit_scan_r3_20260918/raw_arrays.npz)

原始输入 endpoint、baseline-relative endpoint 和 Path IG 使用独立 API/名称。本轮没有移动历史模块或重命名历史图像：`endpoint_vit.py` 是主扫描接口，`endpoint_nns.py` 保留旧的显式 reference 扩展，`attribution.py` 继续提供标准 IG。
