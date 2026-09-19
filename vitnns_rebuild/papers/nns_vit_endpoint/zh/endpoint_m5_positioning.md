# M5：方法语义与 reference 依赖

对注册标量目标 $F$ 与 normalized input $x$，Endpoint 为
$E(x)=x\odot\nabla_xF(x)$，不需要外部选择的 intervention reference。它并非与坐标无关：若 $x=(r-\mu)/\sigma$，则
$x_i\partial F/\partial x_i=(r_i-\mu_i)\partial F/\partial r_i$，因此 canonical Endpoint 在 raw-RGB 中以 normalization mean 为原点。对 logit 目标，Endpoint 就是 normalized-input 坐标中的 Input × Gradient。

我们复用 M1 的两种封存 reference。每图每层分别排列全部 attention heads，再对 12 个层级 Spearman 取图内均值并汇总 400 图。全部 163,200 个成分配对的 Endpoint absolute mass 和 signed sum 在两 reference 间逐项完全相同；Taylor magnitude 和 absolute finite effect 则在每一项都发生变化。9,600 个 head-block 比较中，Taylor 排名只有 46 项完全相同，有限效应排名只有 64 项完全相同。

| 模型 | Taylor 跨 reference rho | Effect 跨 reference rho | Endpoint rho：zero / blur | Taylor rho：zero / blur | Taylor−Endpoint：zero / blur |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 | 0.3782 | 0.3759 | 0.2815 / 0.2727 | 0.6742 / 0.6856 | 0.3878 / 0.4082 |
| DeiT-S/16 | 0.3000 | 0.3071 | 0.1952 / 0.1476 | 0.6095 / 0.6238 | 0.4048 / 0.4738 |

较低的跨 reference 一致性说明 intervention reference 会改变成分排序，而非只改变分数尺度。Endpoint 保持一套 observed-input 排名，再分别与 reference-specific outcome 比较；activation-space Taylor 纳入相应 displacement，在四个 model-reference 组合中均有更高的行为关联。封存 M2 数组在空间选择上给出相同的任务级模式：24 个 model-reference-block-outcome 组的 regional Taylor 中位 AUC 都高于 Endpoint，原 M2 Holm 判定不变。

| 方法 | 主要问题 | 内部标量 | 输入图 | 外部 reference | 局部 Jacobian 闭合 | 有限替换 |
|---|---|---:|---:|---:|---:|---:|
| Endpoint | 当前输入处的局部依赖 | 是 | 是 | 否 | 是 | 有限、需实证 |
| Activation-space Taylor | 指定 replacement 的一阶预测 | 是 | 区域分数 | 是 | 非主要合同 | 强匹配 |
| Integrated Gradients | baseline 到输入的路径归因 | 是 | 是 | 是 | 可扩展 | baseline difference |
| Attention rollout | 跨层 attention flow | 通常否 | 是 | 否 | 否 | 非主要合同 |
| Grad-CAM | 类别相关粗粒度定位 | 部分 | 是 | 否 | 否 | 非主要合同 |
| Chefer/AttnLRP | Transformer relevance propagation | 部分 | 是 | 通常否 | 依实现 | 非统一 |
| Attention weights | token allocation | attention 对象 | patch/token | 否 | 否 | 不能仅由权重给出 |

该比较聚焦 estimand 与信息条件。输出定位方法回答的问题不同于注册内部目标扫描，本文不主张输出定位优越性。Path IG 未进入跨模型发布：在冻结的最大 512 点下，ViT-B 的 ground-truth 与 clean-top1 校准分别只有 41/64 和 40/64 通过，低于 61/64 门。

观测 package 成本为 M1 新增采集 0.079 GPU-hours、M2 联合干预 5.982 GPU-hours。两者都共享 forward，并混合多个 predictor、控制和反事实，不能分离成可比的逐方法 runtime。操作上，Endpoint 使用当前输入 forward 和目标 backward/VJP；Taylor 另需 reference displacement；IG 评估多个路径点；有限 replacement 对候选执行反事实 forward。
