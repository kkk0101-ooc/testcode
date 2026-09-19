# Endpoint M3：NNS 忠实性与分层 token-channel 图谱

## 方法与执行状态

原始 NNS 显式逐层 learning-image 传播与 `x * grad` 在五类兼容网络、float64/float32 各 100 个随机输入上完成 5,000 个逐目标比较。全部通过；最大 relative L2 为 `2.427e-07`，最大容差比为 `1.243e-03`。11 类 ViT 算子共 88 个方向的解析 JVP、autograd JVP/VJP 和中心有限差分均通过；最大解析-autograd 相对误差 `3.431e-16`，最坏的最佳有限差分相对误差 `1.632e-10`。

双模型共享 100 张冻结图像；每模型每图审计 12×16×32=6,144 个 post-block token-channel 标量，共 `1,228,800` 个目标记录。该分层审计覆盖 ViT-B/16 全部 post-block token-channel 的 0.338% 和 DeiT-S 的 0.677%。zero-gradient 共 `0` 个，其派生比率严格记录为 NA。冻结 float32 有限差分检查中 57,600 项有 36 项未过门槛，并保留为初始 FAIL 证据。预先发布的恢复规则选中全部 10 个相关 task；其 float64 多步长复核通过，数值状态为 `PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE`。

## 等价命题与条件

定义 `L(u;x)=J_(u,x)(x) diag(x)`。若 `v=f(u_1,...,u_r)`，链式法则给出 `L(v;x)=sum_a J_(f,u_a)(x)L(u_a;x)=J_(v,x)(x)diag(x)`；对标量 `s` 即为 `E_s(x)=x*grad_x s(x)`。数值复现独立实现 Linear、Conv、ReLU、MaxPool 和 residual add 的递推。命题以模型实际接收的 normalized input 为坐标；bias 不作为图像坐标；ReLU 零点使用框架次梯度；MaxPool 使用实际 winner；等价对象是局部输入依赖图而非函数值 completeness；CLS、position embedding 和模型 bias 不分配图像输入贡献；ViT 平滑算子使用当前点真实 Jacobian。

## 图谱结果

| 模型 | 指标 | Early | Middle | Late |
|---|---|---:|---:|---:|
| vit_b16 | entropy | 0.6733 | 0.8645 | 0.8908 |
| vit_b16 | locality | 0.4887 | 0.2199 | 0.1300 |
| vit_b16 | distance | 0.1626 | 0.2565 | 0.2978 |
| vit_b16 | cancellation | 0.9962 | 0.9986 | 0.9987 |
| deit_s16 | entropy | 0.6593 | 0.8525 | 0.8783 |
| deit_s16 | locality | 0.5139 | 0.2389 | 0.1761 |
| deit_s16 | distance | 0.1531 | 0.2489 | 0.2748 |
| deit_s16 | cancellation | 0.9966 | 0.9984 | 0.9986 |

| 模型 | 冻结比较 | 中位图像级差值 | 正/负/平 | Holm p | 判定 |
|---|---|---:|---:|---:|---|
| deit_s16 | late_minus_early_entropy | 0.2250 | 100/0/0 | 7.889e-30 | SUPPORTED |
| deit_s16 | late_minus_early_locality | -0.3424 | 0/100/0 | 7.889e-30 | SUPPORTED |
| deit_s16 | late_minus_early_distance | 0.1219 | 100/0/0 | 7.889e-30 | SUPPORTED |
| deit_s16 | late_minus_early_cancellation | 0.0020 | 100/0/0 | 7.889e-30 | SUPPORTED |
| deit_s16 | cls_minus_patch_entropy | 0.1185 | 100/0/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | late_minus_early_entropy | 0.2201 | 100/0/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | late_minus_early_locality | -0.3558 | 0/100/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | late_minus_early_distance | 0.1312 | 100/0/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | late_minus_early_cancellation | 0.0025 | 100/0/0 | 7.889e-30 | SUPPORTED |
| vit_b16 | cls_minus_patch_entropy | 0.0897 | 99/1/0 | 1.593e-28 | SUPPORTED |

“由局部趋向全局”的联合判据要求 entropy 上升、locality 下降和 distance 上升同时通过 Holm：ViT-B/16 为 `SUPPORTED`，DeiT-S 为 `SUPPORTED`。该判定描述冻结抽样 token-channel 的局部一阶输入依赖。

## 神经元图相似性

| 模型 | 深度 | 配对组 | 指标 | 中位数 | 95% image-bootstrap CI |
|---|---|---|---|---:|---:|
| deit_s16 | early | different_token | patch_absolute_cosine | 0.0299 | [0.0284, 0.0321] |
| deit_s16 | early | different_token | patch_signed_cosine | -0.0000 | [-0.0000, 0.0001] |
| deit_s16 | early | same_token_different_channel | patch_absolute_cosine | 0.9948 | [0.9946, 0.9950] |
| deit_s16 | early | same_token_different_channel | patch_signed_cosine | -0.0062 | [-0.0190, 0.0061] |
| deit_s16 | late | different_token | patch_absolute_cosine | 0.3712 | [0.3530, 0.3893] |
| deit_s16 | late | different_token | patch_signed_cosine | 0.0011 | [0.0006, 0.0017] |
| deit_s16 | late | same_token_different_channel | patch_absolute_cosine | 0.9515 | [0.9465, 0.9553] |
| deit_s16 | late | same_token_different_channel | patch_signed_cosine | -0.0016 | [-0.0085, 0.0016] |
| deit_s16 | middle | different_token | patch_absolute_cosine | 0.2706 | [0.2373, 0.2909] |
| deit_s16 | middle | different_token | patch_signed_cosine | 0.0006 | [0.0002, 0.0008] |
| deit_s16 | middle | same_token_different_channel | patch_absolute_cosine | 0.9645 | [0.9592, 0.9685] |
| deit_s16 | middle | same_token_different_channel | patch_signed_cosine | 0.0049 | [-0.0016, 0.0136] |
| vit_b16 | early | different_token | patch_absolute_cosine | 0.0317 | [0.0306, 0.0346] |
| vit_b16 | early | different_token | patch_signed_cosine | -0.0000 | [-0.0001, 0.0001] |
| vit_b16 | early | same_token_different_channel | patch_absolute_cosine | 0.9953 | [0.9949, 0.9956] |
| vit_b16 | early | same_token_different_channel | patch_signed_cosine | 0.0033 | [-0.0038, 0.0286] |
| vit_b16 | late | different_token | patch_absolute_cosine | 0.4724 | [0.4543, 0.4959] |
| vit_b16 | late | different_token | patch_signed_cosine | 0.0009 | [-0.0003, 0.0016] |
| vit_b16 | late | same_token_different_channel | patch_absolute_cosine | 0.9378 | [0.9342, 0.9423] |
| vit_b16 | late | same_token_different_channel | patch_signed_cosine | 0.0024 | [-0.0056, 0.0079] |
| vit_b16 | middle | different_token | patch_absolute_cosine | 0.2761 | [0.2661, 0.2936] |
| vit_b16 | middle | different_token | patch_signed_cosine | 0.0006 | [0.0003, 0.0010] |
| vit_b16 | middle | same_token_different_channel | patch_absolute_cosine | 0.9595 | [0.9575, 0.9618] |
| vit_b16 | middle | same_token_different_channel | patch_signed_cosine | -0.0009 | [-0.0038, 0.0077] |

Absolute-distribution cosine 在 same-token/different-channel 组始终较高；different-token 组随深度增加。Signed cosine 大多接近零。以上为冻结配对的描述性结果。

## 证据边界

Endpoint 在此处表示 normalized image coordinates 上的局部 Jacobian learning image。跨模型 channel ID 不构成对应神经元。M3 评估分层 atlas；有限替换行为仍由 M1/M2 标定。
