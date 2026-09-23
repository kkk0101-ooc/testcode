# 补充材料：NNS 计算图解释与 ViT 结构路径分析（修订 v2）

[返回正文](manuscript.md)。本文件包含复现参数、完整统计、历史图索引与后续实验范围。主文 RQ 编号按修订版使用。

## S1. Model and registry details

Models: `vit_base_patch16_224.augreg2_in21k_ft_in1k` and `deit_small_patch16_224.fb_in1k`. Torch 2.2.2+cu121; timm 0.9.16; torchvision 0.17.2+cu121; NumPy 1.26.4; Pillow 10.2.0.

**Table S1. Original scalar registry coverage.**

| Model | Object | Candidate definition | Candidates | Registered/image | Coverage | Strategy | Scalar neuron |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b16 | post-block token-channel | 12*197*D | 1,815,552 | 192 | 0.01058% | 12 blocks x 4 tokens x 4 channels | yes |
| vit_b16 | post-block patch sum/energy | 2*12*D | 18,432 | 96 | 0.52083% | 12 blocks x 4 channels x 2 aggregates | no |
| vit_b16 | attention-head coordinate | 12*H*197*64 | 1,815,552 | 144 | 0.00793% | all heads; CLS token; within-head channel 0 | yes |
| vit_b16 | module patch aggregate | 3*(4D+2*4D) | 27,648 | 18 | 0.06510% | blocks 0/5/11; six modules; channel 0 | no |
| vit_b16 | patch embedding aggregate | D | 768 | 4 | 0.52083% | four fixed channels; sum over patches | no |
| vit_b16 | final CLS coordinate | D | 768 | 1 | 0.13021% | channel 0 | yes |
| vit_b16 | ground-truth logit | 1000 | 1,000 | 1 | 0.10000% | frozen image label | output |
| vit_b16 | clean-top1 logit | 1000 | 1,000 | 1 | 0.10000% | frozen clean prediction | output |
| deit_s16 | post-block token-channel | 12*197*D | 907,776 | 192 | 0.02115% | 12 blocks x 4 tokens x 4 channels | yes |
| deit_s16 | post-block patch sum/energy | 2*12*D | 9,216 | 96 | 1.04167% | 12 blocks x 4 channels x 2 aggregates | no |
| deit_s16 | attention-head coordinate | 12*H*197*64 | 907,776 | 72 | 0.00793% | all heads; CLS token; within-head channel 0 | yes |
| deit_s16 | module patch aggregate | 3*(4D+2*4D) | 13,824 | 18 | 0.13021% | blocks 0/5/11; six modules; channel 0 | no |
| deit_s16 | patch embedding aggregate | D | 384 | 4 | 1.04167% | four fixed channels; sum over patches | no |
| deit_s16 | final CLS coordinate | D | 384 | 1 | 0.26042% | channel 0 | yes |
| deit_s16 | ground-truth logit | 1000 | 1,000 | 1 | 0.10000% | frozen image label | output |
| deit_s16 | clean-top1 logit | 1000 | 1,000 | 1 | 0.10000% | frozen clean prediction | output |

**Table S2. Resources and extended validation coverage.**

| Resource / validation | Value |
| --- | --- |
| Formal scan wall time | 1,752.71 s |
| Four-card GPU-hours upper bound | 1.9475 |
| Throughput | 192.16 targets/s |
| Retained data | 9.290 GB |
| Peak reserved GPU memory | 2.284 GB |
| Peak CPU RSS | 2.018 GB |
| M1 random permutations | 128,000,000 |
| M4 attention / residual / JVP records | 86,400 / 19,200 / 691,200 |

### S1.1 实现与核心指标伪代码

模型均处于 evaluation mode。M3 使用 torch.autograd.grad 的 is_grads_batched 模式分块执行 VJP，冻结 gradient_chunk_size=16；具体随机种子、模型与图像哈希、worker 环境和每任务耗时记录在对应 evidence JSON 中。主扫描的时间、吞吐、显存与保留数据量见 Table S2；补充控制实验的 456 个任务合计 2.4502 GPU-hours。核心实现位于 src/vitnns/endpoint_m3_atlas.py、src/vitnns/endpoint_m4.py 与相应 scripts/run_endpoint_*.py。

    Endpoint(s, x):
        g <- VJP(s, x) at the observed forward state
        return x * g

    StructuredCut(F, edges, reconstruction_target):
        for each actual use-edge z_k in a valid cut:
            g_k <- downstream cotangent dF/dz_k
            E_k <- x * VJP(z_k, x; vector=g_k)
        assert sum_k E_k closes to reconstruction_target # branch contribution or Total Endpoint

    SpatialMetrics(E, target_patch):
        M[p] <- sum_rgb abs(E[p, rgb])
        if sum_p M[p] == 0: return NA
        p_patch <- patch_sum(M) / sum_p M[p]
        entropy <- -sum_j p_patch[j] log(p_patch[j]) / log(196)
        locality <- neighborhood_mass(target_patch) - uniform_neighborhood_mass
        distance <- sum_j p_patch[j] normalized_distance(j, target_patch)

    RandomRegionAUC(repeats):
        for each random repeat r:
            curve_r <- absolute finite effect at every area budget
            auc_r <- trapezoid(curve_r, area in [0, 0.25]) / 0.25
        return mean_r auc_r

CLS 没有自然的目标 patch，因此 locality 与 target-relative distance 仅对 patch-token 目标定义；零质量样本传播为 NA。图内 token、channel 与 head 先按预定规则汇总，再以图像为统计单位。

## S2. Frozen decisions and numerical diagnostics

C4a: BROAD_BUT_NON_UNIVERSAL_FAMILY_LEVEL_SUPPORT; original all-family omnibus NOT_SUPPORTED. C4b: NOT_SUPPORTED. The numerical atlas status is PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE: 36/57,600 float32 failures are retained and 36/36 same-coordinate float64 checks pass. These fields are not interchangeable.

**Table S2.1. 原 pooled C4a provenance；图像级中位数 [95% CI]。** Holm p 来自原成分和区域合计 28 项的冻结 family。

| Model | Reference | Heads | A/V pooled | Residual pooled |
| --- | --- | --- | --- | --- |
| vit_b16 | normalized-zero | 0.291 [0.281, 0.298] | 0.0996 [0.075, 0.123] | 0.177 [0.155, 0.195] |
| vit_b16 | blur | 0.306 [0.296, 0.318] | 0.157 [0.131, 0.177] | 0.209 [0.182, 0.237] |
| deit_s16 | normalized-zero | 0.191 [0.177, 0.205] | 0.107 [0.078, 0.129] | 0.135 [0.113, 0.164] |
| deit_s16 | blur | 0.182 [0.168, 0.195] | 0.160 [0.128, 0.185] | 0.229 [0.204, 0.251] |

12 项原 Holm 调整后 p 均小于 $10^{-6}$。原 composite 中 11/12 family 达到操作门槛，但 all-family C4a 仍为 `NOT_SUPPORTED`。这张表保存历史判定身份；S18 的分层诊断不追溯改变它。

## S3. Reference sensitivity

Change uses atol=1e-12 and rtol=0. Relative change is abs(q0-qblur)/max(abs(q0),abs(qblur)); both-zero denominators are NA. Component quantiles describe the pooled records, not independent inferential samples.

**Table S3. Changes in Taylor magnitude and absolute finite effect.**

| Model | Quantity | Component pairs | Above tolerance | Abs change median [q10,q90] | Relative change median [q10,q90] |
| --- | --- | --- | --- | --- | --- |
| DeiT-S/16 | finite_effect | 67200 | 100.0% | 0.1365 [0.0161, 1.6575] | 0.4927 [0.0985, 0.8997] |
| DeiT-S/16 | taylor | 67200 | 100.0% | 0.1215 [0.0150, 0.7867] | 0.5186 [0.1172, 0.9033] |
| ViT-B/16 | finite_effect | 96000 | 100.0% | 0.0713 [0.0076, 1.4414] | 0.5300 [0.1117, 0.9137] |
| ViT-B/16 | taylor | 96000 | 100.0% | 0.0688 [0.0078, 0.5471] | 0.5225 [0.1133, 0.9052] |

Head rankings use average ranks and Kendall tau-b. Top-K ties are broken by ascending component ID; Top-3 overlap is intersection size / 3. Rank displacement uses original rank positions, so model head counts affect its scale. Each image averages 12 blocks, then summaries use 400 images. Full-order equality counts are Taylor 0/4,800 and effect 0/4,800 for ViT-B, Taylor 46/4,800 and effect 64/4,800 for DeiT-S.

**Table S4. Image-level rank concordance distributions.**

| Model | Quantity | Metric | Image median [q10,q90] | n / NA |
| --- | --- | --- | --- | --- |
| DeiT-S/16 | finite_effect | spearman | 0.3071 [0.1238, 0.4524] | 400 / 0 |
| DeiT-S/16 | finite_effect | kendall_tau_b | 0.2444 [0.0989, 0.3678] | 400 / 0 |
| DeiT-S/16 | finite_effect | top1_agreement | 0.4167 [0.1667, 0.5833] | 400 / 0 |
| DeiT-S/16 | finite_effect | top3_overlap | 0.6111 [0.5278, 0.6944] | 400 / 0 |
| DeiT-S/16 | finite_effect | mean_absolute_rank_displacement | 1.5278 [1.2778, 1.7528] | 400 / 0 |
| DeiT-S/16 | taylor | spearman | 0.3000 [0.1000, 0.4476] | 400 / 0 |
| DeiT-S/16 | taylor | kendall_tau_b | 0.2389 [0.0778, 0.3667] | 400 / 0 |
| DeiT-S/16 | taylor | top1_agreement | 0.3333 [0.1667, 0.5833] | 400 / 0 |
| DeiT-S/16 | taylor | top3_overlap | 0.6111 [0.5278, 0.6944] | 400 / 0 |
| DeiT-S/16 | taylor | mean_absolute_rank_displacement | 1.5278 [1.3056, 1.8056] | 400 / 0 |
| ViT-B/16 | finite_effect | spearman | 0.3759 [0.2622, 0.4904] | 400 / 0 |
| ViT-B/16 | finite_effect | kendall_tau_b | 0.2803 [0.1967, 0.3740] | 400 / 0 |
| ViT-B/16 | finite_effect | top1_agreement | 0.3333 [0.2500, 0.5833] | 400 / 0 |
| ViT-B/16 | finite_effect | top3_overlap | 0.5000 [0.4167, 0.6111] | 400 / 0 |
| ViT-B/16 | finite_effect | mean_absolute_rank_displacement | 2.9444 [2.6111, 3.2917] | 400 / 0 |
| ViT-B/16 | taylor | spearman | 0.3782 [0.2539, 0.4914] | 400 / 0 |
| ViT-B/16 | taylor | kendall_tau_b | 0.2828 [0.1942, 0.3737] | 400 / 0 |
| ViT-B/16 | taylor | top1_agreement | 0.3333 [0.2500, 0.5833] | 400 / 0 |
| ViT-B/16 | taylor | top3_overlap | 0.5000 [0.4167, 0.6111] | 400 / 0 |
| ViT-B/16 | taylor | mean_absolute_rank_displacement | 2.9444 [2.5972, 3.2792] | 400 / 0 |

## S4. Method objects

**Table S5. Functional comparison.**

| Method | Primary object | External reference | Propagation / contract |
| --- | --- | --- | --- |
| Endpoint | Normalized-input Total Endpoint | No; normalization origin matters | Global cuts close to total; branch partitions close to branch |
| Activation-space Taylor | Specified replacement effect | Yes | Clean cotangent × displacement |
| Integrated Gradients | Baseline-to-input attribution | Yes | Path-average gradient; baseline-difference completeness |
| Attention rollout | Token flow | No | Accumulated attention / residual flow |
| Grad-CAM | Class-conditioned coarse localization | No | Gradient-weighted feature maps |
| Chefer / AttnLRP | Transformer relevance | Implementation-specific | Method-specific relevance rules |
| Raw attention | Token allocation | No | Forward attention weights |

**Table S5a. 完整近邻差分矩阵。** “未报告”限于参考文献表所列核验版本；不同方法的 closure 对象不可互换。

| Method family | Explained object | Consumer edge/use | Spatial component map | Closure object | Finite intervention |
| --- | --- | --- | --- | --- | --- |
| Gradient × Input | 当前点输入加权梯度 | 未报告 | 单一输入图 | 一般不满足函数值完整性 | 否 |
| Integrated Gradients | reference-to-input 路径归因 | 未报告 | 单一输入图 | 输出差 | 否 |
| Conductance | 经过隐藏单元的路径归因 | 隐藏单元集合 | 可选输入分辨率 | 层级输出差 | 否 |
| GlobEnc / ALTI | 跨层 token contribution/mixing | 算子级 | token 图 | 局部前向分解 | 否 |
| DecompX | representation 与 logit 的 token components | 组件级 | token 分项 | 依 bias 分配规则重组 | 否 |
| AttnLRP / HiLRP / GradSkip | Transformer relevance | 算子连接 | relevance 图 | relevance conservation | 否 |
| EAP / AtP* / Vi-CD | reference-specific component/edge effect | 是 | 通常不保留本文的输入坐标分项 | patching effect 或 circuit fidelity | 是 |
| Activation patching / CAAP | 内部 activation 替换效应 | node 或指定 activation | patch score | 一般不加和 | 是 |
| DAVE | ViT input-gradient 分解 | 未报告 consumer-use 索引 | 像素图 | effective input-output map | 否 |
| Unpack | component interaction、命名路径与 token credit | 组件/路径身份 | token attribution | 方法特定 credit decomposition | 否 |
| NNS–ViT | 指定 use 与输入位置的同一当前点局部微分 | consumer、slot、invocation | 分项输入图 | 声明 branch 或 global cut 的同一 signed differential | 分开评价 |

## S4.1 中英术语合同

| 中文 | English | 定义边界 |
| --- | --- | --- |
| 结构合法切面 | structurally valid cut | 每条输入到目标的结构 use-path 恰穿一次，与当前点数值无关 |
| 点态微分闭合 | pointwise differential closure | 固定输入处 use-specific Jacobian/cotangent 分组之和 |
| 分支微分分组 | branch differential partition | 仅重构 attention 或 MLP branch contribution |
| 全局切面 | global cut | 包含 bypass 并重构同目标 Total Endpoint |
| 路径图 | path map | 指定 tensor use 的局部微分分项，不是唯一因果路径 |
| 局部微分绝对质量比 | local-differential absolute-mass ratio | routing/identity scale；不是因果贡献率 |
| 区域诱导的指定-head activation 替换 | region-induced replacement of designated head activations | 混合输入产生 head activation，再插入 clean forward 的两阶段任务 |
| 候选集合遗憾率 | candidate-set regret | 相对已评价候选集合包络，不是全局最优策略 regret |

## S5. Corrected absolute random AUC and oracle

当前随机基线对每次重复先取绝对值并积分，再平均 AUC。修正结果、重复分布和 MC 稳定性见[supplementary experiments S9](supplement_experiments_v1.md). 旧分析作为原估计器的历史复核保留。

## S6. Detailed attention/residual evidence (historical stage identifiers retained)


### 历史 M4：Attention routing/value 与 residual 机制

M4 完成两模型共享 400 图、12 个 blocks 和全部 heads 的 ground-truth-logit 局部输入依赖分解，共 86,400 个 image-block-head attention records、19,200 个 image-block-residual records 和 691,200 个 direction-head JVP records。Q+K+V 逐 head 重构 native attention Endpoint；routing+value 重构同一对象；attention 和 MLP residual 的 identity/carry 加 branch 均重构完整 logit Endpoint。

独立 CPU 审计复算 230,400 个 E3 summary 字段。最大 Q/K/V closure 绝对误差为 1.574e-05，最大 residual closure 绝对误差为 2.480e-05，显式 attention differential 与 native JVP 的最大相对误差为 2.502e-05。全部 19,200 个省略 identity/carry 负控按预期闭合失败；冻结前 16 个 evaluation orders 的 384 个 attention-detach block checks 通过。

#### Attention routing/value

| 模型 | Block | Routing share | Frozen-attention Endpoint error | Frozen-attention JVP error | Mean-query saturation | Routing-value path cancellation |
|---|---:|---:|---:|---:|---:|---:|
| vit_b16 | 0 | 0.3092 | 0.4407 | 0.4508 | 0.4656 | 0.1555 |
| vit_b16 | 5 | 0.5239 | 0.7792 | 0.8547 | 0.3183 | 0.2342 |
| vit_b16 | 11 | 0.5154 | 0.8022 | 0.9516 | 0.3470 | 0.3069 |
| deit_s16 | 0 | 0.2041 | 0.2774 | 0.3099 | 0.3825 | 0.1377 |
| deit_s16 | 5 | 0.4852 | 0.7375 | 0.8548 | 0.3191 | 0.2635 |
| deit_s16 | 11 | 0.4808 | 0.8488 | 0.9636 | 0.3329 | 0.3692 |

Frozen-attention 只保留 A dV。因此 Endpoint 与 JVP error 分别测量它在输入缩放方向和 8 个冻结 Rademacher 方向上遗漏的 routing differential。以下检验先在图内按层和 head 聚合，再以图像为统计单位；Holm family 按模型冻结。

| 模型 | Domain | 类型 | 冻结检验 | 图像级中位数 | 正/负/平 | Holm p | 判定 |
|---|---|---|---|---:|---:|---:|---|
| vit_b16 | attention | attention-head | late_minus_early_routing_share | 0.0523 | 360/40/0 | 1.715e-65 | SUPPORTED_DIFFERENCE |
| vit_b16 | attention | attention-head | late_minus_early_frozen_attention_endpoint_error | 0.1444 | 383/17/0 | 5.545e-91 | SUPPORTED_DIFFERENCE |
| vit_b16 | attention | attention-head | saturation_routing_relation | -0.2937 | 5/395/0 | 2.611e-109 | SUPPORTED_DIFFERENCE |
| vit_b16 | attention | attention-head | saturation_error_relation | -0.2238 | 12/388/0 | 7.108e-98 | SUPPORTED_DIFFERENCE |
| vit_b16 | residual | attention | late_minus_early_identity_share | -0.1242 | 38/362/0 | 4.068e-67 | SUPPORTED_DIFFERENCE |
| vit_b16 | residual | attention | late_minus_early_path_cancellation | -0.0105 | 173/227/0 | 7.970e-03 | SUPPORTED_DIFFERENCE |
| vit_b16 | residual | mlp | late_minus_early_identity_share | 0.1798 | 396/4/0 | 1.644e-111 | SUPPORTED_DIFFERENCE |
| vit_b16 | residual | mlp | late_minus_early_path_cancellation | 0.0688 | 304/96/0 | 2.840e-26 | SUPPORTED_DIFFERENCE |
| deit_s16 | attention | attention-head | late_minus_early_routing_share | 0.1355 | 399/1/0 | 9.317e-118 | SUPPORTED_DIFFERENCE |
| deit_s16 | attention | attention-head | late_minus_early_frozen_attention_endpoint_error | 0.2896 | 400/0/0 | 3.098e-120 | SUPPORTED_DIFFERENCE |
| deit_s16 | attention | attention-head | saturation_routing_relation | -0.3143 | 15/379/6 | 5.158e-92 | SUPPORTED_DIFFERENCE |
| deit_s16 | attention | attention-head | saturation_error_relation | -0.2286 | 52/338/10 | 1.794e-52 | SUPPORTED_DIFFERENCE |
| deit_s16 | residual | attention | late_minus_early_identity_share | -0.0449 | 48/352/0 | 6.053e-58 | SUPPORTED_DIFFERENCE |
| deit_s16 | residual | attention | late_minus_early_path_cancellation | -0.0074 | 171/229/0 | 4.314e-03 | SUPPORTED_DIFFERENCE |
| deit_s16 | residual | mlp | late_minus_early_identity_share | 0.1780 | 397/3/0 | 1.652e-113 | SUPPORTED_DIFFERENCE |
| deit_s16 | residual | mlp | late_minus_early_path_cancellation | -0.0588 | 103/297/0 | 6.661e-23 | SUPPORTED_DIFFERENCE |

Saturation 关系是在每张图、每一层的 heads 内计算 Spearman 后再汇总，避免把共同层深趋势当成 head 内关系。其方向与支持状态由上表直接给出，不预设 softmax 越尖锐时 routing omission 越大。

#### Residual identity/branch

| 模型 | Residual | Block | Identity share | Path cancellation | Identity-branch cosine | Output-identity patch TV |
|---|---|---:|---:|---:|---:|---:|
| vit_b16 | attention | 0 | 0.8847 | 0.1465 | -0.3315 | 0.0150 |
| vit_b16 | attention | 5 | 0.6100 | 0.2381 | 0.0606 | 0.1089 |
| vit_b16 | attention | 11 | 0.4442 | 0.3578 | -0.0506 | 0.1803 |
| vit_b16 | mlp | 0 | 0.2225 | 0.1654 | 0.1849 | 0.1398 |
| vit_b16 | mlp | 5 | 0.5648 | 0.2450 | 0.1321 | 0.1287 |
| vit_b16 | mlp | 11 | 0.6418 | 0.6333 | -0.9071 | 0.1023 |
| deit_s16 | attention | 0 | 0.8729 | 0.1820 | -0.4181 | 0.0284 |
| deit_s16 | attention | 5 | 0.6775 | 0.1983 | 0.1826 | 0.0798 |
| deit_s16 | attention | 11 | 0.6559 | 0.2538 | 0.0757 | 0.1047 |
| deit_s16 | mlp | 0 | 0.3078 | 0.2831 | -0.0186 | 0.1798 |
| deit_s16 | mlp | 5 | 0.6257 | 0.1886 | 0.3342 | 0.0923 |
| deit_s16 | mlp | 11 | 0.6079 | 0.5878 | -0.7599 | 0.1276 |

Identity share 描述 absolute Endpoint mass 在 skip 与 branch 两条局部路径间的比例；path cancellation 比较两条 signed map 相加前后的 absolute mass。Output-identity patch TV 衡量 residual 输出相对只走 identity 时的空间依赖改变。ViT-B/16 中 attention/MLP residual 的最小与最大空间 TV 位于 blocks 0/11 和 8/10；DeiT-S 对应为 1/4 和 9/0。这些层级定位来自冻结 cohort 的描述性中位数。

| 模型 | Phase | Attention−MLP identity share | Attention−MLP path cancellation | Attention−MLP spatial TV |
|---|---|---:|---:|---:|
| vit_b16 | early | 0.3382 | 0.0004 | -0.0902 |
| vit_b16 | middle | 0.0327 | 0.0165 | -0.0050 |
| vit_b16 | late | 0.0376 | -0.0664 | -0.0173 |
| deit_s16 | early | 0.2697 | -0.0564 | -0.0796 |
| deit_s16 | middle | 0.0280 | 0.0009 | -0.0026 |
| deit_s16 | late | 0.0512 | -0.0041 | -0.0047 |

| 模型 | Residual | Branch share–path cancellation Spearman（跨 12 blocks） |
|---|---|---:|
| vit_b16 | attention | 0.5140 |
| vit_b16 | mlp | -0.2657 |
| deit_s16 | attention | 0.3182 |
| deit_s16 | mlp | 0.4650 |

前一表直接比较 attention 与 MLP residual；后一表回答 branch mass 较大时 path cancellation 是否一并变化。两表均为冻结 A1 描述性分析，不加入原 M4 Holm families。

#### Signed、absolute 与 cancellation

每条路径均同时保留 signed sum、absolute mass、positive/negative mass 和 map cancellation。M4 区分同一 map 内的正负抵消与 routing/value 或 identity/branch 之间的 path cancellation；二者没有合并为单一“重要性”指标。所有零分母、零范数和常数 head 关系严格传播为 NA，没有加入 epsilon。

| 模型 | 路径 | Block | Signed magnitude | Absolute mass | Map cancellation |
|---|---|---:|---:|---:|---:|
| vit_b16 | routing | 0 | 0.0362 | 2.6596 | 0.9867 |
| vit_b16 | routing | 5 | 0.0600 | 33.7856 | 0.9982 |
| vit_b16 | routing | 11 | 0.0608 | 40.7842 | 0.9982 |
| vit_b16 | value | 0 | 0.0306 | 7.6041 | 0.9952 |
| vit_b16 | value | 5 | 0.0380 | 28.0522 | 0.9986 |
| vit_b16 | value | 11 | 0.0654 | 39.0241 | 0.9981 |
| vit_b16 | attention_identity | 0 | 1.3315 | 495.2271 | 0.9972 |
| vit_b16 | attention_identity | 5 | 0.9816 | 387.1208 | 0.9977 |
| vit_b16 | attention_identity | 11 | 0.8778 | 313.3689 | 0.9974 |
| vit_b16 | attention_branch | 0 | 0.3930 | 63.8344 | 0.9934 |
| vit_b16 | attention_branch | 5 | 0.4140 | 250.3045 | 0.9983 |
| vit_b16 | attention_branch | 11 | 0.6133 | 398.0202 | 0.9986 |
| vit_b16 | mlp_identity | 0 | 0.6860 | 130.4998 | 0.9945 |
| vit_b16 | mlp_identity | 5 | 1.0976 | 334.0699 | 0.9971 |
| vit_b16 | mlp_identity | 11 | 1.8214 | 827.8407 | 0.9977 |
| vit_b16 | mlp_branch | 0 | 0.6309 | 438.0538 | 0.9986 |
| vit_b16 | mlp_branch | 5 | 0.4504 | 271.9139 | 0.9983 |
| vit_b16 | mlp_branch | 11 | 0.7961 | 460.2701 | 0.9981 |
| deit_s16 | routing | 0 | 0.0756 | 4.5294 | 0.9829 |
| deit_s16 | routing | 5 | 0.0506 | 38.3865 | 0.9986 |
| deit_s16 | routing | 11 | 0.0504 | 35.9257 | 0.9986 |
| deit_s16 | value | 0 | 0.0818 | 18.4261 | 0.9950 |
| deit_s16 | value | 5 | 0.0776 | 40.1955 | 0.9979 |
| deit_s16 | value | 11 | 0.0586 | 38.8767 | 0.9985 |
| deit_s16 | attention_identity | 0 | 1.6888 | 459.3077 | 0.9960 |
| deit_s16 | attention_identity | 5 | 1.1087 | 370.3635 | 0.9969 |
| deit_s16 | attention_identity | 11 | 1.1539 | 372.6598 | 0.9966 |
| deit_s16 | attention_branch | 0 | 0.4543 | 63.9204 | 0.9922 |
| deit_s16 | attention_branch | 5 | 0.4112 | 169.7186 | 0.9974 |
| deit_s16 | attention_branch | 11 | 0.3639 | 185.3893 | 0.9980 |
| deit_s16 | mlp_identity | 0 | 0.8741 | 170.6143 | 0.9946 |
| deit_s16 | mlp_identity | 5 | 1.1847 | 317.7498 | 0.9962 |
| deit_s16 | mlp_identity | 11 | 1.7620 | 614.5721 | 0.9973 |
| deit_s16 | mlp_branch | 0 | 0.7619 | 412.5634 | 0.9981 |
| deit_s16 | mlp_branch | 5 | 0.3466 | 197.6662 | 0.9982 |
| deit_s16 | mlp_branch | 11 | 0.4532 | 354.6503 | 0.9988 |

#### Review-R2 敏感性与 estimand 收紧

Review-R2 检验 late-block Value-only JVP discrepancy 是否由小分母造成。Block 11 的
relative discrepancy 中位数在 ViT-B/16 和 DeiT-S/16 中为 0.9516 和 0.9636；对应的
full-response denominator 为 14.9711 和 1.5155，absolute discrepancy 为 13.8064 和
1.4228。因此，差异与完整响应处于相同 norm 尺度，并非普遍来自接近零的分母。

![M4 Review Archived R2 Figure 1](../supplement/S77_supp_m4_review_01_m4_review_jvp_diagnostics.png)

**M4 Review Archived R2 Figure 1.** 两模型 block 11 的 normalized JVP discrepancy、absolute
discrepancy 和 full-response denominator 分布。

控制 head projection-input activation norm 和 downstream-cotangent norm 后，两模型的
mean-query saturation 仍与 routing share、Endpoint error 和 JVP error 呈负相关。ViT-B/16
与 DeiT-S/16 的 fully adjusted correlations 依次为 −0.1442/−0.3257、
−0.1129/−0.2185 和 −0.1453/−0.2786。CLS-query 关系更弱：ViT-B 的 adjusted routing
share 和 Endpoint-error 检验均为 Holm $p=0.0629$。因此正式 summary 使用 all-query mean
saturation，并把关联主要定位在 early/middle blocks。

![M4 Review Archived R2 Figure 2](../supplement/S78_supp_m4_review_02_m4_review_saturation_robustness.png)

**M4 Review Archived R2 Figure 2.** Saturation 的 unadjusted/partial-rank 关系、phase sensitivity 与较弱的
CLS-query 结果。

全 cohort 空间分析使用 output 与 identity 的归一化 14×14 patch absolute distributions
之间的 total variation。Attention residual 的 late-minus-early 差在 ViT-B/16 和 DeiT-S/16
中为 +0.0520（[0.0488, 0.0547]）和 +0.0282（[0.0257, 0.0300]）；MLP residual 为
−0.0187（[−0.0220, −0.0150]）和 −0.0462（[−0.0516, −0.0412]）。四项 post-hoc
sensitivity tests 全部通过 Holm 校正，branch-relative path scale 与 spatial TV 在图内跨层
也呈正相关。

![M4 Review Archived R2 Figure 3](../supplement/S79_supp_m4_review_03_m4_review_residual_spatial.png)

**M4 Review Archived R2 Figure 3.** Attention 与 MLP 子层的全 cohort residual spatial-change sensitivity。

在每模型 16 张事前保留的 class-logit maps 上，Full–Value patch TV 从 block 0 的
0.0550/0.0307 增至 block 11 的 0.1865/0.1811。其 12-block trajectory 与 M3 entropy、
locality 和 distance 方向相容；但 M3 使用 100 图 token-channel targets，且每模型仅一张图
重叠。因此这里只形成模型级 trajectory bridge，不构成 routing 导致空间扩展的逐单元或
因果联系。

![M4 Review Archived R2 Figure 4](../supplement/S80_supp_m4_review_04_m4_review_m3_m4_bridge.png)

**M4 Review Archived R2 Figure 4.** M3 空间轨迹与 Full–Value class-logit map 差异之间的描述性桥接。

#### 图

- Image #1（m4_attention_depth）：逐 block 展示 routing share、frozen-attention Endpoint error 与 8-direction JVP error；阴影为 10,000 次图像 bootstrap 95% CI。
- Image #2（m4_attention_saturation）：展示 block-level saturation 与 routing/error 的描述性位置；正式相关检验仍使用图内 layer-wise head Spearman。
- Image #3（m4_residual_depth）：同时展示 attention/MLP residual 的 identity share、path cancellation 和空间 TV。

#### 解释边界

M4 定量描述当前输入点和冻结 ground-truth logit 下的局部 Jacobian 路径记账。较高 routing share 或 branch mass 表示该路径承载较多局部输入依赖；因果重要性仍需独立有限干预证据。


## S7. Complete supplementary image index

- [S01 — S01_figure_01_endpoint_scanner_method_overview](../supplement/S01_figure_01_endpoint_scanner_method_overview.png)
- [S02 — S02_figure_02_endpoint_cnn_and_path_method_identity](../supplement/S02_figure_02_endpoint_cnn_and_path_method_identity.png)
- [S03 — S03_figure_03_same_target_endpoint_versus_path](../supplement/S03_figure_03_same_target_endpoint_versus_path.png)
- [S04 — S04_figure_04_original_cnn_endpoint_gates](../supplement/S04_figure_04_original_cnn_endpoint_gates.png)
- [S05 — S05_figure_05_attention_q_k_v_differential_paths](../supplement/S05_figure_05_attention_q_k_v_differential_paths.png)
- [S06 — S06_figure_06_residual_edge_accounting](../supplement/S06_figure_06_residual_edge_accounting.png)
- [S07 — S07_figure_07_unit_sum_and_energy_targets](../supplement/S07_figure_07_unit_sum_and_energy_targets.png)
- [S08 — S08_figure_08_state_seed_and_input_coordinates](../supplement/S08_figure_08_state_seed_and_input_coordinates.png)
- [S09 — S09_figure_09_layernorm_and_gelu_jacobians](../supplement/S09_figure_09_layernorm_and_gelu_jacobians.png)
- [S10 — S10_figure_10_structured_edge_cuts](../supplement/S10_figure_10_structured_edge_cuts.png)
- [S11 — S11_figure_11_registered_target_atlas_design](../supplement/S11_figure_11_registered_target_atlas_design.png)
- [S12 — S12_figure_12_sloth_bear_class_logit_maps_across_models](../supplement/S12_figure_12_sloth_bear_class_logit_maps_across_models.png)
- [S13 — S13_figure_13_sloth_bear_multi_target_atlas](../supplement/S13_figure_13_sloth_bear_multi_target_atlas.png)
- [S14 — S14_figure_14_scan_scale_latency_and_resources](../supplement/S14_figure_14_scan_scale_latency_and_resources.png)
- [S15 — S15_figure_15_structural_closure_contracts](../supplement/S15_figure_15_structural_closure_contracts.png)
- [S16 — S16_figure_16_formal_closure_and_negative_controls](../supplement/S16_figure_16_formal_closure_and_negative_controls.png)
- [S17 — S17_figure_17_sloth_bear_structural_component_maps](../supplement/S17_figure_17_sloth_bear_structural_component_maps.png)
- [S18 — S18_figure_18_condition_scaled_analytic_jvp_checks](../supplement/S18_figure_18_condition_scaled_analytic_jvp_checks.png)
- [S19 — S19_figure_19_matched_finite_activation_replacement](../supplement/S19_figure_19_matched_finite_activation_replacement.png)
- [S20 — S20_figure_20_c4a_family_estimates_and_intervals](../supplement/S20_figure_20_c4a_family_estimates_and_intervals.png)
- [S21 — S21_figure_21_rq3_threshold_zoom_and_boundary_exception](../supplement/S21_figure_21_rq3_threshold_zoom_and_boundary_exception.png)
- [S22 — S22_figure_22_endpoint_and_registered_secondary_predictors](../supplement/S22_figure_22_endpoint_and_registered_secondary_predictors.png)
- [S23 — S23_figure_23_spatial_intervention_protocol](../supplement/S23_figure_23_spatial_intervention_protocol.png)
- [S24 — S24_figure_24_sloth_bear_block_5_maps_for_spatial_ranking](../supplement/S24_figure_24_sloth_bear_block_5_maps_for_spatial_ranking.png)
- [S25 — S25_figure_25_one_image_spatial_effect_curves](../supplement/S25_figure_25_one_image_spatial_effect_curves.png)
- [S26 — S26_figure_26_c4b_mandatory_baseline_contrasts](../supplement/S26_figure_26_c4b_mandatory_baseline_contrasts.png)
- [S27 — S27_figure_27_rq4_family_summary](../supplement/S27_figure_27_rq4_family_summary.png)
- [S28 — S28_figure_28_identity_correctness_utility_conclusion](../supplement/S28_figure_28_identity_correctness_utility_conclusion.png)
- [S29 — S29_figure_29_m3_block_trajectories](../supplement/S29_figure_29_m3_block_trajectories.png)
- [S30 — S30_figure_30_m3_cls_minus_patch_entropy_by_block](../supplement/S30_figure_30_m3_cls_minus_patch_entropy_by_block.png)
- [S31 — S31_figure_31_m3_cancellation_scales](../supplement/S31_figure_31_m3_cancellation_scales.png)
- [S32 — S32_figure_32_original_nns_versus_endpoint_numerical_equivalence](../supplement/S32_figure_32_original_nns_versus_endpoint_numerical_equivalence.png)
- [S33 — S33_figure_33_m3_finite_difference_diagnostics](../supplement/S33_figure_33_m3_finite_difference_diagnostics.png)
- [S34 — S34_figure_34_registered_target_learning_images](../supplement/S34_figure_34_registered_target_learning_images.png)
- [S35 — S35_figure_35_block5_structural_learning_images](../supplement/S35_figure_35_block5_structural_learning_images.png)
- [S36 — S36_figure_36_spatial_ranking_learning_images](../supplement/S36_figure_36_spatial_ranking_learning_images.png)
- [S37 — S37_figure_37_channel0_depth_progression](../supplement/S37_figure_37_channel0_depth_progression.png)
- [S38 — S38_figure_38_attention_head_depth_atlas](../supplement/S38_figure_38_attention_head_depth_atlas.png)
- [S39 — S39_figure_39_vit_b16_attention_head_depth_atlas](../supplement/S39_figure_39_vit_b16_attention_head_depth_atlas.png)
- [S40 — S40_figure_40_vit_b16_module_chain](../supplement/S40_figure_40_vit_b16_module_chain.png)
- [S41 — S41_figure_12_sloth_bear_class_logit_maps_across_models_signed](../supplement/S41_figure_12_sloth_bear_class_logit_maps_across_models_signed.png)
- [S42 — S42_figure_13_sloth_bear_multi_target_atlas_signed](../supplement/S42_figure_13_sloth_bear_multi_target_atlas_signed.png)
- [S43 — S43_figure_17_sloth_bear_structural_component_maps_signed](../supplement/S43_figure_17_sloth_bear_structural_component_maps_signed.png)
- [S44 — S44_figure_24_sloth_bear_block_5_maps_for_spatial_ranking_signed](../supplement/S44_figure_24_sloth_bear_block_5_maps_for_spatial_ranking_signed.png)
- [S45 — S45_block_00](../supplement/S45_block_00.png)
- [S46 — S46_block_01](../supplement/S46_block_01.png)
- [S47 — S47_block_02](../supplement/S47_block_02.png)
- [S48 — S48_block_03](../supplement/S48_block_03.png)
- [S49 — S49_block_04](../supplement/S49_block_04.png)
- [S50 — S50_block_05](../supplement/S50_block_05.png)
- [S51 — S51_block_06](../supplement/S51_block_06.png)
- [S52 — S52_block_07](../supplement/S52_block_07.png)
- [S53 — S53_block_08](../supplement/S53_block_08.png)
- [S54 — S54_block_09](../supplement/S54_block_09.png)
- [S55 — S55_block_10](../supplement/S55_block_10.png)
- [S56 — S56_block_11](../supplement/S56_block_11.png)
- [S57 — S57_block_00](../supplement/S57_block_00.png)
- [S58 — S58_block_01](../supplement/S58_block_01.png)
- [S59 — S59_block_02](../supplement/S59_block_02.png)
- [S60 — S60_block_03](../supplement/S60_block_03.png)
- [S61 — S61_block_04](../supplement/S61_block_04.png)
- [S62 — S62_block_05](../supplement/S62_block_05.png)
- [S63 — S63_block_06](../supplement/S63_block_06.png)
- [S64 — S64_block_07](../supplement/S64_block_07.png)
- [S65 — S65_block_08](../supplement/S65_block_08.png)
- [S66 — S66_block_09](../supplement/S66_block_09.png)
- [S67 — S67_block_10](../supplement/S67_block_10.png)
- [S68 — S68_block_11](../supplement/S68_block_11.png)
- [S69 — S69_supp_m2_01_M2_F1](../supplement/S69_supp_m2_01_M2_F1.png)
- [S70 — S70_supp_m2_02_M2_F2](../supplement/S70_supp_m2_02_M2_F2.png)
- [S71 — S71_supp_m2_03_M2_F3](../supplement/S71_supp_m2_03_M2_F3.png)
- [S72 — S72_supp_m3_01_m3_atlas_depth](../supplement/S72_supp_m3_01_m3_atlas_depth.png)
- [S73 — S73_supp_m3_02_m3_map_similarity](../supplement/S73_supp_m3_02_m3_map_similarity.png)
- [S74 — S74_supp_m4_01_m4_attention_depth](../supplement/S74_supp_m4_01_m4_attention_depth.png)
- [S75 — S75_supp_m4_02_m4_attention_saturation](../supplement/S75_supp_m4_02_m4_attention_saturation.png)
- [S76 — S76_supp_m4_03_m4_residual_depth](../supplement/S76_supp_m4_03_m4_residual_depth.png)
- [S77 — S77_supp_m4_review_01_m4_review_jvp_diagnostics](../supplement/S77_supp_m4_review_01_m4_review_jvp_diagnostics.png)
- [S78 — S78_supp_m4_review_02_m4_review_saturation_robustness](../supplement/S78_supp_m4_review_02_m4_review_saturation_robustness.png)
- [S79 — S79_supp_m4_review_03_m4_review_residual_spatial](../supplement/S79_supp_m4_review_03_m4_review_residual_spatial.png)
- [S80 — S80_supp_m4_review_04_m4_review_m3_m4_bridge](../supplement/S80_supp_m4_review_04_m4_review_m3_m4_bridge.png)
- [S81 — S81_supp_m6_01_m6_neuron_atlas](../supplement/S81_supp_m6_01_m6_neuron_atlas.png)
- [S82 — S82_supp_m6_02_m6_model_layer_comparison](../supplement/S82_supp_m6_02_m6_model_layer_comparison.png)
- [S83 — S83_supp_m6_03_m6_attention_cases](../supplement/S83_supp_m6_03_m6_attention_cases.png)
- [S84 — S84_supp_m6_04_m6_residual_cases](../supplement/S84_supp_m6_04_m6_residual_cases.png)
- [S85 — S85_supp_m6_05_m6_m2_semantic_boundary](../supplement/S85_supp_m6_05_m6_m2_semantic_boundary.png)
- [S86 — S86_figure_34_registered_target_learning_images_signed](../supplement/S86_figure_34_registered_target_learning_images_signed.png)
- [S87 — S87_figure_35_block5_structural_learning_images_signed](../supplement/S87_figure_35_block5_structural_learning_images_signed.png)
- [S88 — S88_figure_36_spatial_ranking_learning_images_signed](../supplement/S88_figure_36_spatial_ranking_learning_images_signed.png)
- [S89 — S89_figure_37_channel0_depth_progression_signed](../supplement/S89_figure_37_channel0_depth_progression_signed.png)
- [S90 — S90_figure_38_attention_head_depth_atlas_signed](../supplement/S90_figure_38_attention_head_depth_atlas_signed.png)
- [S91 — S91_figure_39_vit_b16_attention_head_depth_atlas_signed](../supplement/S91_figure_39_vit_b16_attention_head_depth_atlas_signed.png)
- [S92 — S92_figure_40_vit_b16_module_chain_signed](../supplement/S92_figure_40_vit_b16_module_chain_signed.png)

## S8. 当前定性比较

六项 CPU image-model 计算使用相同来源图像、checkpoint 与预处理。Grad-CAM 使用最后一个 norm1 的 patch activation；attention rollout 与 raw attention 不依赖类别。展示样本不进入总体推断。具体规则见[显示协议](../evidence/display_protocol.md)与[计算收据](../evidence/compute_receipt.json)。

修改意见 v2 将下列展示性或重复性图件从主文移至补充材料；文件名保留原 publication-R2 编号以维持哈希和来源绑定：

- [坐标初始化](../figures/png/fig_03_coordinate_initialization.png)
- [局部导数传播](../figures/png/fig_04_local_derivative_generation.png)
- [Patch embedding](../figures/png/fig_05_patch_embedding.png)
- [DeiT-S/16 十二层图](../figures/png/fig_09_deit_s16_layerwise.png)
- [ViT-B/16 十二层图](../figures/png/fig_10_vit_b16_layerwise.png)
- [模块级学习图像与前向 activation](../figures/png/fig_11_module_learning_images.png)
- [全部 attention heads](../figures/png/fig_13_all_heads.png)
- [跨模型展示样本](../figures/png/fig_14_cross_model_examples.png)
- [DeiT-S/16 方法展示](../figures/png/fig_19_deit_s16_method_comparison.png)
- [ViT-B/16 方法展示](../figures/png/fig_20_vit_b16_method_comparison.png)
- [Positive、signed 与 absolute 读图对照](../figures/png/fig_22_signed_and_blue.png)

**Table S6. 跨 reference 排名一致性。**

| Model | Quantity | Spearman | Kendall tau-b | Top-1 agreement | Top-3 overlap | Mean abs rank displacement |
| --- | --- | --- | --- | --- | --- | --- |
| DeiT-S/16 | finite_effect | 0.3071 | 0.2444 | 0.4167 | 0.6111 | 1.5278 |
| DeiT-S/16 | taylor | 0.3000 | 0.2389 | 0.3333 | 0.6111 | 1.5278 |
| ViT-B/16 | finite_effect | 0.3759 | 0.2803 | 0.3333 | 0.5000 | 2.9444 |
| ViT-B/16 | taylor | 0.3782 | 0.2828 | 0.3333 | 0.5000 | 2.9444 |


[新增补充实验 S9–S14](supplement_experiments_v1.md)


<a id="v6-data"></a>
## S15. 数据、模型、总体与分析身份

ImageNet 来源为 ILSVRC2012 官方验证集。模型由 timm 0.9.16 标准 Block/Attention 实例化并严格加载权重，所有权重文件已重新核验大小和 SHA-256。AugReg 名称标明 IN21K 预训练后 IN1K 微调或 IN1K 训练；DeiT-S 使用公开 Facebook IN1K 参数。所有结果条件于这些参数状态。

| Model ID | Exact timm name / training | Bytes | SHA-256 |
| --- | --- | --- | --- |
| vit_b16 | vit_base_patch16_224.augreg2_in21k_ft_in1k | 346329934 | 489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d |
| deit_s16 | deit_small_patch16_224.fb_in1k | 88251143 | cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f |
| vit_b16_in1k | vit_base_patch16_224.augreg_in1k | 346335542 | 69c9a2822a47838f57e522350b10ccc0d89d67fdcee07603e4ae3259b3350336 |
| vit_s16_augreg | vit_small_patch16_224.augreg_in21k_ft_in1k | 88267574 | 545815b4e770d2fa6ca4b3ccba7c16b035e474354e52d17ca197ea4efecbf4d3 |

原 discovery、calibration、evaluation 分别 64、64、400 类，每类一图。Unit-study 的 100 类各一图是 evaluation 子集。E1 为 100 类各五图，E5 为 400 类各一图。E5 从排除历史曝光和 E1 的候选集中按 class_e5/image_e5 哈希排序选择。以下矩阵报告图像身份重叠；同类别可以跨后续队列出现，图像独立不等于类别独立。E5 的 train/validation/test 标签用于另一个预测分析，不改变此处全部 400 图 H1–H4 检查。

| Cohort | discovery | calibration | evaluation | unit_study | e1 | e5 |
| --- | --- | --- | --- | --- | --- | --- |
| discovery | 64 | 0 | 0 | 0 | 0 | 0 |
| calibration | 0 | 64 | 0 | 0 | 0 | 0 |
| evaluation | 0 | 0 | 400 | 100 | 0 | 0 |
| unit_study | 0 | 0 | 100 | 100 | 0 | 0 |
| e1 | 0 | 0 | 0 | 0 | 500 | 0 |
| e5 | 0 | 0 | 0 | 0 | 0 | 400 |

预处理使用 Pillow 10.2.0 RGB，不应用 EXIF transpose。Torchvision 0.17.2 先对 PIL 图像以 bicubic 将短边 resize 到 248，antialias=True，再中心 crop 224，转 float32 并除 255。每模型 normalization 使用 Table 2 参数。Blur 在 crop 后 RGB 上使用 31×31 Gaussian kernel、sigma=10、reflect padding=15，再 normalization。Evaluation mode 关闭 dropout 和 drop path。CLS、位置编码与 bias 固定。原 logit 目标为官方 ground-truth，clean-top1 为另一个预先固定读出；它们不参与 cohort 选择。原 registry 每层四 tokens/四 channels 的规则和实际列表由封存 registry 给出。M3 固定 CLS 加 15 个 3×5 分层 tokens，每层 32 channels；跨模型同序号不代表同一语义神经元。结构研究扫描全部 heads 和 12 blocks，区域主任务为 block 5，ViT-B heads 0/6/11、DeiT-S heads 0/3/5。

所有抽样是确定性规则或公开种子。实际 selected IDs 与 targets 由 [cohort bindings](../evidence/revision_v6/cohort_bindings.json) 提供。随机控制的具体派生键与调用顺序保留于原协议，不能把不同分析的种子混用。

| Purpose | Frozen seed / deterministic rule |
| --- | --- |
| Original image selection | SHA256(NNS_VIT_ENDPOINT_EP0_R1_20260918 &#124; class &#124; WNID), then namespace &#124; image &#124; filename |
| Unit cohort | SHA256(ENDPOINT_M3_COHORT_20260919 &#124; sample_id &#124; image_sha256), lowest 100 |
| Unit channels | SHA256(ENDPOINT_M3_CHANNEL_20260919 &#124; model &#124; block &#124; channel), lowest 32 |
| Unit tokens | CLS + nearest patch to each of 3×5 strata centers; row-major ties |
| Original bootstrap | SHA256(namespace &#124; bootstrap &#124; family_id), first 8 bytes big-endian; PCG64 |
| Unit bootstrap | SHA256(ENDPOINT_M3_BOOTSTRAP_20260919 &#124; M3 &#124; model &#124; metric), first 8 bytes |
| Structural bootstrap | SHA256(M4 &#124; test &#124; model &#124; test_id), with residual type inserted where applicable |
| Random initialization | 202609201, 202609202, 202609203 |
| Independent cohort / target selection | NNS_VIT_REVISION_V5_20260921_R1 namespace; class_e5, image_e5 and post/block/channel/token keys |
| Independent-cohort bootstrap | ENDPOINT_M1_20260918_V1 &#124; 20260918 &#124; bootstrap &#124; V5 &#124; (model,hypothesis) |
| Coordinate/threshold bootstrap | SHA256(2026092302 &#124; analysis key), first 8 bytes; 10,000 PCG64 replicates |
| Analytic directions | NumPy default_rng(2026092301), three normalized Gaussian directions |
| Regional random orders | Frozen namespace/model/image/reference/head/repeat keys in original intervention registry; 20 repeats |

确认性身份相对于各自分析前的本地冻结计划。外部公开注册未提供。下表保留各 family 的原成员与用途；后续检查不与原 28 项追溯合并。完整 28 个 test IDs 与所有 SHA-256 位于 [analysis identity](../evidence/revision_v6/analysis_identity.json)。0.10/0.05 没有外部效用校准来源，属于研究者操作阈值。原判断要求有效性、完整覆盖、n 和区间精度门，以及点估计和 Holm 检验共同通过；不要求 CI lower > 0.10。阈值敏感性见 [68 行完整结果](../evidence/revision_v6/threshold_sensitivity.csv)。

| Analysis | Identity | Multiplicity family | Local freeze evidence |
| --- | --- | --- | --- |
| original_component_and_region | confirmatory | 28 tests jointly Holm; 12 component correlations + 16 regional baseline contrasts | local pre-outcome plan; first committed occurrence below; 6237ef87a1af0205973b76bee172defe33a0a6da 2026-09-18T09:26:20+08:00 |
| unit_depth | confirmatory | five per model: entropy, locality, distance, cancellation, CLS-patch entropy | local pre-outcome plan; first committed occurrence below; f2eba3f60d765e6a9729d9ff12a57bd01f93f952 2026-09-19T13:53:48+08:00 |
| attention_residual_depth | confirmatory | four attention tests/model; two residual tests/model/type | local pre-outcome plan; first committed occurrence below; 1a1000bcdd15d8b0e76406f4376a82d60ec79b86 2026-09-19T17:25:17+08:00 |
| head_baselines | exploratory | separate paired and absolute Spearman families; other metrics descriptive | local pre-outcome plan; first committed occurrence below; 33c8c320446a631b2e451016b3268eefdd9c926b 2026-09-19T04:20:36+08:00 |
| residual_spatial_tv | post-hoc | four Holm contrasts across model and residual type | 2026-09-19T10:59:17.569991+00:00; protocol timestamp and hashes |
| QKV_spatial | prospective robustness | 30 descriptive contrasts; no confirmatory family combined with prior tests | 2026-09-20T17:14:57.250515+00:00; protocol timestamp and hashes |
| independent_cohort_checkpoints | prospective robustness | H1-H4 Holm separately within each checkpoint | local pre-outcome plan; first committed occurrence below; 944f54c99c8d83803d461224565b93a779e75915 2026-09-21T11:31:10+08:00 |
| training_target_sampling_controls | exploratory | descriptive 2000 bootstrap; training control resamples image and initialization seed; sampling control resamples target strata and images | 2026-09-20T00:22:59.620440+00:00; 395d30b3804983d0e51e99a28d0c3c900aca8fda 2026-09-20T22:41:48+08:00 |
| coordinate_sensitivity_v6 | prospective robustness | no significance-based success criterion; report all directions and intervals | 2026-09-22T16:43:58.690139+00:00; protocol timestamp and hashes |
| threshold_sensitivity_v6 | post-hoc | descriptive thresholds only; original tests and decisions immutable | 2026-09-22T16:43:58.690139+00:00; protocol timestamp and hashes |

<a id="v6-intervention"></a>
## S16. 有限替换变量与可执行伪代码

A 和 V 是前向变量，routing-induced/value-induced 是微分分组。Table 6 的 A/V family 每个 block 各一个全-head A 替换和 V 替换，共 24 成分；head family 为所有 block/head，residual family 每 block attention 与 MLP branch 各一项。Reference activation 来自完整 reference forward。下面的 consumer slot 是替换身份的一部分。

| Replacement | Forward tensor / shape (batch=1) | Consumer input slot | Clean state and recomputation |
| --- | --- | --- | --- |
| Attention probability | A [1,H,197,197], after softmax/dropout | left input of A @ V | V, other branches and residual identity clean; recompute AV, concat, output projection, residual suffix |
| V projection | V [1,H,197,64], qkv projection slice | right input of A @ V | Q/K/A and identity clean; recompute AV and entire suffix |
| Head output | slice of projection_input [1,197,H×64], head h channels h×64:(h+1)×64 | attn.proj input channels | all other head slices and residual identity clean; recompute projection and suffix |
| Attention branch | drop_path1 output [1,197,D] | residual1 add branch input | identity u clean; recompute residual sum, MLP and remaining blocks |
| MLP branch | drop_path2 output [1,197,D] | residual2 add branch input | carry r1 clean; recompute residual sum and remaining blocks |

    clean_rgb = RGB(image) / 255
    clean_rgb = center_crop(resize_shorter_side(clean_rgb, 248, bicubic, antialias=True), 224)
    x = (clean_rgb - mu) / sigma
    reference_zero = zeros_like(x)             # raw RGB equals mu
    reference_blur = (GaussianBlur(clean_rgb, kernel=31, sigma=10, reflect_padding=15) - mu) / sigma
    clean = forward_and_capture(x)
    reference = forward_and_capture(reference_input)
    for fixed block, component, consumer slot:
        replacement = reference.tensor_at(component, slot)
        patched = forward(x, replace_only_this_use=replacement)
        effect = clean.GT_logit - patched.GT_logit
        # All sibling heads and bypass inputs remain clean before the replacement.
        # Every node downstream of the replaced use is recomputed normally.

    for singleton region P in the fixed 4x4 grid of 56x56 pixels:
        x_P = x.copy(); x_P[P] = reference_input[P]
        mixed[P] = forward_and_capture(x_P)
        regional_Taylor[P] = abs(dot(clean_head_cotangent, clean_head - mixed[P].head))
    rank_regions(score, ties=ascending_region_id)
    for budget in [1, 2, 4]:
        selected = top_regions(budget)
        x_selected = x with selected pixels from reference_input
        mixed = forward_and_capture(x_selected)
        full_input_effect = clean.GT_logit - mixed.GT_logit
        isolated = forward(x, target_head_slice=mixed.head_slice)
        isolated_effect = clean.GT_logit - isolated.GT_logit
    AUC = trapezoid(abs(isolated_effect), areas=[0,1/16,2/16,4/16]) / 0.25
    image_AUC = mean(AUC across the three fixed heads)
    random_AUC = mean(AUC per random repeat, then heads) # abs before averaging repeats

Full-input effect 改变整条前向计算；isolated-component effect 只将 mixed forward 的目标张量放入 clean forward，其他 head 与 bypass 保持 clean。区域分数的候选集合相同，但 Taylor 需额外 singleton mixed forwards，因此信息预算不同。区域 ties 用编号升序，head ties 用 component ID 升序，Spearman 用平均秩。所谓 candidate-envelope AUC 是在每个面积点从已评价候选中取最大 absolute effect，再积分、平均 heads，可能拼接不同排序的点，因此不是一条可执行的全局最优区域策略。它只用作已评价集合的上界和 regret 分母。

<a id="v6-coordinate"></a>
## S17. 坐标敏感性与独立解析检查

三种表示均在同一当前输入点、同一固定目标与 checkpoint 上计算。Raw gradient 除去模型 sigma；common-origin weighted gradient 使用原始 RGB 零点，不随模型 mean 改变。以下每格为图像级中位差和 95% bootstrap CI。Unit 每模型 n=100，结构项 n=400；缺失图像为零。每个 coordinate 在图内配对，不作为额外样本。

$$
E_i^{(x)}=x_i\frac{\partial s}{\partial x_i},\qquad
G_i^{(r)}=\frac1{\sigma_i}\frac{\partial s}{\partial x_i},\qquad
E_i^{(0)}=r_i\frac{\partial s}{\partial r_i}
=\left(x_i+\frac{\mu_i}{\sigma_i}\right)\frac{\partial s}{\partial x_i}.
$$

| Model | Object / metric | Contrast | Normalized Endpoint | Raw-pixel gradient | Common-origin gradient | n / NA |
| --- | --- | --- | --- | --- | --- | --- |
| deit_s16 | K / entropy | block11_minus_block0 | -0.067 [-0.074, -0.059] | -0.066 [-0.073, -0.060] | -0.061 [-0.069, -0.053] | 400 / 0 |
| deit_s16 | K / entropy | late_minus_early | -0.028 [-0.031, -0.024] | -0.029 [-0.032, -0.026] | -0.027 [-0.031, -0.024] | 400 / 0 |
| deit_s16 | Q / entropy | block11_minus_block0 | -0.004 [-0.008, 0.0005] | 0.003 [-0.001, 0.006] | 0.002 [-0.003, 0.006] | 400 / 0 |
| deit_s16 | Q / entropy | late_minus_early | 0.018 [0.013, 0.022] | 0.018 [0.015, 0.022] | 0.015 [0.012, 0.018] | 400 / 0 |
| deit_s16 | V / entropy | block11_minus_block0 | -0.067 [-0.075, -0.061] | -0.076 [-0.081, -0.068] | -0.070 [-0.077, -0.062] | 400 / 0 |
| deit_s16 | V / entropy | late_minus_early | -0.013 [-0.016, -0.011] | -0.016 [-0.017, -0.013] | -0.015 [-0.017, -0.013] | 400 / 0 |
| deit_s16 | attention / output_identity_tv | block11_minus_block0 | 0.078 [0.072, 0.085] | 0.079 [0.070, 0.085] | 0.074 [0.067, 0.081] | 400 / 0 |
| deit_s16 | attention / output_identity_tv | late_minus_early | 0.028 [0.026, 0.030] | 0.028 [0.026, 0.030] | 0.027 [0.026, 0.029] | 400 / 0 |
| deit_s16 | mlp / output_identity_tv | block11_minus_block0 | -0.050 [-0.059, -0.040] | -0.044 [-0.057, -0.037] | -0.037 [-0.043, -0.029] | 400 / 0 |
| deit_s16 | mlp / output_identity_tv | late_minus_early | -0.046 [-0.052, -0.041] | -0.044 [-0.051, -0.039] | -0.039 [-0.044, -0.034] | 400 / 0 |
| deit_s16 | unit / distance | block11_minus_block0 | 0.163 [0.159, 0.169] | 0.169 [0.164, 0.175] | 0.171 [0.167, 0.178] | 100 / 0 |
| deit_s16 | unit / distance | late_minus_early | 0.122 [0.120, 0.128] | 0.129 [0.126, 0.132] | 0.127 [0.122, 0.131] | 100 / 0 |
| deit_s16 | unit / entropy | block11_minus_block0 | 0.397 [0.387, 0.404] | 0.426 [0.412, 0.438] | 0.418 [0.400, 0.427] | 100 / 0 |
| deit_s16 | unit / entropy | late_minus_early | 0.225 [0.212, 0.228] | 0.245 [0.241, 0.249] | 0.221 [0.217, 0.229] | 100 / 0 |
| deit_s16 | unit / locality | block11_minus_block0 | -0.455 [-0.467, -0.432] | -0.453 [-0.469, -0.441] | -0.477 [-0.493, -0.455] | 100 / 0 |
| deit_s16 | unit / locality | late_minus_early | -0.342 [-0.353, -0.333] | -0.359 [-0.366, -0.352] | -0.356 [-0.361, -0.344] | 100 / 0 |
| vit_b16 | K / entropy | block11_minus_block0 | -0.059 [-0.064, -0.053] | -0.052 [-0.058, -0.048] | -0.046 [-0.054, -0.041] | 400 / 0 |
| vit_b16 | K / entropy | late_minus_early | -0.021 [-0.024, -0.019] | -0.019 [-0.023, -0.016] | -0.019 [-0.022, -0.016] | 400 / 0 |
| vit_b16 | Q / entropy | block11_minus_block0 | 0.027 [0.022, 0.031] | 0.026 [0.018, 0.031] | 0.014 [0.009, 0.018] | 400 / 0 |
| vit_b16 | Q / entropy | late_minus_early | 0.037 [0.033, 0.043] | 0.039 [0.033, 0.044] | 0.034 [0.030, 0.037] | 400 / 0 |
| vit_b16 | V / entropy | block11_minus_block0 | -0.053 [-0.059, -0.047] | -0.057 [-0.061, -0.049] | -0.055 [-0.060, -0.050] | 400 / 0 |
| vit_b16 | V / entropy | late_minus_early | -0.010 [-0.012, -0.008] | -0.009 [-0.011, -0.007] | -0.010 [-0.012, -0.007] | 400 / 0 |
| vit_b16 | attention / output_identity_tv | block11_minus_block0 | 0.166 [0.159, 0.172] | 0.167 [0.159, 0.174] | 0.167 [0.160, 0.175] | 400 / 0 |
| vit_b16 | attention / output_identity_tv | late_minus_early | 0.052 [0.049, 0.055] | 0.054 [0.051, 0.056] | 0.053 [0.050, 0.055] | 400 / 0 |
| vit_b16 | mlp / output_identity_tv | block11_minus_block0 | -0.037 [-0.043, -0.031] | -0.034 [-0.038, -0.029] | -0.032 [-0.039, -0.025] | 400 / 0 |
| vit_b16 | mlp / output_identity_tv | late_minus_early | -0.019 [-0.022, -0.015] | -0.017 [-0.020, -0.013] | -0.018 [-0.021, -0.014] | 400 / 0 |
| vit_b16 | unit / distance | block11_minus_block0 | 0.197 [0.190, 0.204] | 0.205 [0.200, 0.211] | 0.209 [0.203, 0.214] | 100 / 0 |
| vit_b16 | unit / distance | late_minus_early | 0.131 [0.129, 0.136] | 0.138 [0.134, 0.141] | 0.134 [0.129, 0.138] | 100 / 0 |
| vit_b16 | unit / entropy | block11_minus_block0 | 0.398 [0.392, 0.409] | 0.440 [0.426, 0.447] | 0.412 [0.402, 0.426] | 100 / 0 |
| vit_b16 | unit / entropy | late_minus_early | 0.220 [0.211, 0.227] | 0.240 [0.235, 0.246] | 0.217 [0.209, 0.223] | 100 / 0 |
| vit_b16 | unit / locality | block11_minus_block0 | -0.526 [-0.534, -0.516] | -0.543 [-0.554, -0.532] | -0.545 [-0.552, -0.538] | 100 / 0 |
| vit_b16 | unit / locality | late_minus_early | -0.356 [-0.363, -0.350] | -0.367 [-0.373, -0.361] | -0.356 [-0.364, -0.347] | 100 / 0 |

DeiT-S Q 的 B11−B0 在三坐标下的点估计符号不同且三个区间跨零。Phase contrast 与 B11−B0 的差异还说明，终点比较不能替代整个轨迹。主文始终限定 normalized-input Endpoint。其他对象只作这三表示下的方向保持判断，不推出一般参数化不变性。

独立参考以 NumPy float64 显式构造 affine、LayerNorm、softmax $\operatorname{diag}(a)-aa^{\mathsf T}$、Q/K/V、head concat/output projection、GELU/MLP 和 residual identity Jacobian。参考模块不导入 scanner、hook 或 autograd。独立驱动将其结果与实际主扫描器在单头、双头、pre-LN residual attention/MLP tiny 模型上逐项比较。32 个分量比较和 9 个冻结方向有限差分通过，最大分量绝对误差 $1.11\times10^{-16}$；每项 relative L2 和容差见 [analytic results](../evidence/revision_v6/analytic_reference_results.json)。共享 producer 双 slot toy、Q/K slot 交换、提前合并 fan-out 和重复合并项均被拒绝，正确实现先通过。此能力限于这些数值图和故障类型，不等于排除全部 scanner 实现错误。


原区域数组键 raw_gradient 记录 $\nabla_xs$，不是 $\nabla_rs$。主文已按实际坐标重命名。Random region order 以 NNS_VIT_ENDPOINT_E5_RANDOM_REGIONS_V1、task_seed、model_id、reference_id、head、repeat 拼接字符串作 SHA-256，前八字节 big-endian 作为 PCG64 seed，再 permutation(16)。task_seed 来自 NNS_VIT_ENDPOINT_QUEUE_V1 与 task_id。

## S18. Revision v7 分类型、层级与质量诊断

本节结果来自现役 nns_vit_endpoint_revision_v7_r4_20260923 CPU-only 根。R1–R3 的统计或 provenance 问题按递增根保留；R4 恢复 R2 冻结的 bootstrap seed，限定 concordance 为描述性比例，在 residual-TV 质量分层中先按图聚合，并在执行前绑定最终 analysis/validator/test 哈希。旧 pooled family、28 项 Holm 和全部 sealed 结论均未改变。

**Table S18.1. 分类型相关；图像级中位数 [95% CI]。** 每项在一张图的 12 个 blocks 间计算；这是事后诊断。

| Model | Reference | A only | V only | Attention branch only | MLP branch only |
| --- | --- | --- | --- | --- | --- |
| vit_b16 | normalized-zero | 0.049 [0.007, 0.091] | 0.224 [0.192, 0.259] | -0.028 [-0.056, -0.003] | 0.287 [0.266, 0.318] |
| vit_b16 | blur | 0.073 [0.028, 0.112] | 0.311 [0.266, 0.350] | 0.000 [-0.021, 0.042] | 0.378 [0.336, 0.402] |
| deit_s16 | normalized-zero | 0.091 [0.049, 0.119] | 0.035 [-0.021, 0.087] | -0.157 [-0.189, -0.119] | 0.378 [0.346, 0.406] |
| deit_s16 | blur | 0.101 [0.056, 0.133] | 0.147 [0.112, 0.182] | -0.024 [-0.070, 0.014] | 0.371 [0.332, 0.395] |

**Table S18.2. 类型/层级残差化 rank correlation 与 head 内相关。** O、B 和 O+B 分别表示控制 operation、block、二者；逐图先对 $m$ 与 $|e|$ 取平均秩，再以带截距的 one-hot 固定效应最小二乘残差化。Head 列先在每层 heads 内计算 Spearman，截断至 $[-1+10^{-12},1-10^{-12}]$ 后作 Fisher-z 平均并反变换。

| Model | Reference | A/V O | A/V B | A/V O+B | Residual O | Residual B | Residual O+B | Head within-block | A/V concordance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b16 | normalized-zero | 0.122 | -0.044 | 0.030 | 0.168 | -0.010 | -0.128 | 0.321 | 0.500 |
| vit_b16 | blur | 0.164 | -0.015 | -0.018 | 0.184 | 0.134 | 0.042 | 0.303 | 0.500 |
| deit_s16 | normalized-zero | 0.077 | 0.167 | 0.089 | 0.110 | 0.019 | -0.091 | 0.245 | 0.583 |
| deit_s16 | blur | 0.139 | 0.186 | 0.095 | 0.161 | 0.280 | 0.142 | 0.183 | 0.583 |

A/V concordance 是 12 层中 $\operatorname{sign}[(m_A-m_V)(|e_A|-|e_V|)]$ 一致的比例；零乘积作为 tie 排除。它不是 Spearman，仅作描述性报告，不做以 0 为原假设的符号检验。完整置信区间、分位数、单层结果、自由度、condition number 与独立验证位于 v7 r4 根。所有必需相关都有 400/400 个有效图像。

结果说明旧 pooled family 的正相关不能解释为类型内部统一排序能力。V-only 在 ViT-B 较强，DeiT-S normalized-zero 的区间跨零；MLP branch 在四种设置中为正，attention branch 为负或接近零。类型和 block 同时控制后，A/V 在 ViT-B 两种 reference 下接近零，residual 在 normalized-zero 下为负；DeiT-S 的调整结果随 reference 改变。

**Table S18.3. Own-token locality 半径敏感性；late-minus-early 中位差 [95% CI]。**

| Model | Radius 0 | Radius 1 | Radius 2 |
| --- | --- | --- | --- |
| vit_b16 | -0.302 [-0.310, -0.298] | -0.356 [-0.363, -0.350] | -0.335 [-0.344, -0.322] |
| deit_s16 | -0.260 [-0.263, -0.253] | -0.342 [-0.353, -0.333] | -0.315 [-0.323, -0.303] |

三个半径的方向一致。半径 1 仍是预先定义的主描述量；该敏感性不用于选择新主指标。

Residual output–identity TV 的 absolute-mass 四分位诊断以图像为统计单位。Attention 的 Q1→Q4 中位 TV 为 ViT-B 0.107→0.073、DeiT-S 0.071→0.055；MLP 为 ViT-B 0.139→0.105、DeiT-S 0.117→0.090。空间差异不只存在于最低质量组，但这些未调整的事后分层不能排除梯度质量或噪声的影响。

H4 从 1,600 个原始 JSON/NPZ 独立复算；四个 checkpoint 的中位数依次为 vit_b16 -0.114、deit_s16 -0.186、vit_b16_in1k -0.186、vit_s16_augreg -0.136，正/负/平分别为 101/297/2、96/298/6、56/342/2、107/285/8。四项 checkpoint 内 Holm 调整值均小于 $10^{-6}$。原始逐图、block-level 相关、完整 p 值及机器收据见 v7 r4 根。

## S19. 架构切面、空间熵质量与 H4 聚合敏感性

### S19.1 三类 registry 与实际切面认证

机器合同现按对象分为标量目标、局部微分分解和有限替换干预三个只读派生 registry。标量目标数由原 EP0 逐行重算，ViT-B/16 与 DeiT-S/16 分别为 457/385。每个模型均含 192 个 post-block token-channel、48 个 patch sum、48 个 patch energy、18 个 module patch aggregate、4 个 patch-embedding aggregate、1 个 final-CLS coordinate 和 2 个类别 logits；head coordinate 分别为 144 与 72。Q/K/V-induced、routing/value、residual 分项和可替换变量不再混入这一计数。

| Registry | 包含对象 | 明确不包含 |
| --- | --- | --- |
| scalar target | 标量 readout、图中位置、聚合规则 | 路径分项、干预变量 |
| local-differential decomposition | native use、解析分组、cut、重构对象 | 标量目标数量、有限替换效果 |
| finite-replacement intervention | 可替换 activation、reference、effect 与区域任务 | 局部微分闭合对象 |

冻结的 pre-LN ViT tensor-use DAG 对 842 个 target、12 层与 8 类 cut 的 80,832 个配对逐一检查。算法在目标祖先子图上动态传播每条输入路径经过的 cut-edge 数，并要求完整 cut 的计数集合严格为 $\{1\}$，同时要求 cut 中每条边均属于该祖先子图。26,094 个配对通过完整 target-cut 条件，15,660 个通过 attention branch-only partition 条件，24,444 个因声明重构对象不适用而拒绝，14,634 个因目标位于上游或 branch 不在目标祖先图而拒绝。该证明覆盖 registry 绑定的标准 unfused pre-LN ViT；数值闭合与 mutation 测试继续作为实现层面的独立证据。

### S19.2 空间熵的总质量敏感性

阈值在每个 model--block 内由全部冻结 M3 unit map 的总 L1 mass 分布确定。原始 late-minus-early entropy 从源数组复现的最大误差为 0。下表以图像为单位给出中位数和 95% bootstrap 区间；截尾项保留全部 100 图，四分位项对合法空组传播 NA。

| Model | Analysis | Median [95% CI] | Valid / total |
| --- | --- | --- | --- |
| ViT-B/16 | no trimming | 0.220 [0.211, 0.227] | 100 / 100 |
| ViT-B/16 | trim lowest 1% mass | 0.222 [0.213, 0.230] | 100 / 100 |
| ViT-B/16 | trim lowest 5% mass | 0.229 [0.221, 0.237] | 100 / 100 |
| ViT-B/16 | trim lowest 10% mass | 0.234 [0.227, 0.242] | 100 / 100 |
| DeiT-S/16 | no trimming | 0.225 [0.212, 0.228] | 100 / 100 |
| DeiT-S/16 | trim lowest 1% mass | 0.226 [0.215, 0.229] | 100 / 100 |
| DeiT-S/16 | trim lowest 5% mass | 0.230 [0.219, 0.235] | 100 / 100 |
| DeiT-S/16 | trim lowest 10% mass | 0.233 [0.223, 0.237] | 100 / 100 |

按 model--block mass 四分位分层后，ViT-B 的 Q1--Q4 中位差为 0.090、0.194、0.238、0.266，DeiT-S 为 0.112、0.205、0.222、0.224。八个 image--quartile 组合因 early 或 late 组为空而记为 typed NA，没有删除或替换图像。同图同层 log1p(total mass) 与 entropy 的平均 Spearman 中位数为 ViT-B −0.729 [−0.747, −0.693]、DeiT-S −0.760 [−0.775, −0.723]；late-minus-early log1p mass 则分别为 2.893 [2.829, 2.961] 与 2.183 [2.109, 2.227]。因此熵与质量明显相关，但总体扩展方向不由最低质量图单独产生。

### S19.3 H4 匹配聚合与随机区域精度

匹配聚合将 path score 定义为 $M^{\mathrm{sum}}_{jbh}=|\sum_i\mathcal L_{jbh,i}|$，使其与 activation × gradient 对照都采用“先坐标求和、再取绝对值”。其余 target、normalized-zero reference、四个 blocks、有限替换 effect 与图像级聚合不变。

| Checkpoint | Matched path minus activation × gradient | Positive / negative / tie |
| --- | --- | --- |
| vit_b16 | −0.124 [−0.147, −0.102] | 89 / 311 / 0 |
| deit_s16 | −0.114 [−0.157, −0.079] | 130 / 261 / 9 |
| vit_b16_in1k | −0.226 [−0.253, −0.205] | 49 / 349 / 2 |
| vit_s16_augreg | −0.171 [−0.200, −0.143] | 110 / 285 / 5 |

原 H4 从相同数组再次复现，逐图最大误差为 0。匹配聚合下四个 checkpoint 仍为负，因而原结果不由两种分数的 absolute/sum 顺序差异解释。

随机区域基线无需新增模型运行。源复核覆盖 1,600 个 model--reference--image 条件和 32,000 条 repeat 记录；每个条件均有 20 个完整重复。相对 Monte Carlo 标准误中位数为 5.1%--6.9%，q90 为 7.6%--8.9%；5/10/20-repeat 重采样、split-half 与 leave-one-repeat-out 均已保留。该精度支持 cohort-level 比较，单图细小优势不据此判定。
