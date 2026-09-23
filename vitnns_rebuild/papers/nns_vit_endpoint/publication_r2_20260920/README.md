# NNS–ViT Endpoint 论文、实验与图件复现指南

本目录是论文《面向视觉 Transformer 内部目标的双索引局部 Jacobian 扫描与结构审计》的 revision-v8 匿名送审包。本文档说明四件事：论文在讲什么，核心方法是什么，实验代码和证据在哪里，以及怎样从一个最小 CPU demo 开始生成 learning-image 图。Revision v6/v7 证据仍作为不可变的历史输入保留。

当前正文入口：

- [中文正文](zh/manuscript.md)
- [English manuscript](en/manuscript.md)
- [中文补充材料](zh/supplement.md)
- [English supplement](en/supplement.md)
- [中文补充控制实验](zh/supplement_experiments_v1.md)
- [English supplementary controls](en/supplement_experiments_v1.md)
- [22 张图的网页总览](gallery.html)
- [论文图 manifest](figure_manifest.json)
- [revision-v6 证据包](evidence/revision_v6/package_manifest.json)
- [revision-v7 逐条意见处理](qc/revision_v7/reviewer_comment_disposition.md)
- [revision-v7 六维终审](qc/revision_v7/adversarial_final_review.md)
- [revision-v8 结构与敏感性补强证据](../../../reports/nns_vit_endpoint_revision_v8_r2_20260923/terminal_receipt.json)
- [revision-v8 写作与发布审计](qc/revision_v8/)
- [中文 LaTeX/PDF](zh/manuscript.tex) / [English LaTeX/PDF](en/manuscript.tex)
- [共享参考文献数据库](references.bib)

## 1. 一句话概括

论文提出面向 ViT 内部目标的 Endpoint 扫描框架：把当前输入点的 Gradient × Input 按“输入坐标”和“具体 tensor use”展开为结构化局部图，用合法切面验证分项重构，并在统一输入坐标中比较 attention、MLP 和 residual 的空间组织。

论文以已有的 Gradient × Input 作为局部测量原语：

$$
E_s(x)=x\odot\nabla_xs(x).
$$

方法增量在于把同一个局部微分组织成可定位、可分组、可闭合检查和可跨模块比较的 ViT 内部结构表示。

## 2. 论文主线

### 2.1 数学身份：先说明 NNS learning image 是什么

在所有局部 Jacobian 都取自同一个当前前向状态，而且传播过程中不额外归一化、截断或重分配时，标量 NNS learning image 等于同目标的 Gradient × Input。

在此基础上，论文把 $x\odot\nabla_xs(x)$ 扩展到 ViT 的内部标量目标和结构分项，使不同模块能够在统一输入坐标中分析。

### 2.2 双索引表示：同时保留“在哪里”和“经过哪个 use”

对注册标量目标 $s$、实际 consumer-use edge $e$ 和输入坐标 $i$，论文定义

$$
\mathcal L_{e,i}(s;x)
=x_i\left[J_{z_e,x}(x)^{\mathsf T}\bar z_e^{(s)}\right]_i.
$$

两个索引分别回答：

- $i$：局部输入依赖出现在哪个像素和颜色通道。
- $e$：该项经过哪个 producer、consumer、input slot 和 invocation。

只保留输入索引会得到 Total Endpoint，但会丢失 Q/K/V、head 和 identity/branch 身份。只保留 edge 标量会丢失空间位置，因而不能比较分项熵、空间 TV 和空间抵消。

### 2.3 分支分组与全局切面：闭合对象必须分清

Attention 内部有

$$
\mathrm dO_h=(\mathrm dA)V+A\mathrm dV.
$$

其中 Q/K 诱导项合为 routing-induced differential，V 诱导项为 value-induced differential。它们只重构 attention branch contribution。

Residual 输出满足

$$
y=u+f(u).
$$

因此完整目标的全局切面必须包含 identity bypass：

$$
E_s=\mathcal L^{\mathrm{identity}}+\mathcal B^{\mathrm{branch}}.
$$

论文用 consumer、slot、invocation、fan-out 和 residual bypass 审计检查声明的切面合同。结构闭合验证指定计算图上的局部微分记账；有限替换实验另行刻画参考状态下的行为变化。

### 2.4 实证分析：从结构表示到有限变化

实验按三个证据层级展开。

1. **表示与正确性**：验证 NNS 与 Gradient × Input 的数值身份，认证实际 target--cut 配对，并验证 branch/global cut、Q/K/V、head 和 residual 的重构。
2. **结构发现**：在固定模型检查点和目标集合上，分析空间熵、locality、distance、routing/value 以及 identity/branch 随 block index 的轨迹。
3. **行为连接**：用指定参考状态的 activation replacement 区分当前状态的 Endpoint 局部依赖与沿实际位移方向的一阶变化。

### 2.5 论文最终得到什么

- 实际 ViT 图上的 80,832 个 target--cut 配对得到逐项认证，标量目标、局部微分分解和有限替换变量由三类 registry 分别约束。
- 固定抽样目标的 normalized-input Endpoint 在后层呈现更高空间熵、更低 own-token locality 和更大归一化距离；这一空间扩展方向在低质量分层、截尾分析和独立图像队列中保持。
- K/V、routing/value 和 attention/MLP residual 呈现可区分的空间与尺度轨迹，说明双索引图能够揭示总图或单一成分标量无法恢复的结构差异。
- 多数主要方向在三种输入坐标量和四个公开模型检查点上复现；Q 分项的模型与坐标敏感性进一步界定了对象适用范围。
- Endpoint 成分质量与同一内部对象的有限替换效应在 11/12 个原分析族中形成预设关联，且不同结构类型呈现不同关联轮廓。
- Endpoint 描述当前状态的结构化局部依赖；加入参考位移的一阶量进一步描述沿指定替换方向的变化。二者共同连接结构观察与行为分析。

这些结论对应论文声明的 ImageNet 样本、固定模型检查点、固定目标、计算图、坐标定义、参考状态和替换合同。

## 3. 从问题到代码

所有相对路径均从 `vitnns_rebuild/` 仓库根目录解释。

### 3.1 核心方法实现

| 代码 | 作用 |
| --- | --- |
| [`src/vitnns/endpoint_nns.py`](../../../src/vitnns/endpoint_nns.py) | 一般 NNS Endpoint 原语。计算当前点局部梯度，并形成 input-weighted local map。 |
| [`src/vitnns/endpoint_vit.py`](../../../src/vitnns/endpoint_vit.py) | ViT 主实现。注册内部目标，扫描 token/channel/head，捕获 unfused attention use，并实现 `decompose_endpoint_vit` 的 head、Q/K/V、routing/value 和 residual 分解。 |
| [`src/vitnns/modeling.py`](../../../src/vitnns/modeling.py) | 严格加载 checkpoint、绑定模型身份并生成 canonical preprocessing。 |
| [`scripts/run_endpoint_vit_scan.py`](../../../scripts/run_endpoint_vit_scan.py) | 单图 CPU demo。输出原始数组、扫描报告和 learning-image/edge-decomposition 图。 |

### 3.2 正式 Endpoint 主实验

| 论文问题 | 核心模块 | 执行与验证入口 | 主要证据根 |
| --- | --- | --- | --- |
| 注册目标 atlas | [`endpoint_e2_atlas.py`](../../../src/vitnns/endpoint_e2_atlas.py) | [`run_endpoint_e2_atlas.py`](../../../scripts/run_endpoint_e2_atlas.py)、[`validate_endpoint_e2_atlas.py`](../../../scripts/validate_endpoint_e2_atlas.py) | `reports/nns_vit_endpoint_e2_atlas_r4_20260918/` |
| 结构切面闭合 | [`endpoint_e3_closure.py`](../../../src/vitnns/endpoint_e3_closure.py) | [`run_endpoint_e3_closure.py`](../../../scripts/run_endpoint_e3_closure.py)、[`validate_endpoint_e3_closure.py`](../../../scripts/validate_endpoint_e3_closure.py) | `reports/nns_vit_endpoint_e3_closure_r2_20260918/`，原始 component maps 在 R1 根 |
| 成分有限替换 | [`endpoint_e4_intervention.py`](../../../src/vitnns/endpoint_e4_intervention.py) | [`run_endpoint_e4_intervention.py`](../../../scripts/run_endpoint_e4_intervention.py)、[`run_endpoint_e4_recovery_r3.py`](../../../scripts/run_endpoint_e4_recovery_r3.py)、[`validate_endpoint_e4_recovery_r3.py`](../../../scripts/validate_endpoint_e4_recovery_r3.py) | `reports/nns_vit_endpoint_e4_intervention_r3_20260918/` |
| 区域诱导的指定 head 替换 | [`endpoint_e5_spatial.py`](../../../src/vitnns/endpoint_e5_spatial.py) | [`run_endpoint_e5_spatial.py`](../../../scripts/run_endpoint_e5_spatial.py)、[`validate_endpoint_e5_spatial.py`](../../../scripts/validate_endpoint_e5_spatial.py) | `reports/nns_vit_endpoint_e5_spatial_r1_20260918/` |
| 图像级统计与 Holm | [`endpoint_e6_release.py`](../../../src/vitnns/endpoint_e6_release.py) | [`run_endpoint_e6_release.py`](../../../scripts/run_endpoint_e6_release.py)、[`validate_endpoint_e6_release.py`](../../../scripts/validate_endpoint_e6_release.py) | `reports/nns_vit_endpoint_e6_release_r3_20260918/` |

E2–E6 是带冻结协议、原子队列、资源门和不可变收据的正式工作流，不是快速 demo。历史 FAIL/恢复根属于证据链，不能删除或用新结果覆盖。

### 3.3 层深、attention/residual 与行为边界实验

| 分析 | 计算代码 | 分析代码 | 证据根 |
| --- | --- | --- | --- |
| 固定 token-channel 的层深空间轨迹 | [`endpoint_m3_atlas.py`](../../../src/vitnns/endpoint_m3_atlas.py)、[`run_endpoint_m3.py`](../../../scripts/run_endpoint_m3.py) | [`run_endpoint_m3_analysis.py`](../../../scripts/run_endpoint_m3_analysis.py) | `reports/nns_vit_endpoint_m3_r1_20260919/` |
| Q/K/V、routing/value、residual 轨迹 | [`endpoint_m4.py`](../../../src/vitnns/endpoint_m4.py)、[`run_endpoint_m4.py`](../../../scripts/run_endpoint_m4.py) | [`run_endpoint_m4_analysis.py`](../../../scripts/run_endpoint_m4_analysis.py) | `reports/nns_vit_endpoint_m4_r1_20260919/` |
| Head/Taylor 排序与跨层区域补充 | [`endpoint_m1_compute.py`](../../../src/vitnns/endpoint_m1_compute.py)、[`endpoint_m2_compute.py`](../../../src/vitnns/endpoint_m2_compute.py) | `scripts/run_endpoint_m1_*`、`scripts/run_endpoint_m2_*` | `reports/nns_vit_endpoint_m1_r1_20260918/`、`reports/nns_vit_endpoint_m2_r1_20260919/` |
| 独立图像队列、额外 checkpoint 和 H4 | [`revision_v5_extended.py`](../../../src/vitnns/revision_v5_extended.py) | [`run_revision_v5.py`](../../../scripts/run_revision_v5.py)、[`analyze_revision_v5.py`](../../../scripts/analyze_revision_v5.py) | `reports/nns_vit_revision_v5_r1_20260921/` |
| 三种坐标量敏感性 | [`revision_v6_coordinate.py`](../../../src/vitnns/revision_v6_coordinate.py) | [`prepare_revision_v6.py`](../../../scripts/prepare_revision_v6.py)、[`launch_revision_v6.py`](../../../scripts/launch_revision_v6.py)、[`analyze_revision_v6.py`](../../../scripts/analyze_revision_v6.py) | `reports/nns_vit_endpoint_revision_v6_r1_20260923/` |
| 独立 float64 解析参考与 mutation | [`revision_v6_analytic.py`](../../../src/vitnns/revision_v6_analytic.py) | [`run_revision_v6_analytic.py`](../../../scripts/run_revision_v6_analytic.py)、[`validate_revision_v6_experiments.py`](../../../scripts/validate_revision_v6_experiments.py) | 同一 revision-v6 根 |
| 实际 cut 图认证、三 registry、entropy/H4 敏感性 | `scripts/run_revision_v8_r2_analysis.py` | `scripts/validate_revision_v8_r2_analysis.py` | `reports/nns_vit_endpoint_revision_v8_r2_20260923/` |

当前论文数字与便携证据副本集中在 [`evidence/revision_v6/`](evidence/revision_v6/)。v8 增量证据位于上表的新根；大规模逐任务原始数组仍在各 `reports/` 根中。

## 4. 环境

### 4.1 已验证版本

当前工作区最后验证的关键版本为：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.21 |
| PyTorch | 2.2.2+cu121 |
| CUDA runtime | 12.1 |
| torchvision | 0.17.2+cu121 |
| timm | 0.9.16 |
| NumPy | 1.26.4 |
| SciPy | 1.12.0 |
| Matplotlib | 3.8.3 |
| Pillow | 10.2.0 |

Python 依赖锁位于 [`requirements.lock.txt`](../../../requirements.lock.txt)，包定义位于 [`pyproject.toml`](../../../pyproject.toml)。[`environment.yml`](../../../environment.yml) 只建立 Python 3.10 与 pip 的基础环境，不包含完整依赖。

### 4.2 使用当前工作区环境

从 `testcode/vitnns_rebuild` 运行：

```bash
source ../.conda/etc/profile.d/conda.sh
conda activate vitnns
python -m pip install -e . --no-deps
python -c "import torch,timm,numpy,scipy; print(torch.__version__, timm.__version__, numpy.__version__, scipy.__version__)"
```

也可以直接使用已验证解释器：

```bash
../.conda/envs/vitnns/bin/python3.10 -c "import torch,timm; print(torch.__version__, timm.__version__)"
```

### 4.3 新机器重建环境

```bash
conda env create -f environment.yml
conda activate vitnns
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.lock.txt
python -m pip install -e . --no-deps
```

正式 CUDA 实验使用确定性配置：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

历史生产运行使用 NVIDIA H20。不同阶段的正式启动脚本还会检查 GPU 型号、空闲显存、利用率、进程占用和共享锁。单图 demo 和论文图重绘均可 CPU-only 运行。

## 5. 数据、模型和证据资产

ImageNet 资产受 ImageNet 许可约束，不随 Git 仓库分发。大模型、JPEG、正式 `reports/` 和大数组也被 `.gitignore` 排除。新机器仅克隆代码并不足以复现全部实验。

### 5.1 模型 checkpoint

正式主模型由 [`model_registry.json`](../../../manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json) 绑定：

| Model ID | timm 名称 | 本地路径 | SHA-256 |
| --- | --- | --- | --- |
| `vit_b16` | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | `assets/models/vit_base_patch16_224_weights.semantic_candidate.pth` | `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d` |
| `deit_s16` | `deit_small_patch16_224.fb_in1k` | `assets/models/stage_x_x2/deit_small_patch16_224.fb_in1k.pth` | `cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f` |

额外 checkpoint 的精确名称和哈希在 [`manifests/nns_vit_revision_v5/protocol.json`](../../../manifests/nns_vit_revision_v5/protocol.json) 与补充材料 S15 中。

### 5.2 ImageNet 数据

主要本地资产为：

- `assets/imagenet/ILSVRC2012_img_val.tar`：官方 50,000 张 validation archive。
- `assets/imagenet/ILSVRC2012_devkit_t12.tar.gz`：官方 devkit。
- `assets/imagenet/imagenet_2012_validation_synset_labels.txt`：validation WNID 标签序列。
- `assets/data/nns_vit_endpoint_e2_evaluation_20260918/`：正式 Endpoint evaluation 的 400 张 materialized JPEG。
- `assets/data/revision_v5/`：revision-v5 的独立队列与控制样本。
- `assets/data/revision_v5_e1_r2/`：E1-R2 的匹配复核样本。

数据选择、排除、样本 SHA-256 和 source tar member 位于：

- [`cohort_registry.json`](../../../manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json)
- [`target_registry.json`](../../../manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json)
- [`intervention_registry.json`](../../../manifests/nns_vit_endpoint/ep0_r1_20260918/intervention_registry.json)
- [`statistical_analysis_plan.json`](../../../manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json)
- [`manifests/nns_vit_revision_v5/`](../../../manifests/nns_vit_revision_v5/)

不要按目录顺序重新挑图，也不要用模型正确率或结果选择替换样本。正式 worker 会核验 checkpoint 和 JPEG 的字节数与 SHA-256。

### 5.3 仅重绘论文图时需要什么

如果不重新运行完整模型实验，只重绘当前论文图，至少需要：

- 本目录的 `assets/*.npz` 与 `evidence/compute_receipt.json`。
- E3 的示例 component maps：`reports/nns_vit_endpoint_e3_closure_r1_20260918/`。
- E5 的示例替换曲线：`reports/nns_vit_endpoint_e5_spatial_r1_20260918/`。
- M3/M4 的 block summary CSV。
- `reports/nns_vit_endpoint_paper_revision_v1_20260920/analysis.json`。

完整绑定见 [`evidence/figure_sources.json`](evidence/figure_sources.json)。正式重绘器会先校验这些源文件的 SHA-256。

## 6. Start demo：一张图生成 Endpoint learning images

这是推荐的第一个可运行入口。它不需要 GPU，不读取 ImageNet 全量数据，也不会改写论文图包。它使用仓库中已有的 224×224 RGB 示例数组和 ViT-B checkpoint，在一个新输出目录中生成 learning images、head atlas、类别图与 block-0 edge decomposition。

从 `testcode/vitnns_rebuild` 运行：

```bash
CUDA_VISIBLE_DEVICES='' \
MPLCONFIGDIR=/tmp/nns_endpoint_demo_mpl \
OPENBLAS_NUM_THREADS=1 \
../.conda/envs/vitnns/bin/python3.10 scripts/run_endpoint_vit_scan.py \
  --output-dir /tmp/nns_endpoint_demo \
  --blocks 0 \
  --head-block 0 \
  --threads 4
```

`--output-dir` 必须是不存在的新目录。默认输入是：

```text
reports/nns_vit_unified_paper_figures_r9_20260917/source_arrays/original_rgb.npy
```

它必须是 `uint8 [224,224,3]` RGB crop。没有传 `--class-id` 时，类别目标使用该图的 clean top-1 logit。

预期输出：

```text
/tmp/nns_endpoint_demo/
├── raw_arrays.npz
├── scan_report.json
├── artifact_manifest.json
└── figures/
    ├── 01_block_00.png
    ├── 02_head_atlas.png
    ├── 03_classifier.png
    ├── 04_single_units.png
    └── 05_block_00_edge_decomposition.png
```

本 README 编写时已实际运行上述配置。结果为 `PASS_ENDPOINT_IMPLEMENTATION_DEMO`，扫描 24 个目标，完整运行约 11.7 秒；各项结构闭合最大绝对误差在约 $10^{-16}$ 量级。时间只说明当前主机上的起步成本，不是跨机器性能承诺。

需要 blocks 0/5/11 的更完整演示时，改为：

```bash
CUDA_VISIBLE_DEVICES='' \
MPLCONFIGDIR=/tmp/nns_endpoint_demo_mpl \
OPENBLAS_NUM_THREADS=1 \
../.conda/envs/vitnns/bin/python3.10 scripts/run_endpoint_vit_scan.py \
  --output-dir /tmp/nns_endpoint_demo_b0_b5_b11 \
  --blocks 0 5 11 \
  --head-block 5 \
  --threads 4
```

这个 demo 使用 float64 CPU 模型并保留原始图，内存和磁盘开销高于普通 inference。block-0 实测 `raw_arrays.npz` 约 82 MB。

## 7. 重建当前论文图片

### 7.1 只看已有图片

无需运行代码。打开 [`gallery.html`](gallery.html)，或查看：

- `figures/png/`：网页和 Markdown 使用。
- `figures/pdf/`：论文排版使用。
- `figures/svg/`：矢量编辑使用。
- [`figure_analysis.md`](figure_analysis.md)：逐图用途和证据说明。

当前包含 22 张图，其中正文引用 11 张。

### 7.2 正式图包重建顺序

以下命令会原地覆盖 `figures/`、manifest、gallery 和部分图注镜像。只应在源报告齐全、工作树状态已检查时运行。若只想试用，请先运行上一节的 `/tmp` demo。

从 `testcode/vitnns_rebuild` 运行：

```bash
git status --short

export CUDA_VISIBLE_DEVICES=''
export MPLCONFIGDIR=/tmp/nns_endpoint_publication_mpl
export OPENBLAS_NUM_THREADS=1
export NNS_ENDPOINT_PY=../.conda/envs/vitnns/bin/python3.10

# 可选：从绑定的模型、JPEG 和任务 manifest 重算论文展示数组。
# 如果 publication assets 已存在且哈希正确，脚本会复用。
$NNS_ENDPOINT_PY papers/nns_vit_endpoint/scripts/compute_publication_r2.py

# 生成基础 22 图、三种格式、gallery 与 manifest。
$NNS_ENDPOINT_PY papers/nns_vit_endpoint/scripts/render_publication_r2.py

# 应用当前概念图视觉系统，重画 Figures 1--7 并同步图注/manifest。
$NNS_ENDPOINT_PY papers/nns_vit_endpoint/scripts/render_concept_figures_v2.py

# 从真实数组重新检查显示、数字、双语链接与所有文件哈希。
$NNS_ENDPOINT_PY papers/nns_vit_endpoint/scripts/validate_publication_r2.py
$NNS_ENDPOINT_PY papers/nns_vit_endpoint/scripts/validate_revision_v6.py --stage final

git diff --check
git status --short
```

封存版本的 `validate_publication_r2.py` 曾通过 2,678 项检查，并从真实数组重算了 162 个 display panels。当前分支随后扩展了包内 `.gitattributes`，而历史 `artifact_inventory.json` 保留旧哈希；因此直接运行当前验证器可能只报告 `inventory_hash:.gitattributes`。这属于发布清单漂移，不是图件或科学数组失败。不要为消除它而静默改写历史 inventory；正式新发布应生成新的版本化 inventory。

不要使用 [`write_publication_r2.py`](../scripts/write_publication_r2.py) 重写当前稿。它是历史模板生成器，会主动拒绝无意覆盖当前 evidence-bounded 手工修订。

### 7.3 图件代码分别负责什么

| 脚本 | 作用 |
| --- | --- |
| [`compute_publication_r2.py`](../scripts/compute_publication_r2.py) | CPU 重算展示所需的真实 input、Endpoint、Grad-CAM、raw attention、rollout 和部分前向 feature。 |
| [`render_publication_r2.py`](../scripts/render_publication_r2.py) | 从封存数组和汇总 CSV 生成 22 张基础图及 PNG/PDF/SVG。 |
| [`render_concept_figures_v2.py`](../scripts/render_concept_figures_v2.py) | 使用统一视觉系统重画 Figures 1--7；Figure 1 以懒熊真实 Endpoint 开场，Figures 6--7 使用真实 E3 分项图，并同步 manifest、图注和 display receipt。 |
| [`validate_publication_r2.py`](../scripts/validate_publication_r2.py) | 验证图件哈希、显示计算、真实 panel、双语正文链接和便携包完整性。 |

## 8. 完整实验如何开始

完整论文实验不是一个 `run_all.sh`。它是按协议冻结、执行、验证、封存顺序形成的阶段链，且后续阶段不能在前一阶段失败时自动越过。当前 Endpoint 主轨已经完成，不建议为了生成图片重跑 400 图生产任务。

如需审计或在新根中复现，请先阅读：

- [`docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md`](../../../docs/NNS_ViT_Endpoint_Scanner统一研究计划与执行路线_20260918.md)
- [`manifests/nns_vit_endpoint/ep0_r1_20260918/README.md`](../../../manifests/nns_vit_endpoint/ep0_r1_20260918/README.md)
- 本 README 第 3 节列出的每阶段 runner、validator 和 terminal receipt。

原则上先运行 validator 复核现有根，而不是直接执行生产 runner。正式 runner 使用固定结果根，可能因根已存在而拒绝，也可能要求显式授权 token、H20 资格检查和全局任务队列。任何真正的全量重跑都应创建前瞻性的新协议与新结果根。

## 9. 快速验证当前论文包

不重绘图片，只验证当前文件。验证器会刷新对应的 `qc/*.json` 报告：

```bash
CUDA_VISIBLE_DEVICES='' \
../.conda/envs/vitnns/bin/python3.10 \
papers/nns_vit_endpoint/scripts/validate_publication_r2.py

CUDA_VISIBLE_DEVICES='' \
../.conda/envs/vitnns/bin/python3.10 \
papers/nns_vit_endpoint/scripts/verify_publication_r2_methods.py

CUDA_VISIBLE_DEVICES='' \
../.conda/envs/vitnns/bin/python3.10 \
papers/nns_vit_endpoint/scripts/validate_revision_v6.py --stage final
```

相关回归测试：

```bash
CUDA_VISIBLE_DEVICES='' \
../.conda/envs/vitnns/bin/python3.10 -m pytest -q \
tests/test_revision_v5.py tests/test_revision_v6.py
```

## 10. 阅读和复现时最容易混淆的边界

- **Endpoint 与 Gradient × Input**：同一 scalar target、同一当前点下是同一个基础量。
- **Branch partition 与 global cut**：Q/K/V、routing/value 和 heads 单独闭合到 attention branch；加入 residual identity 后才闭合到完整目标。
- **微分分组与替换变量**：routing-induced/value-induced 是解析微分项；有限替换实际修改的是前向 $A$、$V$、head output 或 residual branch activation。
- **当前依赖与参考变化**：Endpoint 描述观测状态的结构化局部依赖；reference-aware Taylor 在同一局部导数上加入实际位移，用于指定替换方向的一阶变化。
- **显示与统计**：蓝底正部图只用于空间阅读；统计使用 raw signed/absolute arrays。
- **独立队列与外部复现**：额外图像和 checkpoint 仍由同一代码和团队分析，不是外部团队复现。
- **结果完整性**：主文用正面对象关系组织叙事，H4、区域门槛和 all-family 的冻结数值仍完整保留在结果表、S18/S19 与机器证据中。

## 11. 当前包状态

revision-v8 包已经完成中英文 Markdown、LaTeX/PDF、共享 Bib 数据库、实际切面图认证、三类 registry、空间熵质量敏感性、H4 聚合匹配、四版本一致性、逐页视觉和匿名检查，以及六维对抗式复审。

现役 v8 r2 机器证据位于 reports/nns_vit_endpoint_revision_v8_r2_20260923；v8 r1 的 typed-NA 合同失败和既有 v7 历史根均按不可变证据保留。作者身份字段按匿名送审要求移除，终审结论为“可送审”。
