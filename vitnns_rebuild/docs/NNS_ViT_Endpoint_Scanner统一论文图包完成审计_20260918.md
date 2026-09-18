# NNS-ViT Endpoint Scanner：统一论文图包完成审计

日期：2026-09-18。审计对象是用户要求的“像
`nns_vit_unified_paper_figures_r10_r4_20260917` 一样，包含 Endpoint-ViT 论文所需所有
图片的一组图”。权威根为
`reports/nns_vit_endpoint_unified_paper_figures_r2_20260918`。R1 因英文 caption 混入中文而
降为非权威恢复证据；R2 修复语言分离和可移植 slug，不改变图像科学内容。

## 1. 具体完成标准

1. 一套独立、版本化、不覆盖旧 R10-R4 的图包；
2. 从方法定义到 RQ1–RQ4、结论和边界的完整论文叙事覆盖；
3. 每张图同时提供 PNG、PDF、SVG；
4. 提供索引、HTML gallery、中英文 caption 和 `Image #n` 中文逐图分析；
5. 所有实证图绑定 E2–E6 封存证据，不运行新模型实验或结果后选样；
6. 明确展示 RQ3 整体支持（11/12）与严格 omnibus C4a `NOT_SUPPORTED` 的分层；
7. 独立验证来源哈希、数值、格式、尺寸、链接、布局、视觉 QC 和 whole-root inventory；
8. 提供 manuscript-facing manifest 与主文/补充图映射。

## 2. Prompt-to-artifact checklist

| 用户要求 / 论文需求 | 具体产物 | 覆盖证据 | 状态 |
|---|---|---|---|
| 与旧统一图包同类的单一入口 | `README.md`、`gallery.html`、`figure_index.md` | 三个入口互链，28 张编号连续 | PASS |
| 所有图三格式 | `figures_png/`、`figures_pdf/`、`figures_svg/` | 28×3=84 个精确导出；validator 检查文件签名和 roster | PASS |
| 论文总览 | Figure 1 | 注册目标→Endpoint→切面→E4→E5→RQ3/RQ4 | PASS |
| 方法身份与数学定义 | Figures 2–10 | Endpoint/CNN/Path、Q/K/V、residual、target、LN/GELU、cuts | PASS |
| RQ1 注册扫描 | Figures 11–14 | 457/385 target、双模型 fixed-first-eight gallery、multi-target atlas、成本 | PASS |
| RQ2 结构正确性 | Figures 15–18 | cut 合同、76,800 closure、28,800 negative controls、真实 maps、6,144 JVP | PASS |
| RQ3 finite replacement | Figures 19–22 | 匹配干预协议、12-family forest、0.099565 临界图、次级 predictors | PASS |
| RQ3 科学解释 | Figures 20、21、28 | 11/12；唯一例外距阈值 0.000435；overall supported | PASS |
| 严格预注册字段保留 | Figures 20、28 | 明示 all-12 omnibus C4a `NOT_SUPPORTED` | PASS |
| RQ4 空间干预 | Figures 23–27 | 固定 4×4 区域、真实 maps、单图曲线、16 contrast、4×4 matrix | PASS |
| RQ4 基线结论 | Figures 26、27 | Endpoint−random 0.0011–0.0107；regional Taylor 四组均更高 | PASS |
| 结论与 Path 边界 | Figure 28 | Identity/Correctness/RQ3/RQ4/Path 五块分离 | PASS |
| 真实图而非只画示意 | Figures 12、13、17、24 | E2/E3/E5 原始数组与 fixed task；逐文件 SHA-256 | PASS |
| 无结果后挑图 | `render_manifest.json`、`source_inventory.json` | 固定 evaluation order 0、注册 target IDs、block 5 与注册 heads | PASS |
| 中英文 caption | `captions_en.md`、`captions_zh.md` | Figure 1–28 完整覆盖；英文无 CJK | PASS |
| 逐图中文分析 | `figure_analysis.md` | `Image #1`–`Image #28` 完整覆盖 | PASS |
| 主文/补充材料分配 | `manuscript/notes/endpoint_scanner_figure_mapping_r1.md` | 12 张主文建议、16 张补充建议 | PASS |
| 论文引用清单 | `manuscript/endpoint_figure_manifest.json` | 28 图的 PNG/PDF/SVG 路径、bytes、SHA-256 | PASS |
| CPU-only 与无新实验 | `source_inventory.json`、terminal | Torch/timm/model forward=0；`new_scientific_inference=false` | PASS |
| 机器验证 | `validation_report.json` | 来源、数值、格式、尺寸、链接、布局均 PASS | PASS |
| 人工视觉检查 | `manual_visual_qc.json` | 28/28 PNG 直接审阅并绑定 SHA-256 | PASS |
| 原子封存 | `whole_root_inventory.json`、`terminal_receipt.json` | 95 个非自引用文件、约 22.6 MB、terminal PASS | PASS |

## 3. 视觉返工记录

首次可审阅版本发现 Figures 1、15、19、23、28 有文字/流程拥挤，全部重排后从空正式根
重新渲染。最终 Figure 1、15、19、21、23、27、28 另做放大审阅；统计图、真实 atlas/map
和全部 inherited 解析图也通过 contact-sheet 复核。失败草稿未进入正式 manifest、inventory
或 terminal。

## 4. 结论

图包覆盖 Endpoint Scanner 论文从方法、扫描、正确性、RQ3、RQ4 到讨论边界的完整图像
需求；不存在仅用单一统计图替代方法图、真实图或协议图的缺口。当前 package terminal 为
`PASS_PRESENTATION_RENDER`，图包可进入论文排版。
