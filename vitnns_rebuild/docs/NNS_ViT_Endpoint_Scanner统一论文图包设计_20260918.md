# NNS-ViT Endpoint Scanner：统一论文图包设计

日期：2026-09-18。目标是在独立版本根中生成一套与
`reports/nns_vit_unified_paper_figures_r10_r4_20260917` 同等级、可直接服务 Endpoint
Scanner 论文正文与补充材料的完整图包。旧 R10-R4 不覆盖、不改写。

正式根：

```text
reports/nns_vit_endpoint_unified_paper_figures_r1_20260918
```

## 1. 固定图表清单

| Figure | 内容 | 证据角色 |
|---:|---|---|
| 1 | Endpoint Scanner study overview | 全文叙事总览 |
| 2 | Endpoint / CNN / Path method identity | 继承解析图 |
| 3 | Same target: Endpoint versus Path | 继承解析图 |
| 4 | Original CNN endpoint gates | 继承解析图 |
| 5 | Attention Q/K/V differential paths | 继承解析图 |
| 6 | Residual edge accounting | 继承解析图 |
| 7 | Unit, sum, and energy targets | 继承解析图 |
| 8 | Path, seed, and input coordinates | 继承解析图 |
| 9 | LayerNorm and GELU Jacobians | 继承解析图 |
| 10 | Structured edge cuts | 继承解析图 |
| 11 | Registered target atlas design | E2 target registry |
| 12 | Frozen first-eight endpoint gallery | E2 两模型真实图 |
| 13 | One-image multi-target atlas | E2 raw full atlas |
| 14 | Scan scale, latency, and resources | E2 完整 cohort |
| 15 | Structural closure contracts | E3 方法图 |
| 16 | Formal closure and negative controls | E3 统计 |
| 17 | Real structural component maps | E3 raw component arrays |
| 18 | Condition-scaled analytic JVP checks | E4-R3 correctness |
| 19 | Matched finite activation replacement | E4 方法图 |
| 20 | C4a family estimates and intervals | E6 confirmatory families |
| 21 | RQ3 threshold zoom and boundary exception | 11/12 pattern |
| 22 | Endpoint and registered secondary predictors | E4 描述性比较 |
| 23 | Spatial intervention protocol | E5 方法图 |
| 24 | Real block-5 component/baseline maps | E5 raw maps |
| 25 | One-image spatial effect curves | E5 curves |
| 26 | C4b mandatory-baseline contrasts | E6 confirmatory tests |
| 27 | RQ4 family summary | E6 4×4 baseline matrix |
| 28 | Identity–Correctness–Utility conclusion | 全文结论与边界 |

每张图必须有 PNG、PDF、SVG；PNG 最小 1200×700，供快速审阅；PDF/SVG 供论文排版。

## 2. 数据与访问边界

- CPU-only，`CUDA_VISIBLE_DEVICES=''`；不导入 Torch/timm，不加载模型，不运行新 forward。
- Figure 2–10 从已验证的 Endpoint 数学图包逐字节继承并按新论文序号复制。
- Figure 11–28 只读取 E2–E6 已封存 JSON、NPY、NPZ 和 gallery PNG。
- E2/E3/E5 raw arrays 仅用于显示和按既有公式重算 panel 数值；不新增样本选择、统计
  family、p value、bootstrap、Holm 或 scientific decision。
- RQ3 论文解释固定为：整体支持，11/12 C4a family 通过；唯一注册边界例外为
  `vit_b16 × normalized_zero × routing_value`，中位数 0.099565，距离 0.10 为 0.000435；
  冻结 12/12 omnibus C4a 字段仍为 `NOT_SUPPORTED`。
- RQ4 固定为 `NOT_SUPPORTED`；四个 family 均未同时超过四个 mandatory baseline。

## 3. 交付结构

除 84 个图文件外，图包必须包含：

- `README.md`、`figure_index.md`、`gallery.html`；
- `captions_en.md`、`captions_zh.md`；
- `figure_analysis.md`，按 `Image #1` 至 `Image #28` 逐图说明；
- `render_manifest.json`、`source_inventory.json`、`numeric_display_data.json`；
- `layout_audit.json`、`manual_visual_qc.json`、`validation_report.json`；
- `whole_root_inventory.json`、`terminal_receipt.json`。

验证器必须检查精确图号、三格式、文件签名、PNG 尺寸、来源哈希、数值绑定、链接、
caption/analysis 覆盖、RQ3/RQ4 文本分层、CPU-only 边界和 whole-root inventory。
