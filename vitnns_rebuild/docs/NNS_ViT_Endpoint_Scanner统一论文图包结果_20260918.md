# NNS-ViT Endpoint Scanner：统一论文图包结果

日期：2026-09-18。状态：`PASS_PRESENTATION_RENDER`。

正式根：`reports/nns_vit_endpoint_unified_paper_figures_r2_20260918`。R1 作为英文 caption
语言-QC 恢复前证据保留。权威 R2 图包包含 28 张编号
论文图，每张同时提供 PNG、PDF、SVG，共 84 个主导出；另有中英文 caption、
`Image #1`–`Image #28` 中文逐图分析、HTML gallery、来源清单、数值显示数据、布局审计、
28/28 人工视觉 QC、whole-root inventory 和 terminal receipt。

## 证据覆盖

- Figure 1–10：总体方法、Endpoint/CNN/Path 身份、attention Q/K/V、residual、target 与 cut；
- Figure 11–14：双模型注册 target atlas、固定 gallery、真实 multi-target atlas 与资源；
- Figure 15–18：结构切面、76,800 项闭合、28,800 项负例、真实 component maps 与
  6,144 项 JVP correctness；
- Figure 19–22：有限 replacement、12 个 C4a family、0.099565 临界例外与次级 predictor；
- Figure 23–27：空间干预、真实 map、单图曲线、16 个 baseline contrasts 与 RQ4 matrix；
- Figure 28：Identity–Correctness–Utility 总结。

Figure 21 明确呈现论文解释：RQ3 整体支持（11/12），唯一边界例外为 ViT-B、
normalized-zero、routing/value，0.099565 距 0.10 为 0.000435。Figure 28 同时保留冻结
机器字段 `omnibus C4a NOT_SUPPORTED`，没有修改预注册阈值。

## 验证

图包全程 CPU-only，没有模型 forward 或新科学推断。Figure 2–10 从封存的 Endpoint
数学图包逐字节继承；Figure 1、11–28 从 E2–E6 封存 JSON/NPY/NPZ/PNG 渲染。独立
验证器检查精确图号和格式、文件签名、PNG 尺寸、来源哈希、关键数值、RQ3/RQ4 文本、
caption/analysis 覆盖、链接、布局、视觉 QC 与 whole-root inventory，seal 前后均 PASS。
