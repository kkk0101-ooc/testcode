# NNS-ViT Endpoint Scanner：统一论文图包恢复 R2

日期：2026-09-18。R1 图像、数值、来源和视觉布局均通过，但完成审计发现
`captions_en.md` 使用英文标题加中文正文，不满足中英文 caption 分离要求。R1 已封存，
保持不可变，不在原根修补。

R2 使用新根 `reports/nns_vit_endpoint_unified_paper_figures_r2_20260918`，保持 28 张图的
科学数据、图号、版式和 RQ3/RQ4 解释不变，只做两项前瞻性发布修复：

1. 为 Figure 1–28 提供完整英文 caption；
2. 将导出文件 slug 规范为仅含小写字母、数字和下划线。

R2 仍从原始封存 E2–E6 来源完整渲染，Figures 2–10 继续逐字节继承解析图；验证器额外
拒绝英文 caption 中的 CJK 字符，并检查 R1 terminal 保持存在且未被修改。
