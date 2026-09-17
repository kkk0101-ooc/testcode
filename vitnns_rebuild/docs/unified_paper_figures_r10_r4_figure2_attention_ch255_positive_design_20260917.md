# R10-R4：Figure 2 使用 attention channel 255 正贡献图

R10-R4 只重绘 Figure 2，其余 Figure 1、3–35 从已封存 R10-R2 逐字节继承。R10-R2 与本地中间根 R10-R3 均保持不变。

## 显示规则

- Figure 2 使用 Figure 30 中 B0 attention channel 255 背后的同一原始数组 `module_response_maps[1,1]`，目标为该通道在 196 个 patch-token 输出上的总和。不是从 PNG 截图。
- 右侧只保留 `positive=max(L_s,0)` 一张图，严格复用 Figure 30 channel 255 的逐面板正部 P99 规则。超过 P99 的值仅在显示层裁到 1。
- 主图使用 `config/nns_vit_unified_paper_figures_r4.json` 的蓝底 heat palette、nearest interpolation、无空间平滑。深蓝表示零或低正贡献，亮色表示更高正贡献。
- Figure 2 底部公式和 `Full signed map sum = delta s` 仍指向未裁剪的完整 signed 原始数组，并单独报告 displayed positive mass。
- 当前 channel-255 目标的完整 signed 对照位于 R10-R4 `supplement/figure_30_signed.*` 的 channel 255 面板。R10-R2 的旧 Figure 2（attention channel 0）另按 PNG/PDF/SVG 逐字节复制为 `supplement/figure_02_signed_original.*`，只作为历史显示备份，不冒充当前目标。

本次修改只改变论文展示层，不产生新模型前向、归因计算、样本选择、统计推断或科学结论。
