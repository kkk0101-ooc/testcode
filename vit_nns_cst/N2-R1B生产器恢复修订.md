# NNS-CST N2-R1B 生产器恢复修订

日期：2026-09-22  
状态：`FROZEN_BEFORE_R1B_PAIR_GENERATION_AND_MODEL_OUTCOME_ACCESS`

N2-R1 在没有加载 checkpoint 或读取任何模型 outcome 前完成 8 个数值匹配 pair，但盲化图像质检发现最近背景像素填充会在 donor mask 区域产生明显块状或放射状纹理。R1 已由 `evidence/n2_r1_20260922/qc_rejection_receipt.json` 终止，所有文件保持不变，不可作为 N2 可识别性 PASS。

R1B 仅恢复 producer：donor foreground mask 先膨胀 3 pixels，再使用 OpenCV 4.9.0 Telea inpainting、radius 3。此操作仍只使用 donor RGB 与外部 mask，不读取模型 outcome。recipient、donor roster、语义标签、四臂、低级 matching、行为门、层、rank、readout、Path NNS、JVP、剂量和全部阈值均继承 R1。R1B 生成到 fresh `evidence/n2_r1b_20260922`，通过独立数值重算和盲化图像 QC 后才允许签发模型执行 freeze receipt。
