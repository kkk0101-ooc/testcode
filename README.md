# 论文 1 项目目录说明

本目录同时保存了旧项目归档、历史实验结果、论文材料，以及正在重建的新实现。

当前开发应从 `vitnns_rebuild/` 开始。旧代码和旧结果主要用于追溯，不应直接作为论文结论继续引用。

## 目录说明

| 目录 | 内容 | 当前用途 |
|---|---|---|
| `vitnns_rebuild/` | 正在重建的干净代码，包括模型加载、数据标签、ViT 中间量提取、测试和阶段文档 | **当前主要开发目录**；后续阶段 B 的 NNS-ViT 重写也应放在这里 |
| `49_vit ` | 原项目代码、模型、数据、小型 ViT、旧 NNS 扫描器和大量历史结果。目录名末尾有一个空格 | 只读归档和问题追溯；不要继续在里面堆叠修复脚本 |
| `scan_vit/` | 两张 ImageNet 图像生成的高密度旧 NNS 扫描张量与可视化，约 34 GB | 历史结果归档；受旧 NNS 数学和前向错误影响，不能作为论文主结果 |
| `ppt_results_nns/` | 从旧 NNS 扫描结果生成的 PPT 图、JSON 和分析报告 | 只可参考图表形式，不能作为有效实验结论 |
| `chatlog1/` | 项目历史聊天记录和附件 | 用于追溯研究目标、需求变化和原始算法来源 |
| `论文1/` | 参考论文、组会 PPT、论文结构、IEEE 模板和已有图片 | 论文材料与参考资料目录，不是当前代码实现目录 |

## 重要文件

| 文件 | 说明 |
|---|---|
| `论文1实现审计与后续计划.md` | 对旧代码、结果缺口和后续阶段的完整审计，是当前工作的总体路线图 |
| `vitnns_rebuild/docs/阶段A.md` | 阶段 A 的通俗说明。阶段 A 还没有实现 NNS，只建立可信的普通 ViT 基线 |
| `requirements.lock.txt` | 已安装环境的完整 Python 依赖锁定信息 |
| `.gitignore` | 控制只提交源码、配置和文档，排除模型、数据、缓存及大型实验产物 |

## 当前进度

阶段 A 已完成：普通 ViT 的模型、预处理、ImageNet 标签、Q/K/V、attention 和 logits 已建立可信基线并通过一致性验证。

阶段 A **不是 NNS 实验**，也没有生成新的 NNS learning image。下一步阶段 B 才会在 `vitnns_rebuild/` 中正式重写 NNS-ViT 的归因传播。

## 开始使用

```bash
source ~/.bashrc
conda activate vitnns
cd /data00/home/yeziqi.831/testcode/vitnns_rebuild
```

详细环境、运行方式和测试命令见 `vitnns_rebuild/README.md`。
