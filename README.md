# 论文 1 项目目录说明

本目录同时保存了旧项目归档、历史实验结果、论文材料，以及正在重建的新实现。

当前开发应从 `vitnns_rebuild/` 开始。旧代码和旧结果主要用于追溯，不应直接作为论文结论继续引用。

## 目录说明

| 目录 | 内容 | 当前用途 |
|---|---|---|
| `vitnns_rebuild/` | 正在重建的干净代码，包括可信 direct baseline、阶段 B Value-path 归因、测试和阶段文档 | **当前主要开发目录**；阶段 C 及后续实现也应放在这里 |
| `49_vit ` | 原项目代码、模型、数据、小型 ViT、旧 NNS 扫描器和大量历史结果。目录名末尾有一个空格 | 只读归档和问题追溯；不要继续在里面堆叠修复脚本 |
| `scan_vit/` | 两张 ImageNet 图像生成的高密度旧 NNS 扫描张量与可视化，约 34 GB | 历史结果归档；受旧 NNS 数学和前向错误影响，不能作为论文主结果 |
| `ppt_results_nns/` | 从旧 NNS 扫描结果生成的 PPT 图、JSON 和分析报告 | 只可参考图表形式，不能作为有效实验结论 |
| `chatlog1/` | 项目历史聊天记录和附件 | 用于追溯研究目标、需求变化和原始算法来源 |
| `论文1/` | 参考论文、组会 PPT、论文结构、IEEE 模板和已有图片 | 论文材料与参考资料目录，不是当前代码实现目录 |

## 重要文件

| 文件 | 说明 |
|---|---|
| `论文1实现审计与后续计划.md` | 对旧代码、结果缺口和后续阶段的完整审计，是当前工作的总体路线图 |
| `vitnns_rebuild/docs/阶段D-step-by-step.md` | fidelity-only 阶段 D 的 D0–D10 串行计划、数据访问边界、论文产物映射与跨对话交接模板 |
| `vitnns_rebuild/docs/stage_d_scope_decision.md` | D0 冻结的 post-C10 fidelity-only 研究范围、历史状态、claim 边界与准备授权；不等于 Stage D 执行授权 |
| `vitnns_rebuild/docs/阶段A.md` | 阶段 A 的通俗说明。阶段 A 还没有实现 NNS，只建立可信的普通 ViT 基线 |
| `vitnns_rebuild/docs/阶段B.md` | 阶段 B 的方法边界、运行方式与真实单图验收结果 |
| `vitnns_rebuild/docs/stage_b_acceptance.md` | 阶段 B 的验收契约与 formal 判定规则 |
| `requirements.lock.txt` | 已安装环境的完整 Python 依赖锁定信息 |
| `.gitignore` | 控制只提交源码、配置和文档，排除模型、数据、缓存及大型实验产物 |

## 当前进度

阶段 A 已完成：普通 ViT 的模型、预处理、ImageNet 标签、Q/K/V、attention 和 logits 已建立可信基线并通过一致性验证。

阶段 B 已实现首个可审计的最小方法：固定目标 block 的真实 attention，
对该 block 的内部 `O=A@V` 标量生成输入归因，并以 conservation、重复性、
sanity、retention 和预算门禁验证。它不是完整 Q/K/V 归因。

Stage C/C0–C10 的执行和证据封存已完成。C10 v1 综合决策为
`NO_GO_NOVELTY`、acceptance 为 `BLOCKED`、blocker 为
`HOLM_BLOCKED_UNDERSPECIFIED`，spatial 为 `FAIL_SPATIAL`，且历史
`downstream_authorized=false`、`stage_d_started=false`。与此同时，固定 15 张、
非类别均衡 holdout 上，冻结 ViT-B/16、Value-path target、primary `(0,9)`、
matched patch baseline 和 fine-4×4-cell endpoint 的 fidelity component 为
`PASS_EFFECT`；这不是普遍 ImageNet 主张。D0 首次尝试因运行了模型/推理并解码
非-reserve JPEG 而保持 `BLOCKED`；随后独立、严格的 model-free/no-JPEG D0 重试
通过并冻结 fidelity-only scope，仅签发 `APPROVED_TO_PREPARE_STAGE_D`。后续按
`vitnns_rebuild/docs/阶段D-step-by-step.md`
执行 D1–D7 准备，在 D7 签发独立执行授权前不得启动正式 GPU workload。
当前 D0–D4 已完成；D4 仅用 synthetic/mock fixtures 实现并验证了
Stage-D-specific producer、image-group runner、sharding/recovery runtime、
immutable attempts 和 provenance。下一任务是 D5 synthetic/tampered
finalize/validator；正式执行仍未授权、未启动。

## 开始使用

```bash
source ~/.bashrc
conda activate vitnns
cd /data00/home/yeziqi.831/testcode/vitnns_rebuild
```

详细环境、运行方式和测试命令见 `vitnns_rebuild/README.md`。
