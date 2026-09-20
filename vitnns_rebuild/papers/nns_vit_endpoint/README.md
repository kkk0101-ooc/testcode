# NNS-ViT Endpoint paper

当前版本为 [publication_r2_20260920](publication_r2_20260920/README.md)，论文和所有引用图片位于同一目录。

- [中文论文](publication_r2_20260920/zh/manuscript.md)
- [English manuscript](publication_r2_20260920/en/manuscript.md)
- [22 张完整图件总览](publication_r2_20260920/gallery.html)
- [逐图说明](publication_r2_20260920/figure_analysis.md)
- [逐项反馈验收](publication_r2_20260920/qc/completion_audit.md)
- [修改意见 v2 完成审计](publication_r2_20260920/qc/revision_v2_completion_audit.md)
- [补充实验 v1（中文）](publication_r2_20260920/zh/supplement_experiments_v1.md) / [English](publication_r2_20260920/en/supplement_experiments_v1.md)
- [补充实验 v1 完成审计](publication_r2_20260920/qc/supplement_v1_completion_audit.md)
- [修改意见 v3 第一步冻结主张](notes/revision_v3_frozen_claims.md)
- [修改意见 v3 第一步完成审计](publication_r2_20260920/qc/revision_v3_step1_completion_audit.md)
- [修改意见 v3 第二步可信度实验（中文）](publication_r2_20260920/zh/revision_v3_step2_experiments.md) / [English](publication_r2_20260920/en/revision_v3_step2_experiments.md)
- [修改意见 v3 第二步后继主张](notes/revision_v3_step2_claims.md)

当前双语主文已执行修改意见 v3 第一步：题名与核心贡献限定为 NNS 在指定 ViT 计算图上的结构化实现、图相对路径分解和系统验证；四个 RQ 分别覆盖数学身份、结构化分解、冻结抽样中的层深规律与规定有限替换任务中的排序。相关工作及方法定位表已扩展，Section 4.4 给出等价重参数化反例；训练结果仅解释为训练参数状态与随机参数状态的差异。冻结主张见 [revision_v3_frozen_claims.md](notes/revision_v3_frozen_claims.md)。

修改意见 v3 第二步已完成多幅度 head-output replacement、400 图/模型的 Q/K/V 全体统计，以及 5 组独立 channel/token 抽样复现。完整结果、95% 图像 bootstrap 区间和轻量证据见上方第二步报告；替代训练 checkpoint 与同目标机制桥接仍未执行。整篇学位论文重写属于第三步，本轮未提前执行。

zh/full_draft.md 和 en/full_draft.md 同步指向本轮正文；其图片路径链接到完整 publication 目录。早期 full_draft_m*、full_draft_r3_archive.md 和 full_draft_revision_v1_archive.md 为历史版本。原报告图包保留为来源，论文阅读请以本页面链接为准。

Validation from the repository root:

    ../.conda/envs/vitnns/bin/python papers/nns_vit_endpoint/scripts/validate_publication_r2.py
    ../.conda/envs/vitnns/bin/python papers/nns_vit_endpoint/scripts/verify_publication_r2_methods.py
    ../.conda/envs/vitnns/bin/python scripts/validate_endpoint_supplement_v1.py --paper
    ../.conda/envs/vitnns/bin/python papers/nns_vit_endpoint/scripts/validate_revision_v3_step2.py
    ../.conda/envs/vitnns/bin/python scripts/seal_endpoint_revision_v3_step2.py --verify

Source estimates remain governed by the original evidence matrix and scientific decisions. The new six image-model qualitative comparisons are presentation-selected and add no population claims.
