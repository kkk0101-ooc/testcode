# NNS-ViT Endpoint paper

当前版本为 [publication_r2_20260920](publication_r2_20260920/README.md)，论文和所有引用图片位于同一目录。

- [中文论文](publication_r2_20260920/zh/manuscript.md)
- [English manuscript](publication_r2_20260920/en/manuscript.md)
- [22 张主图总览](publication_r2_20260920/gallery.html)
- [逐图说明](publication_r2_20260920/figure_analysis.md)
- [逐项反馈验收](publication_r2_20260920/qc/completion_audit.md)

本轮补充了 NNS 概念总览、learning-image 初始化与生成、patch embedding、attention/residual 分解、两模型全部 12 层的学习图像与真实前向特征、以及 Endpoint/Grad-CAM/rollout/raw-attention 同输入比较。主图使用蓝底高亮样式，稀疏误差点和相关区间移入表格。全文按正式论文的动机、方法、结果与讨论组织。

zh/full_draft.md 和 en/full_draft.md 同步指向本轮正文；其图片路径链接到完整 publication 目录。早期 full_draft_m*、full_draft_r3_archive.md 和 full_draft_revision_v1_archive.md 为历史版本。原报告图包保留为来源，论文阅读请以本页面链接为准。

Validation from the repository root:

    ../.conda/envs/vitnns/bin/python papers/nns_vit_endpoint/scripts/validate_publication_r2.py
    ../.conda/envs/vitnns/bin/python papers/nns_vit_endpoint/scripts/verify_publication_r2_methods.py

Source estimates remain governed by the original evidence matrix and scientific decisions. The new six image-model qualitative comparisons are presentation-selected and add no population claims.
