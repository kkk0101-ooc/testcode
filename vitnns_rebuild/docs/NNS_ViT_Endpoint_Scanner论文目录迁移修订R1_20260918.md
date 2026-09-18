# NNS-ViT Endpoint Scanner：论文目录迁移修订 R1

日期：2026-09-18。用户要求把原 Path NNS-ViT 论文与 Endpoint Scanner 论文从共享的
`manuscript/` 拆为两个自包含 paper root：

```text
papers/nns_vit_path/
papers/nns_vit_endpoint/
```

本修订只迁移论文组织结构，不修改任何科学数组、统计、阈值、family decision、论文图或
封存结果根。Endpoint 的活动论文文件、证据笔记、QC、参考文献、验证器和 figure manifest
统一位于 `papers/nns_vit_endpoint/`；Path 稿对应内容位于 `papers/nns_vit_path/`。

E6 的已封存 run spec、reproduction manifest、source-read audit 和 terminal receipt 继续记录
生产时的 `manuscript/...` 路径。这些字段是不可变 provenance，不作结果后重写。活动 E6
验证器在校验这些历史字段时，将 `manuscript/<relative>` 显式解析为
`papers/nns_vit_endpoint/<relative>`，并继续要求 bytes 与 SHA-256 完全等于封存值。其他
路径不改写。

验证器修订边界：

1. 只增加旧论文路径到新 Endpoint paper root 的只读解析；
2. 不更改统计重算、read whitelist、source hashes、producer ancestry 或 decision checks；
3. Path 与 Endpoint 两套 figure manifest 分别保留 35 图和 28 图；
4. 旧 `manuscript/` 在全部文件迁移后移除，禁止继续作为混合写作根。
