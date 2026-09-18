# NNS-ViT Endpoint Scanner：ENDPOINT-E6 恢复 R2

日期：2026-09-18。E6-R1 在已发布 commit
`adde84e919c1d230b14c3ef3e3793083fc1dc814` 上启动，逐文件读取 E4/E5 已封存 JSON
后，在进入 bootstrap、sign test、Holm、family decision、论文写出之前因
`NameError: name 'selected' is not defined` fail closed。不可变现场为
`reports/nns_vit_endpoint_e6_release_r1_20260918/failure_receipt.json`，SHA-256 为
`be15f9995ba7832a4aabcc1aed025f97be28d76558af28ba300fde4349f789fa`。

R2 只修复 coverage 标志的计算来源：`exact_coverage` 由当前 family 的所有输入列是否
都恰有 400 个图像级值决定。统计量、阈值、seed、10,000 次 bootstrap、28-test Holm、
四态决策、读白名单和论文主张均不改变。R1 根不删除、不覆盖、不升级为 PASS。

R2 使用新根：

```text
reports/nns_vit_endpoint_e6_release_r2_20260918
```

本修复和 run spec 更新必须先提交并推送；确认本地、tracking 和 remote commit 一致后，
才可从头重跑正式 E6。
