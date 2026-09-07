# Stage F F3H-R1 启动恢复补充协议（2026-09-07）

## 1. 目的与不可变失败记录

F3H v1 在四坐标 pilot 的模型反序列化、JPEG 解码和 attribution 开始前，因通用 recovery-v2 loader 重验一个后来发生文档性编辑的 amendment 文件 SHA 而 `BLOCKED`。该失败 root、pending receipt 和 failure event 永久保留，不覆盖、不删除，也不追认 PASS。

F3H-R1 只修复启动适配层，不改变 F3H0/F3H1 已冻结的样本、模型字节、预处理、方法、IG schedule、pilot 坐标、full-head workload、显示合同、预算或停止规则。F4 继续 `PAUSED_PENDING_F3H_AND_F3C`。

## 2. 精确恢复边界

R1 使用既有 `load_canonical_model`，但不构造或传入会重新读取旧 amendment SHA 的 `RecoveryV2Binding`。直接绑定如下不可分割身份：

- checkpoint：`assets/models/vit_base_patch16_224_weights.semantic_candidate.pth`；
- 文件 SHA-256：`489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d`；
- 架构：`vit_base_patch16_224.augreg2_in21k_ft_in1k`；
- canonical tensor digest：`150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd`；
- key count：152；parameter count：86,567,656；
- provenance identity：`RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718`；loader-local direct-pin label 仅表示绕过失效的 amendment 文档校验，不改变模型身份或字节。

模型 loader 仍必须在同一个稳定打开文件上完成文件 hash 与反序列化，并重新计算 canonical tensor digest。任何一项不符立即阻断。不得回退到历史 `2e718...` checkpoint。

R1 本地 sample adapter 只读取 F3H0 已冻结的 order 44：

- `assets/data/ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG`；
- image SHA-256 `c597ef7e061560542edbd77d2cbca6107cc44ff5e86de487a015eb47ae4620a8`；
- RGB，bicubic resize shorter side 248，224×224 center crop，mean/std 均为 `(0.5,0.5,0.5)`；
- clean tensor SHA-256 `f901fab0e43fbbc725aff3157ef7a91604cb30b4fdd92d2cb2f1aea49f092484`。

adapter 必须同时核对 D1 registry 行及冻结 F3H config；不得读取其他样本。

## 3. 新 root 与 Git 门禁

唯一 R1 formal roots 为：

- `evidence/stage_f/f3h_full_head_r1_recovery_v2_20260907/formal/`；
- `reports/stage_f_f3h_full_head_r1_20260907/`。

启动前必须满足：

1. F3H0 completion 与 `refs/archive/stage-f-v1-f3h0-protocol-freeze` 有效；
2. F3H1 completion 与原 `refs/archive/stage-f-v1-f3h1-implementation` 有效且该 ref 不移动；
3. F3H v1 failure root 存在且没有 terminal completion；
4. R1 producer 通过测试并由新 ref `refs/archive/stage-f-v1-f3h-r1-producer` 精确指向执行 HEAD；
5. local、tracking、remote 一致，worktree/index/LFS clean。

## 4. 执行顺序

R1 仍先运行四个冻结 pilot 坐标 `(0,0)`、`(4,0)`、`(8,0)`、`(11,0)`，共 8 个 Value-path terminals。在 pilot terminal、资源投影、无视觉访问门禁通过前，不得运行其余 136 个坐标或 Grad-CAM。

pilot 通过后才授权 8 个 pair-preserving shards 和 exactly one sealed Grad-CAM。全部 288 个 Value-path typed terminals 与 1 个 Grad-CAM terminal 完整 reconcile 后，才允许生成正负拆分、pair-shared log atlas、equal-head consensus 和 class-conditioned basis assessment。

## 5. F3C/F4 边界

F3H-R1 不从 atlas 外观、Grad-CAM overlap、亮度或 entropy 推断 head 权重。完整 atlas 只能按 F3H1 预声明条件判断现有信息是否足以给出 class-conditioned 聚合定义。F3H terminal completion 最多授权 `F3C0_PROTOCOL_FREEZE_ONLY`。

F3C0 必须在 F3H 完成后再冻结，并纳入全部五项比较：class-targeted NNS、ViT Grad-CAM、full-model class-logit IG、attention rollout、block-11 raw CLS attention head mean。F3C0 冻结不等于启动 F3C 计算；F4 在 F3C 完成前保持暂停。
