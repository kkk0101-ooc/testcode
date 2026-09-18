# NNS-ViT Endpoint Scanner：ENDPOINT-E5 结果

日期：2026-09-18。权威状态为：

    ENDPOINT_E5_PASS_SPATIAL_INTERVENTION_COMPLETE_NO_CLAIM_DECISION

## 1. 执行与覆盖

E5 在已发布 commit c876ea2170a4287bd9bd15baa0a933fa00eb31bd 上启动，使用 8 张
通过资格检查和文件锁的 NVIDIA H20。400 张冻结 evaluation 图像在 ViT-B/16 与
DeiT-S/16 上形成 800 个 image-model tasks，全部在第一次 attempt 完成：0 missing、
0 duplicate、0 invalid、0 retry。

每个任务执行 block 5 的三个预注册 head、normalized-zero/blur 两个 reference、
0/1/2/4 区域预算、Endpoint component map 与四个 mandatory baselines。总计保留：

- 115,200 条四点空间干预曲线，其中 96,000 条是 20 次冻结随机排列；
- 25,600 个单区域 regional-Taylor 前向结果；
- 345,600 个 finite-union full-forward 曲线点；
- 345,600 个 isolated-head evaluations；
- 前 16 个 evaluation order × 两模型的 32 组原始 float32 map 包。

## 2. 描述性数值

下表是每图三-head AUC÷0.25 后的 cohort 中位数。差值均为 Endpoint component map
减对应 baseline；这些是 E5 描述性汇总，不含 CI、p 值或 Holm 调整。

| 模型 | reference | Endpoint | total G×I | raw gradient | regional Taylor | random | Δ total | Δ gradient | Δ Taylor | Δ random |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT-B | normalized-zero | 0.030898 | 0.031896 | 0.030756 | 0.055842 | 0.029450 | -0.000932 | -0.000012 | -0.023433 | 0.001069 |
| ViT-B | blur | 0.027597 | 0.029304 | 0.028558 | 0.051761 | 0.022476 | -0.000994 | -0.000699 | -0.023697 | 0.004212 |
| DeiT-S | normalized-zero | 0.061694 | 0.061519 | 0.061348 | 0.096107 | 0.047547 | -0.001799 | -0.000559 | -0.029900 | 0.010689 |
| DeiT-S | blur | 0.039455 | 0.039526 | 0.039061 | 0.076953 | 0.029567 | 0.000177 | -0.000232 | -0.035630 | 0.008460 |

四个 family 中，Endpoint 对 random 的描述性中位差为正；对 regional Taylor 均为负；
对 total G×I 与 raw gradient 的差值接近零且方向混合。C4b 的正式判断需要 E6 按冻结的
共同有效集、效应阈值、bootstrap 精度门、16 个 baseline sign tests 和全 28 项 Holm
一起计算，因此本阶段不发布 SUPPORTED 或 NOT_SUPPORTED。

## 3. 独立验证与资源

独立验证器重新打开全部 800 个 immutable result，逐项验证 queue payload/seed/fencing、
图像和模型 identity、reference identity、曲线覆盖和有限性，并重算确定性排序、20 次
随机 permutation、每条 AUC、随机聚合与逐图配对差。对 32 组 raw map 包，它从数组
重新计算 16 个区域的 absolute mass 并绑定 SHA-256。验证结果为 PASS。

首次封存验证因把共享 ranker 数组误读为对象而在 RANKER_SCHEMA 停止。正式数据没有
修改；解释修订 NNS_ViT_Endpoint_Scanner_ENDPOINTE5封存验证器修订R1_20260918.md
发布后，完整验证通过。

E5 formal wall time 为 179.0970138909761 秒，8 卡上界为 0.3979933642021691
GPU-hours。E4-R3 与 E5 合计为 1.0279409489502467 GPU-hours、
466.41829150811105 wall-seconds。E5 封存前 retained bytes 为 378075471；峰值
reserved GPU memory 为 2290089984 bytes，峰值 CPU RSS 为 1859092480 bytes，
全部低于冻结 cap。

## 4. 下一授权

    ENDPOINT_E6_REQUIRES_EXPLICIT_USER_REQUEST

机器可读入口：

- reports/nns_vit_endpoint_e5_spatial_r1_20260918/terminal_receipt.json；
- reports/nns_vit_endpoint_e5_spatial_r1_20260918/validation_report.json；
- reports/nns_vit_endpoint_e5_spatial_r1_20260918/artifact_manifest.json；
- reports/nns_vit_endpoint_e5_spatial_r1_20260918/formal_coverage_audit.json；
- reports/nns_vit_endpoint_e5_spatial_r1_20260918/task_manifest.json；
- reports/nns_vit_endpoint_e5_spatial_r1_20260918/startup_snapshot.json。
