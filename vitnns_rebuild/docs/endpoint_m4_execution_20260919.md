# Endpoint M4 冻结执行与验收协议

本轮仅执行《阶段M_endpointnnsvit.md》的 M4-A、M4-B 和 M4-C，完成后停在 M4，不进入 M5 或 M6。入口必须重新验证 M3 `ENDPOINT_M3_PASS_EXECUTION_AND_AUDIT` 及其 manifest。正式生产者代码须先提交并被当前远端分支包含，之后才允许读取冻结 400 图。

两模型共享 M1 的 400 图 cohort，目标为冻结 ground-truth logit，覆盖全部 12 blocks 和全部 heads。Attention 对每个 image/block/head 保存 Q、K、routing、value 与 native full Endpoint 的流式统计；frozen-attention 为只保留 $A\,dV$ 的局部微分。每图固定 8 个单位范数 Rademacher 输入方向，报告 native JVP、显式 softmax differential 与 $A\,dV$ 的误差。平均 query-row saturation 与 CLS-query saturation 均保留；主要相关分析使用前者。

Residual 分别在 attention add 和 MLP add 固定 clean output cotangent，使用该 cotangent计算 identity/carry 与 branch Endpoint。每项报告 share、path cancellation、cosine、三种 map cancellation 和 output-vs-identity patch-distribution total variation。零质量、零范数或常数 head 向量不加 epsilon，派生量记 NA。闭合需同时满足元素级 `atol + rtol` 以及 absolute 或 relative summary 分支；每个 residual add 都执行省略非零 identity/carry 的负向闭合控制。

统计单位是图像。Attention phase 值先在 block 内对 heads 取中位数，再在 phase 内对 blocks 取中位数；saturation relation 先在每图每层 heads 内算 Spearman，再对 12 层取中位数。Residual phase 值在图内对四个 blocks 取中位数。所有跨图推断报告中位数、10,000 次 image bootstrap 95% CI、双侧 exact sign test 与冻结 Holm family；未定义项以 `p=1` 留在 family 中。

| M4 原要求 | 冻结实现 | 验收证据 |
|---|---|---|
| 10.1 Q/K/V 与 routing/value | 每 head 显式 Q/K/V VJP，routing=Q+K | native-head closure、逐 head 表 |
| 10.2 frozen attention | $E^{FA}=E^{value}$；full-value=routing | Endpoint identity error 与 detach 对照 |
| 10.3 两模型、400 图、12 层、全 heads | 800 个动态队列 task | manifest、queue 与 coverage audit |
| 10.4 routing share、Endpoint/gradient error | 无 epsilon流式统计 | per-image records、summary tables |
| 10.5 native/explicit/detach/parity/closure/JVP | 全 task parity/closure/8-direction JVP；前16 order detach | validation report |
| 10.6 saturation 层内关系 | 先 layer 内 head Spearman，再 image 汇总 | image values、Holm tests、图 |
| 11.1 正确 clean cotangent | attention 使用 $g_{r1}$，MLP 使用 $g_{r2}$ | 单元测试、closure |
| 11.2 两类 residual 全覆盖 | 19,200 image-block-type records | coverage audit |
| 11.3 share/cancellation/cosine/spatial change | patch absolute distribution TV | residual tables、图 |
| 11.4 六个机制问题 | 深度、类型、抵消、保留/改写及分模型报告 | 结果文档与论文段落 |
| 11.5 省略 identity 负控 | 每个 nonzero identity/carry exact closure 必失败 | negative-control records |
| M4-C signed/absolute/cancellation | 分开 map 与 path cancellation | cancellation tables |
| 15 统计原则与 Holm families | image-first、10k bootstrap、sign、Holm、NA保留 | independent CPU audit |
| 允许结论边界 | 只称 local input dependence，不升级为 causal importance | 中英文稿与写作审计 |
| 本轮 scope 外 | M5、M6 与新 predictor 竞赛不执行 | terminal `STOP_AFTER_M4` |
