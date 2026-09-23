# Revision v7 逐条评审意见处理表

状态采用规程限定的五值。位置均指最终 Markdown，LaTeX/PDF 由同一 Markdown 确定性生成并通过四版本验证。

## 第一份意见

| ID | 最终状态 | 最终动作 | 直接证据 |
| --- | --- | --- | --- |
| A01 | CONFIRMED | 分开定义结构合法切面与点态微分闭合 | zh/en §4.3；cut_definition_counterexample.json |
| A02 | CONFIRMED | 明确目标须位于 cut 下游，新增四类目标适用表 | zh/en §4.1、§4.3 |
| A03 | CONFIRMED | 证明改为多变量链式法则及 use-specific cotangent 分配 | zh/en §4.3 |
| A04 | CONFIRMED | 主文统计表写明双侧精确符号检验、tie、Holm 与 bootstrap 分工 | zh/en §5.3；pvalue_audit.csv |
| A05 | PARTIALLY_CONFIRMED | 主文删除未经限定的 confirmatory，补充保留本地冻结身份 | zh/en §5.3；S15 |
| A06 | CONFIRMED | 主文 Table 6 改报连续分层结果，11/12 移至补充 provenance | zh/en §7.1；S2.1、S18 |
| A07 | PARTIALLY_CONFIRMED | 说明一类一图为条件性重采样且 E5 图像零重叠不等于类别零重叠 | zh/en §5.3、S15 |
| A08 | CONFIRMED | 报告 A-only、V-only 和类型/层级调整结果 | zh/en Table 6；S18；v7 r4 CSV |
| A09 | PARTIALLY_CONFIRMED | 正文给出 replacement variable 与 local-map 对应表，补充给 slot 合同 | zh/en §4.2、§4.6；S16 |
| A10 | CONFIRMED | 主文给出 H4 两个排序量、reference、blocks、聚合、NA 和检验规则 | zh/en §6.6；S18；H4 CSV |
| A11 | PARTIALLY_CONFIRMED | 统一为“区域诱导的指定-head activation 替换” | zh/en §4.6、§7.1 |
| A12 | CONFIRMED | 直接报告 local map 无稳定实用排序增益 | zh/en §7.1、§8.1 |
| A13 | CONFIRMED | 八类坐标核心结果进入主文 Table 5a，96 行留 S17 | zh/en §6.7、S17 |
| A14 | ALREADY_RESOLVED | 保持 block-index association 与 replacement estimand，不扩大因果语言 | zh/en §6–§8 |
| A15 | CONFIRMED | 改为独立图像队列和额外公开 checkpoint 上保持同方向 | zh/en §6.6 |
| A16 | ALREADY_RESOLVED | 摘要、结果和结论均限定固定 checkpoint/target/block | zh/en Abstract、§6、§8 |
| A17 | PARTIALLY_CONFIRMED | 主文保留研究设计和关键数值，运行计数与完整 family 置于补充 | zh/en §5–§7；S15、S18 |
| A18 | CONFIRMED | 主文 Table 1 缩为五个近邻族，完整矩阵置 S4；实时核验六项 2026 预印本 | zh/en Table 1；S4；references.bib |
| A19 | PARTIALLY_CONFIRMED | 新增中英术语合同，首次说明路径图不是唯一因果路径 | zh/en Introduction；S4.1 |
| A20 | CONFIRMED | 修复 −<0.0001、+<0.0001 和极小 p 值格式 | zh/en Tables 8、§5.3；PDF build gate |
| A21 | PARTIALLY_CONFIRMED | 四份 PDF metadata、全文、源文件与临时产物扫描通过 | publication_validation_report.json |
| A22 | PARTIALLY_CONFIRMED | 采用证据边界并按实际分层结果重写，不复制评语结论 | zh/en Abstract、§7、§8 |

## 第二份意见

| ID | 最终状态 | 最终动作 | 直接证据 |
| --- | --- | --- | --- |
| B01 | PARTIALLY_CONFIRMED | 引言明确总图丢 use、edge scalar 丢空间的不可辨识问题 | zh/en Introduction |
| B02 | ALREADY_RESOLVED | 贡献保持为表示、审计与固定参数状态描述 | zh/en Introduction、Conclusion |
| B03 | PARTIALLY_CONFIRMED | 主文按对象、索引、闭合目标比较五个最强近邻族 | zh/en Table 1 |
| B04 | ALREADY_RESOLVED | 保留重参数化反例，将错误检测限定为声明合同的一致性检查 | zh/en §4.3–§4.4 |
| B05 | ALREADY_RESOLVED | 统一写作与 block index 相关 | zh/en §6、§8 |
| B06 | PARTIALLY_CONFIRMED | 说明 block 5 和首/中/末 head 为结果访问前覆盖选择 | zh/en §4.6；S16 |
| B07 | PARTIALLY_CONFIRMED | mutation 统一为拒绝规定人工故障 | zh/en Abstract、§4.3 |
| B08 | PARTIALLY_CONFIRMED | 给出索引删除后的不可恢复量，不伪装为性能消融 | zh/en §4.3、§6.3 |
| B09 | PARTIALLY_CONFIRMED | 解释 Chebyshev 半径 1，并完成半径 0/1/2 CPU 敏感性 | zh/en §4.5；S18.3 |
| B10 | PARTIALLY_CONFIRMED | 区分总体样本与展示样本，空间方差通道仅用于图示 | zh/en §5.1–§5.2 |
| B11 | ALREADY_RESOLVED | 主文统计表列出 family，完整 28 项留补充/机器证据 | zh/en §5.3；pvalue_audit.csv |
| B12 | ALREADY_RESOLVED | 各 checkpoint 分别估计，不解释绝对幅值为架构效应 | zh/en §6.6、§8.2 |
| B13 | CONFIRMED | 增加一阶局部性、非线性交互、饱和、参数化和标量目标限制 | zh/en §8.2 |
| B14 | PARTIALLY_CONFIRMED | routing/identity scale 定义为局部微分 absolute-mass 比 | zh/en §4.5；S4.1 |
| B15 | PARTIALLY_CONFIRMED | 按 absolute-mass 分位组报告 TV；结果不只在最低组，仍不声称排除噪声 | S18；residual_tv_quality_diagnostics.csv |
| B16 | PARTIALLY_CONFIRMED | 集中列出 ImageNet、patch-16、固定 scalar/replacement 边界 | zh/en §8.2 |
| B17 | PARTIALLY_CONFIRMED | 报告模型异质性但不归因训练、规模或数据 | zh/en §8.2 |
| B18 | PARTIALLY_CONFIRMED | 主文保留主问题与关键负结果，完整运行计数置补充 | zh/en §6–§7；S1、S18 |
| B19 | PARTIALLY_CONFIRMED | 公式前给直观句，术语表集中定义 | zh/en §3–§4；S4.1 |
| B20 | PARTIALLY_CONFIRMED | LaTeX/PDF 使用 label/ref 和展开参考文献，不显示工程路径 | tex、pdf、publication validator |
| B21 | NOT_REQUIRED | 核心缺口由旧原始记录 CPU 复算解决，未运行新模型推理 | v7 r4 receipt：GPU inference 0 |

最终统计：第一份意见 CONFIRMED=12、PARTIALLY_CONFIRMED=8、ALREADY_RESOLVED=2；第二份意见 CONFIRMED=1、PARTIALLY_CONFIRMED=14、ALREADY_RESOLVED=5、NOT_REQUIRED=1。无 BLOCKED_BY_EVIDENCE 项。
