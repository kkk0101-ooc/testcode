# Learning image 与论文结论的文字审查

对象：[论文写作前置条件与结论证据图表映射](论文写作前置条件与结论证据图表映射_20260916.md) 的 R8 修订。
按 academic-defensive-writing-auditor 检查：先说研究内容，再给证据与必要范围。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| 1.1 开头 | READY_FOR_SCOPED_MANUSCRIPT_DRAFTING 等状态先行 | D8 | Major | Mixed | 工程状态遮住研究内容 | 难以知道论文研究什么 | 状态信息移到交接部分 | 当前已有结果足以开始写作。 |
| 1.2 主线 | 固定内部标量与实现细节直接充当总论点 | D7 | Major | Mixed | 实验限制先于研究问题 | 主线过窄 | 先用 learning image 概括，再在小结论限定目标 | 在输入图像坐标下查看内部响应及其贡献组织。 |
| 原 Figure 2 解读 | 同一 H9 换层，图也会变 | CLEAN | Minor | Yes | 主要是概念不精确，不是防御性写法 | 容易误以为跨层共享同一个 head | 明确不同参数组；Figure 2 专讲符号 | 选定一个响应，查看其带符号输入贡献。 |
| 1.9 方法贡献 | 输入分辨率或工程完整性代表原创性 | D14 | Major | Yes | 用结果形式代替技术差异 | 分不清继承和新增 | 写出 attention、路径积分、完整切分的差异及继承 | 采用 IG 和链式法则实现 ViT 输入贡献与成分分解。 |
| 3 节小结论 | 单图图谱差异直接推到所有 head 的分工 | D10 | Major | Yes | 形态观察被放大为机制结论 | 证据不足 | 区分描述性图谱和联合检验 | 图谱不同；稳定组织的联合检验仍不确定。 |

### Paper-level defensive-writing score

本说明文档估计 3/10；新方法与结论段落较直接，后半部分仍保留结果边界和投稿检查内容。
这是写作说明文档的评分，不是最终论文的评分。

### Highest-priority removals

1. 开头用 NNS/learning image 与 ViT 的研究问题取代状态码。
2. 统一旧版图表描述到 R8，去除已经完成却仍写“待重画”的条目。
3. 用具体算法差异取代“更细就更新颖”的暗示。
4. 用一个问题表组织 patch、head、层、分支的正结果与未决问题。

### Necessary caveats to preserve

- 单 head、全部 attention 固定的类别响应、完整类别分解是三个不同对象。
- H10 是事后讲解选择，主实验仍为 H9。
- learning image 的符号是相对参考的路径归因，不等于单区域遮挡的因果结论。
- 完整切分内可求和，跨层切分不能重复求和；F3H 显示平均不是类别归因。
- Y 的 patch mixing 为 INVALID，head organization 为 INCONCLUSIVE；S 为事后探索。

### Repeated phrase audit

对修订后映射文档计数：“我们不声称”0 次，“为避免误解”0 次，“不能”28 次，“不等于”6 次。
后两项主要用于定义、限制和例句清单；计数用于定位重复，不把必要的数学条件自动删去。

### 事实核对

原论文依据：第 3 节式 (1)–(14)、第 4.1.4 节残差实验、第 4.2.1 节聚合、第 4.2.3 节池化显示。
ViT 依据：stage_c_value_path.py、stage_f_f3c1_kernels.py、stage_x_nns_v2.py 与 stage_x_protocol.md。
结论依据：D10 原始逐图差值、F3C4 contrast_summary、X1 identity/complete-cut 验证、Y4-R5 claim decisions 和 S5 endpoint decisions。
本轮重新计算这些关键 JSON 的文件 SHA-256 和已有 canonical digest，并复算 D10 的 500 图正差值计数与中位数。
