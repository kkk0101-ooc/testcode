# NNS-ViT Endpoint Scanner 数学分析：写作审计

审计对象：[数学分析及方法对比](NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md)。本审计只检查防御性学术措辞，同时保留影响数学解释的条件。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| 1 节方法身份 | 反复说明 Endpoint 不是新公式 | D14 | Major | 是 | 多次否认新颖性会遮住实际技术内容 | 仍会追问具体增量 | 只在结论和贡献边界各保留一次 | 基础归因是 Gradient × Input；技术内容是内部目标注册与完整计算边分解。 |
| 3.5 原论文公式 | 原论文公式等于 Gradient × Input | MIXED | Critical | 是 | 未限定末端 ReLU 门的记号歧义会把解释写成无条件事实 | 追问式 (11) 的 $W_q^l$ 是否遗漏目标门 | 给出门控一致解释及字面解释的后果 | 在式 (6)、(10) 定义的门控一致解释下，单神经元图等于 Gradient × Input。 |
| 3.6 滤波器聚合 | 原论文所有 CNN 图都可统称 Gradient × Input | D14 | Critical | 是 | 忽略了 feature-map activation-weighted aggregate | 会指出实验聚合目标与单神经元目标不同 | 推导 energy target | 论文的滤波器聚合等于 $\frac12\lVert h\rVert_2^2$ 的 endpoint 图。 |
| 7 节输入坐标 | Endpoint 没有 baseline | MIXED | Major | 是 | 只强调“没有 baseline”会掩盖 normalization 原点 | 会追问 $x=0$ 对应什么 RGB 值 | 正面写出 affine 坐标关系 | 归一化输入上的 Endpoint 在 RGB 坐标中以 $r=\mu$ 为原点，但不运行参考输入前向。 |
| 2.3 节相等条件 | Jacobian 恒定时才相等 | D13 | Critical | 是 | 把充分条件写成必要条件 | 会给出非恒定梯度但路径平均等于终点的反例 | 写出逐坐标充要条件 | Jacobian 恒定时必然相等；其他情况下也可能因路径平均或零输入差值而一致。 |
| 10 节 reference gate | Endpoint 没通过 reference stability | MIXED | Major | 是 | 容易把扩展定义的比较写成 canonical 方法参数 | 会质疑无 reference 的 Endpoint 如何有 reference sensitivity | 写明两个 multiplier | Gate 比较 $x\odot\nabla s$ 与 $(x+1)\odot\nabla s$；它不是 canonical Endpoint 的内生参数。 |
| 10 节闭合 | 极小误差证明方法成立 | D10 | Major | 是 | 把代码 parity 扩展成归因有效性 | 会区分数学恒等式、实现测试和 fidelity | 分开报告三层证据 | 链式法则给出数学身份；数值闭合验证实现；fidelity 由干预指标评价。 |
| 10 节结果 | NO_GO 后继续为实现辩护 | D12 | Major | 是 | 先回应质疑会弱化结果陈述 | 认为作者在回避负结果 | 直接写当前用途 | 当前证据支持终点内部分析，不支持替换 path-integrated 主归因方法。 |

### Paper-level defensive-writing score

2/10。正文以定义、推导、逐项映射和实测数值为主；方法身份、原论文记号边界和 20 图负结果各集中说明一次。修订后将 Jacobian 恒定写成充分条件，将 pilot 的 two-multiplier gate 与 canonical Endpoint 分开。

### Highest-priority removals

1. 保持“基础量是 Gradient × Input”只出现在方法身份和贡献边界两处。
2. 保持原论文式 (6)、(10)、(11) 的具体关系，不换成笼统的忠实性辩护。
3. 保留 20 图的 fidelity、flip 和 runtime 数值，不添加“更好、更稳健或更实用”等形容词。
4. 将 Scanner 的贡献固定为目标注册、native-forward 扫描、结构化分解与闭合验证。

### Necessary caveats to preserve

- 原 NNS-CNN 的 Gradient × Input 等价依赖输入值 seed、当前门控、目标可导和门控一致解释。
- 原论文 filter aggregate 与单神经元图、channel-sum 图是不同目标。
- 规范化输入改变 Endpoint 的坐标原点。
- bias 和模型常数影响状态与 Jacobian，但没有直接像素 seed。
- Computational edge cuts 与 attention differential paths 分别重构同一总图，不能跨分解或跨层相加。
- Jacobian 恒定是 Endpoint 与 IG 相等的充分条件；逐坐标等式才是充要条件。
- 20 图 reference gate 比较两个 baseline-relative multiplier，不是 canonical Endpoint 的方法参数。
- 数值闭合验证代码符合定义，不证明 attribution fidelity 或因果意义。
- 20 图结果不支持用当前 Endpoint 协议替换 path-integrated 主归因方法。

### Repeated phrase audit

最终正文中：“我们不声称”0 次，“为避免误解”0 次，“需要强调”0 次，“应当指出”0 次。“不是”与“不能”主要出现在数学对象辨析和切面求和规则中，没有形成连续免责声明段落。
