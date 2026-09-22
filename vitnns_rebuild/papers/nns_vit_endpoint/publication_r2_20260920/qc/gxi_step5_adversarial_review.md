# G×I 第五步模拟对抗审稿与逐项修订记录

审查对象为第五步现役中英文稿。审查使用最强近邻，不用弱基线制造差异。结论先于修辞：基础 Endpoint 是 G×I，联合方法主张只有在对象、合同、审计和经验必要性同时成立时可保留。

| 对抗问题 | 审稿意见 | 初审证据 | 修订动作 | 修订后裁定 |
| --- | --- | --- | --- | --- |
| 是否只是 G×I 加 hooks | 仅按 hook 导出多个热图不足以构成方法。需要正式索引、use identity、切面合同和不可由总图恢复的结果 | 旧标题与摘要入口未直接给出 $\mathcal L_{e,i}$，容易被读成结构化可视化 | 双语标题加入“双索引”；摘要和贡献二直接定义 $\mathcal L_{e,i}$；Figure 1 与 Table 4 分别给出索引退化和信息必要性 | 已处理。增量落在 target-conditioned $[e,i]$ 对象、valid cut 和审计；基础 G×I 不承担新颖性 |
| 是否只是已有组件拼装 | Conductance、DecompX/LRP、EAP/Vi-CD 和 Unpack 已分别覆盖 hidden-unit path、component/relevance decomposition、edge intervention 和 named paths | Table 1 已统一比较九个字段，但入口段未明确“覆盖全部字段即失效” | 在 Related Work 增加最强近邻逐字段反例；新增创新成立/失效条件清单 | 已收窄。只保留来源边界内的联合字段主张，不保留任何单项首次主张 |
| 结构闭合是否被写成因果正确 | 链式法则闭合只能说明局部微分记账完整，不能证明函数值完整、机制唯一或 finite intervention fidelity | 方法、审计表和 Section 8.1 已分层，结论末句仍可更直接 | 重写双语结论，明确结构闭合与有限替换分别界定正确性和行为用途 | 已处理。闭合不再承担因果或干预主张 |
| 经验结果是否真的需要双索引 | routing share、identity share 和总图 broadening 都不需要同时保留 $e$ 与 $i$ | 第四步已将这些量降为背景或支持，核心结果为 Q/K/V entropy 和 output–identity TV | 摘要改为直接报告 Q/K/V 与 residual 空间变化；claim-to-evidence 绑定样本、区间和替代解释 | 已处理。Attention 与 residual 各有一项 400 图/模型的双索引必要性结果 |
| 四 checkpoint 是否构成强外推 | 两个新增公开权重不是独立训练种子，四个参数状态不能建立训练因果或全部 ViT 规律 | Section 5.1、6.6、8.1 已说明 training-recipe replication；H4 四项均反向 | 终审把适用范围固定为四个公开 checkpoint 和冻结 target/cohort；H4 与 H1–H3 同步保留 | 已处理。允许写方向复现，不允许写训练因果和架构普遍性 |
| 阴性行为结果是否削弱主线 | E3、E4 和 E5-H4 不支持路径图的稳定排序增量，若主张通用效用则证据冲突 | 主稿已报告“广泛但非普遍”、all-family 未获支持与 H4 反向 | 将科学主轴固定为结构化局部测量与机制描述；finite replacement 作为独立行为标定 | 已处理。效用边界收窄，没有追加搜索性实验 |

## 最强近邻逐字段反例

| 近邻 | 已有能力 | 缺少的联合字段 | 对本文的约束 |
| --- | --- | --- | --- |
| Conductance | 隐藏单元的 baseline-to-input path attribution；layer completeness | 具体 consumer-use edge 与 current-point attention/residual cut | 不能以“内部目标输入图”本身主张新颖 |
| DecompX / LRP | representation、logit 或 relevance 的组件传播和 conservation | 同一真实局部 Jacobian 在 consumer-use cut 上的 input-coordinate 分项 | 不能以组件分解、残差分配或精确闭合本身主张新颖 |
| EAP / Vi-CD | reference-specific edge effect、edge circuits 和 intervention fidelity | clean current-point $[e,i]$ 图组及其 signed cut closure | 不能以真实 edge 或 ViT activation patching 本身主张新颖 |
| Unpack | 命名 end-to-end paths、K/Q/V composition label 与 token credit | true-Jacobian input-coordinate maps 和合法切面重构 | 不能以命名路径或 Q/K/V 路径本身主张新颖 |

## 修订结论

四项核心攻击均已通过对象定义、主张收窄或证据绑定处理。处理方式没有改变阴性结果，也没有把结构闭合扩大为因果证据。剩余风险是文献覆盖随时间变化；`NOV-JOINT-1` 因此保持来源截止日限定。
