# G×I 五步计划第三步：防御性写作审计

审计范围：现役中英文方法章节、Figure 1–2 图注、方法消融表和审计协议表。采用 `academic-defensive-writing-auditor` 的 claim → evidence → concise boundary 规则。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 双索引定义 | 容易用“不是普通 G×I 加 hooks”作为开场 | D1 / D14 | Major | 否 | 先回应假想质疑，未先定义对象 | 会继续要求可检验的索引和合同 | 直接给自由索引、目标条件和 use-specific cotangent | 定义 $\mathcal L_{e,i}(s;x)$，再说明固定或删除索引的结果 |
| 合法切面证明 | 容易把链式法则证明写成理论创新 | D14 | Critical | 是 | 标准恒等式被用于补偿方法定位 | 会认为贡献来自重新命名 | 将证明和方法增量分开 | 命题来自链式法则；增量是双索引对象及可审计 cut contract |
| Attention 与 residual | 容易写成首次 Q/K/V 或 residual 分解 | D14 | Critical | 否 | 单项能力已有直接先例 | 最近邻工作会直接反驳 | 绑定联合主张 ID | 两类切面分别实例化 `NOV-ATTN-1` 与 `NOV-RES-1`，不主张单项首次 |
| 审计协议 | 容易用“严格审计保证解释正确”概括 | D10 / D13 | Major | 是 | 从实现合同越界到解释有效性 | 会追问因果、语义和未测错误类型 | 按检查层报告能排除的错误 | 三类 mutation 只支持对应漏项/重复检出；方向 oracle 支持局部数值正确性 |
| 未封存错误类型 | 逐项重复“我们没有验证” | D2 / D8 | Minor | 是 | 多次否认会形成免责声明堆叠 | 会弱化已完成合同 | 集中为一条正面边界 | Wrong-slot、fan-out merge 与 detach mutation 尚无封存结果，不列为已验证检出能力 |

### Paper-level defensive-writing score

2/10。方法段先定义对象、合同和直接证据，再集中说明一个影响解释的边界；没有用宣传性形容词、审稿人预答辩或重复免责声明维持创新定位。

### Highest-priority removals

1. 不使用“不是 G×I 加 hooks”作为贡献句。
2. 不把链式法则或闭合等式本身写成创新。
3. 不写“首次 Q/K/V”“首次 residual 分解”或“保证解释正确”。
4. 不把 attention-detach 一致性检查计为第四类 mutation。
5. 不把未执行的 wrong-slot、fan-out merge 或 detach mutation 写成检出能力。

### Necessary caveats to preserve

- $\mathcal L_{e,i}$ 相对于指定目标、计算图、consumer use、切面和当前前向状态定义。
- 合法切面闭合证明不建立实现不变性、函数值分解或因果贡献。
- 已封存 mutation 只覆盖 omit-head、omit-routing 和 duplicate-identity。
- Attention-detach 的 384 项结果是独立一致性检查。

### Repeated phrase audit

现役双语主稿扫描 `We emphasize that`、`It should be noted that`、`We do not claim`、`To avoid misunderstanding`、`Taken together`、`需要强调的是`、`应当指出`、`为避免误解`、`我们不声称`，计数均为 0。
