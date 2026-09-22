# 第五步证据与结论不匹配表及分级问题清单

## 证据与结论不匹配表

| 论文位置 | 当前结论 | 现有证据 | 缺口 | 允许表述 | 处理状态 |
| --- | --- | --- | --- | --- | --- |
| 标题、摘要、贡献二 | Endpoint 或结构热图公式本身是新方法 | G×I 恒等式、双索引定义、切面与审计 | 旧入口没有直接定义双索引对象 | 基础量是 G×I；技术增量是 $\mathcal L_{e,i}$、合法切面、审计和机制用途 | 已重写入口 |
| Section 2.5 | 联合能力在全部文献中绝对首次出现 | 截止 2026-09-22 的来源矩阵 | 文献检索不能证明绝对不存在 | 所核查来源未见一个方法覆盖全部联合字段 | 已限定来源与失效条件 |
| Sections 4.3、6.1 | 闭合证明结构解释具有因果正确性 | signed local differential closure、mutation 和数值 oracle | 未建立 finite intervention 因果充分性 | 已测切面无漏项或重复记账；有限效用由独立替换实验评价 | 已在讨论和结论分开 |
| Section 6.2 | 空间扩展证明双索引必要 | 总 G×I 图即可计算 broadening | 缺少 edge identity 的必要性 | 空间扩展是背景规律 | 已降级 |
| Sections 6.4–6.5 | routing/identity share 证明空间机制 | edge-level absolute mass | 标量消去了输入位置 | Q/K/V entropy 与 output–identity TV 支持空间重组；share 仅作尺度支持 | 已用 400 图结果替换核心证据 |
| Section 6.6 | 四 checkpoint 证明训练过程导致结构规律 | 四个公开参数状态、400 图/checkpoint | 无配对训练轨迹或独立训练种子 | 四个公开 checkpoint 的方向复现 | 已限定 |
| Sections 7.1–7.2 | 路径图普遍预测 finite intervention，或 Taylor 普遍更优 | 11/12 family、all-family 未获支持、H4 反向、规定区域 Taylor 结果 | estimand 和任务不同 | 路径图关联广泛但非普遍；Taylor 优势限于规定 reference/replacement | 已限定 |
| Figure captions | 蓝底正部表示正向因果支持 | display-normalized positive signed RGB sum | 显示规则不等于 finite deletion effect | 蓝底图用于空间阅读；数值结论来自 raw signed/absolute records | 主文与补充材料已说明 |

## 分级问题清单

### 阻断毕业

无。初审未发现伪造证据、不可复算的核心数字、缺失核心方法定义或将阴性结果删除的问题。

### 重大修改

无未处理项。初审的三项重大问题为入口未直接定义双索引对象、最强近邻失效条件不够显式、闭合与行为证据在结论中仍可更清楚分开。本轮均已通过双语重写和独立清单处理。

### 一般修改

1. 投稿前填写作者、单位和通讯作者字段，并按目标期刊匿名要求选择保留或移除。该项不改变科学结论。
2. 最终归档时填写 repository 与 archive 信息。目前的 `[REPOSITORY AND ARCHIVE REQUIRED]` 是发布元数据占位符。
3. 目标模板确定后检查正文图表浮动、参考文献样式和补充材料编号。

### 修辞优化

防御性写作审计分数为 1/10。可在目标期刊字数限制下继续压缩英文 Related Work，但不得删除最强近邻或把有界来源主张改为绝对首次。
