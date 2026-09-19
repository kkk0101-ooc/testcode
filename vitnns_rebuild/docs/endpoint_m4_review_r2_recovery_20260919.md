# Endpoint M4 Review R2 恢复修订

R1 在构造 M3–M4 trajectory 记录时因 Python 字典中误用 JSON 布尔常量 `false` 而终止。R1 根保留 `freeze.json` 和 `failure_receipt.json`，未签发分析 PASS、统计表、图或科学结论。

R2 为 fresh root，继承 R1 的全部科学范围、数据绑定、统计规则和 CPU-only 边界。唯一允许的实现修复为 `false → False`，并新增 AST 测试拒绝 Python 表达式中的 `false/true` 名称。R2 不利用 R1 中间结果，不修改原 M4 terminal，也不新增检验或改变阈值。
