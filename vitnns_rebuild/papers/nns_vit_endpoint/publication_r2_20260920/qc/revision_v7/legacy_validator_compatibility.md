# v3–v6 历史验证器兼容性记录

规程要求实际运行旧验证器，不允许照抄入口宣称通过。本轮执行结果如下。

| Command | Exit | 结果 |
| --- | ---: | --- |
| validate_gxi_step3.py | 1 | FAIL：要求旧 pooled A/V Table 6 行、逐字双语表行和旧 active mirror |
| validate_gxi_step4.py | 1 | FAIL：另要求旧 Table 6、旧近零值格式，并把 v7/S18 标识当匿名违规 |
| validate_publication_r2.py | 1 | FAIL：聚合上述旧断言，并要求旧 inventory hash |
| validate_gxi_step5.py | 1 | FAIL：聚合上述旧断言和旧 entry mirror |

这些失败是版本断言失配，不是当前稿的科学或发布失败：v7 明确要求用 A/V 与 residual 分层结果替换主文 pooled Table 6、修复极小值格式、增加 S18，并生成新的 LaTeX/PDF。旧验证器会在运行时改写 v6 QC 收据；本轮已把四个 v6 历史 QC 文件恢复为 HEAD 原字节状态，保留历史不可变性。

现役 v7 验证覆盖旧验证器失配的实际目的：

- 逐值验证新 Table 6 与 v7 CSV；
- 在补充 S2.1 保留原 12 个 pooled family 与 11/12 provenance；
- 从原始 E5 JSON/NPZ 独立复算 H1–H4；
- 检查双语 Markdown/LaTeX/PDF 的公式、数字、图表、引用、阴性结论和匿名性；
- 检查所有图片和 bibliography 可解析；
- 连续两次干净 XeLaTeX-compatible 构建。

测试层只对 tests/test_revision_v6.py 做最小版本兼容修改：跳过三项已被 v7 明确取代的布局断言；v6 的 cut/reconstruction 合同及错误实现负例保持强制检查。最终联合测试为 18 passed。
