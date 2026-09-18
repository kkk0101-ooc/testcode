# M1 论文表述审计

范围：中英文主稿 Abstract、RQ3 Results、Discussion、Conclusion，新 M1 小节和 registry 覆盖表。使用 academic-defensive-writing-auditor；原封存 release/R4 历史文件保持历史身份，当前入口为 full_draft。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Abstract / RQ3 | supported overall with one registered boundary exception | D4 | Major | MIXED | 用综合标签淡化预注册门未过 | 结论与门禁不一致 | 改为实测模式 | broad but non-universal family-level support; omnibus NOT_SUPPORTED |
| RQ3 boundary | gap was 0.000435 | D4 | Major | DEFENSIVE | 强调差距小容易构成结果辩护 | 像事后解释阈值 | 改报CI与门槛关系 | CI spans 0.10; attainment remains uncertain |
| Discussion | Reducing the entire RQ3 result to a failure would discard… | D1/D4 | Major | DEFENSIVE | 回应想象中的批评 | 讨论像答辩 | 删除预设反驳 | 11/12 meet conditions; broad but non-universal association |
| Coverage | complete target coverage | D10/D12 | Major | MIXED | 未说明候选总体容易放大覆盖 | 是否扫描全部单元不清楚 | 给出分母和registry范围 | registry-complete; post-block 0.01058%/0.02115% |
| M1 identity result | Endpoint-Taylor rho 0.0865–0.4209; Capture@1 difference zero | CLEAN | Minor | NECESSARY_CAVEAT | 限定是解释所必需 | 排序和决策收益区分清晰 | 保留 | Incremental utility depends on component and metric |

## Paper-level defensive-writing score

1/10（本轮修改段落）。保留具体证据、置信区间、确认性与补充分析边界；删除为小差距辩护的表述。

## Highest-priority removals

1. 摘要和结论的“整体支持”。
2. 0.000435 差距辩护。
3. 对“把RQ3归为失败”的预设反驳。
4. 没有候选分母的complete target coverage。

## Necessary caveats to preserve

- 11/12 family与all-family omnibus NOT_SUPPORTED分别报告。
- M1为原研究之后冻结的supplementary study。
- value/V同一edge，pooled同图像合并且仅描述。
- QKV三项和residual两项的Spearman未定义。
- Endpoint identity排序优势不自动构成Top-1优势。
- Registry-complete不等于全部神经元；结构分解为局部Jacobian。

## Repeated phrase audit

修订后两份主稿中“supported overall”“整体支持”“0.000435”计数均为0。
“广泛但非普遍”/“broad but non-universal”保留于摘要、结果、讨论与结论以同步科学结论；
没有重复增加非主张清单。
