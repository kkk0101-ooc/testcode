# Academic defensive-writing audit — revision v1

Skill: /data01/home/yeziqi.831/.trae/skills/academic-defensive-writing-auditor/SKILL.md. Scope: current Chinese and English manuscripts; full-paper cleanup. Scientific boundaries were evaluated separately from defensive wording.

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Abstract | Long sequence of 800/800, 336,800, 76,800, 28,800 and 6,144 counts before the mechanism finding | D9/D10 | Major | MIXED | Audit volume competes with the scientific result | Main finding is hard to identify | Keep one fidelity count; move detail to results/supplement | Lead with local Jacobian identity, spatial broadening, Value-only discrepancy and behavioral calibration |
| Introduction | 但不把 x⊙∇s 重新命名为新的基础归因公式 | D14 | Minor | MIXED | Anticipates a novelty objection | Technical addition arrives late | State mathematical identity directly | 基础 Endpoint 与同目标 Gradient × Input 数值相同。 / The base Endpoint is numerically identical to Gradient × Input for the same target. |
| Original 5.7 | 不要求 map 达到均匀全图状态，也不等价于获得全局语义 | D2/D12 | Minor | MIXED | Two exclusions follow a clear spatial result | The result seems more uncertain than its statistics | Define the measured object; retain semantic boundary in limitations | 这描述局部输入依赖的空间扩展。 / These measurements describe spatial broadening of local input dependence. |
| Original 5.4 | 不能确认达到预设实际效应水平 | Statistical precision | Major | NECESSARY_CAVEAT | The original line singles out only the failed point estimate | Reader may assume all supported CIs exceed 0.10 | Clarify point threshold vs sign test and show all lower CI bounds | 0.10/0.05 是点估计实际效应门槛；符号检验的原假设针对零 |
| Original 6.1 | 把它们分开可以避免两种常见推断 | D1/D12 | Minor | MIXED | Explains how not to misunderstand the paper | Reads as anticipated rebuttal | State the evaluated properties | 局部 Jacobian 身份、实现正确性和有限干预效用是三个可分别检验的性质 |
| Original limitations | PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE，而非无条件 PASS | D9 | Major | NECESSARY_CAVEAT | Internal state strings interrupt scientific interpretation | Engineering process displaces the numerical fact | Preserve failure count in results; retain exact status in supplement | 36 checks miss float32 tolerance; same-coordinate float64 multi-step reviews pass |
| Original conclusion | Long result inventory with repeated NOT_SUPPORTED fields | D2/D9 | Major | MIXED | Restates release bookkeeping | Conclusion obscures the scientific takeaway | Four connected findings; preserve statuses in RQ5 and supplement | Identity; valid structural cuts; depth patterns; task-specific behavioral boundary |
| Revised limitations | Random-initialization/untrained controls are absent | D5 check | None | NECESSARY_CAVEAT | This changes whether a trend can be attributed to training | Appropriate scope statement | Retain once in limitations | Architecture, initialization and training contributions remain unresolved |
| Revised figures | Signed RGB sum can cancel channels while RGB absolute mass preserves their magnitudes | Evidence condition | None | NECESSARY_CAVEAT | Removing it would equate two different displays | Clarifies the measured quantity | Keep in method and display protocol | Signed=sum(E_c); absolute=sum(abs(E_c)); raw RGB norms accompany maps |

Paper-level defensive-writing score: **2/10** after revision (editorial judgment, not a validated metric). The manuscript states claims first and retains technical boundaries where they change interpretation.

Highest-priority edits completed:

1. Focused the abstract on identity, structure, mechanism observations and behavior.
2. Replaced novelty rebuttal wording with the established G×I identity and concrete additions.
3. Moved release strings, result roots, all-head pages and full diagnostic tables to supplementary material.
4. Removed repeated local/global and causal exclusions around each result; kept the governing interpretation in methods and limitations.
5. Rewrote the conclusion as four scientific findings.
6. Corrected threshold wording for both supported and unsupported families.

Necessary caveats preserved: sampled registry coverage; coordinate origin; lack of function-value completeness; correct cut/cotangent conditions; path scale versus causal percentage; float32 failure retention; original omnibus/C4b decisions; candidate-only oracle; post-hoc label; missing training/target/matched-target controls.

Repeated-phrase audit across both complete current manuscripts: “we do not claim”, “we emphasize”, “it should be noted”, “to avoid misunderstanding”, and “taken together” each occur **0 times**. Necessary negative constructions remain in definitions and limitations; they were not removed mechanically. The Chinese and English displayed equations, table rows, section order and figure order are checked together by the revision validator.
