# NNS-ViT Endpoint Scanner：ENDPOINT-E6 完成审计

日期：2026-09-18。审计对象是用户要求执行的统一研究计划 `ENDPOINT-E6`。权威实现
根为 R3；R1 和 R2 作为不可变恢复证据保留。

## 1. 目标拆解

完成标准为：前置 E4-R3/E5 PASS 可复核；只使用封存 outcome；以图像为统计单位；
完整计算冻结的 16 family、28-test Holm、10,000 次 bootstrap 与确定性四态决策；分别
发布 Identity、Correctness、Utility；保留负结果和 Path 失败边界；生成论文正文/附录、
结果图与可复现审计包；最后提交推送并确认远端包含。

## 2. Prompt-to-artifact checklist

| 要求 | 证据 | 结果 |
|---|---|---|
| 用户显式授权 E6 | `endpoint_e6_release_run_spec.json` authorization token 与本轮请求 | PASS |
| E4-R3 前置有效 | 本轮重新运行 `validate_endpoint_e4_recovery_r3.py`，800-task union、6,144 JVP checks PASS | PASS |
| E5 前置有效 | 本轮重新运行 `validate_endpoint_e5_spatial.py`，800/800 task、115,200 curves PASS | PASS |
| 发布后才读正式 outcome | E6 源码先发布；兼容修复和 R2/R3 合同分别在后续已发布 commit 上执行 | PASS |
| CPU-only，无 JPEG/模型/GPU | `source_read_audit.json`：CUDA 空字符串、forbidden imports 空、image/model read 均 0；`validation_report.json`：GPU-hours 0 | PASS |
| 精确 source read whitelist | 1,632 个唯一读路径等于 run spec、前置 JSON/计划、E4 的 800 JSON 与 E5 的 800 JSON 之精确并集；逐文件 SHA-256 复核 | PASS |
| 统计单位为图像 | `per_image_statistics.json`：1,600 个 image-model-reference 行；head/block/repeat 未作独立样本 | PASS |
| 16 个 family | `family_decisions.json` 精确覆盖 C4a 12 + C4b 4 | PASS |
| 28 个冻结检验 | `sign_test_report.json` 与 `holm_report.json` 精确覆盖 EP0 test ID，未增删 | PASS |
| 10,000 次 paired bootstrap | `bootstrap_report.json` 逐 family 记录 seed、10,000 次、linear 95% percentile CI | PASS |
| exact sign test 与统一 Holm | 28 项按 zero threshold 1e-12 计算；一个全局 Holm family、alpha 0.05 | PASS |
| 360 有效图与 300 非零图门 | 所有 16 family `n_valid=400`，所有 28 tests `n_nonzero=400` | PASS |
| 四态确定性判定 | 11 supported、5 not supported、0 invalid/inconclusive | PASS |
| C4a 联合判定 | 11/12 支持；ViT-B normalized-zero routing/value 为 0.099565 < 0.10，aggregate NOT_SUPPORTED | PASS |
| RQ3 论文解释 | 整体支持，保留一个注册边界例外；不改写 12/12 omnibus C4a 机器字段 | PASS |
| C4b 联合判定 | 0/4 family 同时通过四基线；aggregate NOT_SUPPORTED | PASS |
| Identity/Correctness/Utility 分离 | `claim_decisions.json` 与双语 release 分栏报告 | PASS |
| 负结果和历史失败保留 | R1 failure、R2 非权威 release、E4 R1/R2 FAIL、EP2-R1 Path failure 均保留 | PASS |
| 论文发布 | 双语 R3 release、evidence matrix、defensive-writing audit 存在且哈希绑定 | PASS |
| 图表 | C4a SVG 12 行、C4b SVG 16 行，均由冻结统计生成并哈希绑定 | PASS |
| 独立复算 | validator 不复用生产统计函数，重算逐图值、bootstrap、sign、Holm、决策并 PASS | PASS |
| 原子封存 | 无 pending/failure；terminal、validation、manifest digest 一致 | PASS |
| 主轨终止 | terminal `next_authorization=NONE_ENDPOINT_PRIMARY_TRACK_COMPLETE` | PASS |
| 完整论文图包 | 权威 R2 图包 28 张、84 个三格式导出，覆盖方法、RQ1–RQ4、结论与边界 | PASS |

## 3. 恢复链审计

- **R1**：读取 outcome 后、推断前触发 `NameError`；只有不可变 failure receipt。
- **R2**：统计完整，R2/R3 的逐图统计与 family decisions 逐字节相同；论文 Markdown
  使用了字面段落转义，因此不作为最终 release。
- **R3**：在新根完整重算；双语 Markdown 使用真实换行；独立 source-recomputing
  validator 完整 PASS。

没有要求缺失、弱验证或仅由代理信号覆盖的项目。E6-R3 满足统一研究计划的完成条件。
