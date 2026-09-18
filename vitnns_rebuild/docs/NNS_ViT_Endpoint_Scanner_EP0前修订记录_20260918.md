# NNS-ViT Endpoint Scanner：EP0 前修订与来源验证

日期：2026-09-18。对象：[论文规划 R2](NNS_ViT_Endpoint_Scanner论文思路与规划与步骤_20260918.md)。本次完成规划修订，未执行 EP0 协议冻结、EP1 实现、功效模拟或正式实验。

```text
PLAN_READY_FOR_EP0_WITH_REQUIRED_CLARIFICATIONS
PRODUCTION_EXPERIMENTS_NOT_YET_AUTHORIZED
```

## 1. 审阅意见落实表

| 审阅项 | R2 对应位置 | 已落实的约束 |
|---|---|---|
| 跨图类别语义 | 3.2 | 主 ground-truth logit、保留错分图；次级 clean top-1 冻结；明确两者均为标签/样本依赖的目标规则 |
| 局部合同与效用 | 1、5D、8 | Identity / Correctness / Utility 分层；EP5-A 只支持 correctness；EP5-B 是有限行为评价 |
| C4 预测量 | 5D、6 | C4a 主 $m_k$ 对 $\lvert e_k\rvert$、signed 次级；C4b 检验空间增量，登记匹配的 Taylor 基线 |
| 输入分辨图的价值 | 5D | 区域选择、固定预算、全图效果与同切面结构隔离效果分别定义；C4b 启用/统计角色须 EP0 决定 |
| Mass 尺度 | 3.7 | 原始 normalized-input RGB 坐标、降采样前 absolute；signed sum 同存；空间/显示派生量分列 |
| Attention backend | 3.4、8.2 | canonical unfused、dropout/drop path 关闭、精确 hook edges、fused/no-op parity；模型支持范围限定 |
| 闭合重构对象 | 3.5 | Head-only、QKV、routing/value 闭合到 $A_F$；加 bypass 后在完整路径条件下闭合到 $L_F$ |
| Atlas 选择偏差 | 5B、8.1 | 固定目标/样本/色标/near-zero；提供全索引、分布、固定样本与失败案例 |
| 扫描规模与成本 | 5B–5C、7.2 | registered scanning；K/M/C 分解；targets/sec、images/sec、peak memory、bytes/target 与 P95 |
| Insertion/deletion | 5C | 主 absolute 排序、zero/blur 两 replacement、RGB 聚合顺序、面积 AUC、分母近零和 OOD 参数登记 |
| Flip 语义 | 5C | 主类别目标，比较 $F_n(x)$ 与 $F_n(Tx)$；响应差、预测变化、完整 cohort/预定子集；往返检查单列 |
| 统计可执行性 | 6.1–6.2 | 数量、family、主统计量、效应阈值、CI/bootstrap、检验/Holm、有效样本/宽度、NA 与有序出口 |
| Novelty | 9 | 六类 nearest-work 审计维度；若已有工作覆盖分解，则改为复现/统一实现/实证审计叙事 |
| 最小多卡系统 | 7 | manifest、一卡一 worker、原子 claim、immutable result、覆盖、恢复、stable seed；正确性先于吞吐 |
| 资源 hard cap | 7.2 | wall/GPU-hours/task/storage cap；停止派发/保存；资源暂停独立于科学结论，不自动扩额 |
| EP0–EP1 清单 | 8.1–8.2 | 17 项冻结工作、15 项实现验收；包含非零 omission、residual false-pass 和过期 attempt 拒绝 |

具体 N、最小效应、有效图像门槛、资源 cap、模型 digest、检验 family 清单等仍标 `TBD_EP0`。建议值（如 95% CI、10,000 次 bootstrap）不表示已冻结或执行。EP0 未完成时不签发正式数据访问或 GPU 生产授权。

## 2. 数学说明中的两项精确化

对于固定最终目标 $F$，attention 分支成分 $A_F$ 使用真实下游 cotangent；它不同于另选内部 head 标量的 learning image。图上的分支闭合、全目标闭合必须使用各自 registry 的 reconstruction ID。

水平翻转若精确可逆，$T^{-1}T(x)=x$；对应 logit 差是变换实现检查。模型响应变化以 $F_n(x)-F_n(Tx)$ 衡量，图相似度按 inverse-warp 对齐后再算。

## 3. R1 图包保持原始版本

原图包：[数学对比图 R1](../reports/nns_vit_endpoint_math_comparison_r1_20260918/README.md)。发布 commit：`f78c73eee4455e42ea0c8f3f4f786c6746878e7d`。其 manifest 绑定的规划 R1 SHA-256：

```text
9526218694441b3c56444638c5fc85c03802a9622639f1871c0de784dd5c090b
```

本轮复核结果：

- R1 manifest 内 **34 个 artifact** 的大小和 SHA-256 全部匹配。
- R1 根目录 **35 个文件**（含 manifest）逐字节等于发布 commit。
- **9 个 source** 的发布 commit 字节全部匹配原 source hashes。
- 旧 Path 图目录的完整文件清单及 SHA-256 与 predecessor inventory 一致。
- 当前 source 与 R1 的唯一差异是论文规划 R2；图像、数据、渲染脚本、数学分析和原论文文本未改写。

因此旧 `--verify` 对当前工作区运行时会正确识别规划 source drift；不能把更新后的规划 hash 填回原 manifest。历史图包应与其发布 commit 的来源复核。下方命令只读取 Git 对象和相关文件，适用于本次 R2 修订状态；以后若其他 source 合法改版，应另行登记漂移。

在 `testcode` 根目录运行：

```bash
python3 - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

base = Path.cwd()
repo = base / 'vitnns_rebuild'
root = repo / 'reports/nns_vit_endpoint_math_comparison_r1_20260918'
revision = 'f78c73eee4455e42ea0c8f3f4f786c6746878e7d'
manifest = json.loads((root / 'artifact_manifest.json').read_text())
digest = lambda value: hashlib.sha256(value).hexdigest()
def at_revision(path):
    relative = path.resolve().relative_to(base).as_posix()
    return subprocess.check_output(['git', 'show', revision + ':' + relative])

for item in manifest['artifacts']:
    data = (root / item['path']).read_bytes()
    assert digest(data) == item['sha256'] and len(data) == item['bytes']
for path in root.rglob('*'):
    if path.is_file():
        assert path.read_bytes() == at_revision(path)
drift = []
for item in manifest['sources']:
    path = repo / item['path']
    original = at_revision(path)
    assert digest(original) == item['sha256'] and len(original) == item['bytes']
    if digest(path.read_bytes()) != item['sha256']:
        drift.append(item['path'])
assert drift == ['docs/NNS_ViT_Endpoint_Scanner论文思路与规划与步骤_20260918.md']
old = repo / 'reports/nns_vit_forward_learning_image_comparison_20260917'
inventory = {p.relative_to(old).as_posix(): digest(p.read_bytes())
             for p in sorted(old.rglob('*')) if p.is_file()}
assert inventory == manifest['predecessor_inventory']
print('PASS_HISTORICAL_PACKAGE; documented plan-source revision only')
PY
```

## 4. 本轮文档验证

规划 R2 的 Markdown 表格列数、代码/数学块配对、控制字符、相对文件链接和 `git diff --check` 已核对。修订只涉及规划、[写作审计](NNS_ViT_Endpoint_Scanner论文规划写作审计_20260918.md)、本记录和阶段导航；未改实现或历史实验结论。
