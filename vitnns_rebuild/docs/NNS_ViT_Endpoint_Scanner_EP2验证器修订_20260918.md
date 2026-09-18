# Endpoint Scanner EP2 验证器修订

日期：2026-09-18。EP2 R1 已完成全部 128 个 task 并签发
`EP2_NUMERICAL_CALIBRATION_FAILED` 后，首次独立验证发现 task manifest digest
的重算顺序错误：队列实现 `create_queue()` 按 `task_id` 排序后计算 digest，
外层 `task_manifest.json` 则保留 `calibration split_order × model registry` 的生成顺序。
验证器错误地直接对外层顺序重算，得到 `TASK_DIGEST` false rejection。

本修订只让验证器复现已发布 producer commit 中队列实现的既定 canonicalization：
先按 `task_id` 排序，再重算 digest。它不修改 R1 root、task、图像、模型、IG map、
M 候选、阈值、61/64 门、科学边界或阶段出口。验证器继续使用 producer commit
中的源 blob 哈希验证所有执行代码，并从 128 个不可变 task result 重新计算模型
选择与 `EP2_NUMERICAL_CALIBRATION_FAILED`。此外，验证器从官方 tar 重读且哈希
恰好 64 个 calibration 成员，使复核不依赖被 `.gitignore` 排除的临时 cache；
不解码或读取 discovery/evaluation 成员。
