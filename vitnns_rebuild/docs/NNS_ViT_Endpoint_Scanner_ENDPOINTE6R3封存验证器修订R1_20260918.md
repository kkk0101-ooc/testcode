# NNS-ViT Endpoint Scanner：ENDPOINT-E6 R3 封存验证器修订 R1

日期：2026-09-18。R3 独立验证已完整通过后，统一计划与导航按 R3 terminal 更新为
主轨完成状态。这使当前工作树中的统一计划字节不同于 R3 运行时读取并记录的 producer
commit blob。生产统计、结果根、论文 release 和 terminal 均未修改。

封存验证器据此对 `unified_plan_sha256` 及 source-read audit 中的统一计划记录使用
R3 producer commit 的 Git blob 重算字节数与 SHA-256；其余读取文件继续与当前不可变
文件逐项比对。这样既不把完成后的导航更新伪装成运行时输入，也不放松 1,632 路径的
精确读白名单。本修订不改变统计或科学结论。
