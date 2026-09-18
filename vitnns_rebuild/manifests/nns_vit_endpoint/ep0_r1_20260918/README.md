# Endpoint Scanner EP0 冻结包

[协议说明](../../../docs/NNS_ViT_Endpoint_Scanner_EP0协议冻结_20260918.md) · [验证报告](../../../reports/nns_vit_endpoint_ep0_r1_20260918/validation_report.json)

| 文件 | 冻结内容 |
|---|---|
| [protocol.json](protocol.json) | 方法定位、状态和授权边界 |
| [cohort_registry.json](cohort_registry.json) | 64/64/400图像ID、标签序、排除台账与哈希选择 |
| [model_registry.json](model_registry.json) | 两个模型的真实文件SHA、结构、preprocessing |
| [target_registry.json](target_registry.json) | 457/385目标、8种cut合同、mass尺度 |
| [evaluation_registry.json](evaluation_registry.json) | IG校准、输入fidelity、稳定性、atlas、时间测量 |
| [intervention_registry.json](intervention_registry.json) | EP5-A/B、C4a、C4b、有限replacement与空间预算 |
| [statistical_analysis_plan.json](statistical_analysis_plan.json) | 16族28检验、阈值、CI/Holm/NA和科学出口 |
| [resource_contract.json](resource_contract.json) | 多卡动态队列、hard caps、恢复与retention |
| [implementation_contract.json](implementation_contract.json) | EP1容差、反例、backend与EP2可调范围 |
| [access_contract.json](access_contract.json) | 当前EP0读权限、后续阶段授权门 |
| [literature_registry.json](literature_registry.json) | 12条可追溯来源/摘录与定位决定 |
| [source_bindings.json](source_bindings.json) | 本地文件哈希和本轮访问事实 |
| [environment_contract.json](environment_contract.json) | 元数据核对的Python与包版本 |
| [ep0_checklist.json](ep0_checklist.json) | 17项规划冻结要求的文件/字段映射 |
| [protocol_bundle.schema.json](protocol_bundle.schema.json) | 关闭所有对象且绑定冻结值的JSON Schema2020-12 |

从 testcode 根目录复核：

```bash
CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python vitnns_rebuild/scripts/validate_endpoint_ep0.py
```

本包是EP0协议，不是EP1测试结果。正式图像尚未解码；后续模型执行和evaluation访问都需要新的明确授权。未来修改冻结参数使用前瞻性修订与新根，保留本包。
