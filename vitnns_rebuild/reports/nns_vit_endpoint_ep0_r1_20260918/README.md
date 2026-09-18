# Endpoint Scanner EP0：封存报告

状态：**EP0_PROTOCOL_FROZEN**。科学实验未运行，生产未授权，EP1尚未开始。

- [协议说明](../../docs/NNS_ViT_Endpoint_Scanner_EP0协议冻结_20260918.md)
- [冻结包](../../manifests/nns_vit_endpoint/ep0_r1_20260918/README.md)
- [验证报告](validation_report.json)：17项冻结映射、528个cohort ID、2个模型文件hash、16个突变反例。
- [artifact manifest](artifact_manifest.json)：冻结包、脚本及说明文档的逐文件hash。
- [terminal receipt](terminal_receipt.json)：协议状态、验证digest、下一步权限边界。

下一可请求阶段为 EP1_IMPLEMENTATION_AND_REGRESSION。后续阶段需新的明确用户授权；EP0 PASS不是outcome access授权。所有记录按各自文件SHA与canonical digest核对，既有R1数学图包保持原版本。
