# M1 论文验证器与诊断补充说明

依据《阶段M_endpointnnsvit.md》§9和§13，当前主稿将旧“整体支持”改为“广泛但非普遍”。
因此同步修改当前paper AGENTS与draft validator：不再要求0.000435差距或旧综合标签，
改为检查11/12、omnibus NOT_SUPPORTED、Table 6 coverage与Table 7 M1数字。原E6结果、
封存release和旧R1 QC报告不改；新QC写入独立m1文件。此修改不放宽科学门槛。

M1新增verifier在正式数据完成后做来源、逐集合、指标与推断复算。发现验收器重复执行
原先会因审计文件已存在而报错，因此调整为“存在则严格比较，不存在才写入”，不覆盖
PASS记录，也不改变数值接受条件。

依§10.3，观察到residual identity的Endpoint排序优于Taylor后，补充实现诊断：固定两模型
evaluation order 0和blocks 0/5/11，48个float64 reference-displacement局部方向检查，
多epsilon中心差分以及skip手写反事实前向。诊断明确为post-hoc，不改变cohort、reference、
baseline、metric、主统计量或多重检验集合，不作为新增utility证据。
