# N2-V2 双人独立盲审交接

只打开 `cards/` 中的随机编号图像，不打开 `private/`、源码、候选 registry 或另一位审查者的文件。两位审查者互不讨论，分别复制并填写自己的 template。每张图必须填写语义标签和四个布尔 artifact 字段；不得留 `null` 或占位字符串。

语义标签只能是 `AQUATIC_CONTEXT`、`NON_AQUATIC_CONTEXT`、`AMBIGUOUS`。只要存在显著非背景物体、编辑/填充痕迹、异常合成边界或其他可识别 arm 线索，对应字段设为 `true`。本包中的场景图未做 inpainting 或局部修补；人工仍须逐项检查，任一审查者的任一 artifact=true 将 union-reject。完成后不要互看，分别把文件交还执行者封存。
