# N2-V2 双人独立盲审交接

审查者 A 只打开 `reviewer_A.html`，审查者 B 只打开 `reviewer_B.html`。不要查看 `private/`、来源信息、另一位审查者的文件或模型信息；完成前不要讨论。浏览器会本地保存进度，102 张全部填写后点击导出 JSON。将两个导出的 JSON 分别交回，不要互看。

语义标签只能是 `AQUATIC_CONTEXT`、`NON_AQUATIC_CONTEXT`、`AMBIGUOUS`。四个 artifact 字段逐项判断。任一审查者任一 artifact=true 将 union-reject。卡片为完整画面等比缩放并以灰色留边，没有裁切、内容填充或局部编辑。
