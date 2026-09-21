#!/usr/bin/env python3
"""Place path-map evidence before the auxiliary Taylor analysis in Chapter 7."""
import re
from pathlib import Path

P=Path(__file__).resolve().parents[1]

def section(text,start,end):
    return text.split(start,1)[1].split(end,1)[0]

for lang in ('zh','en'):
    p=P/lang/'manuscript.md';text=p.read_text()
    if 'REVISION_V5_CHAPTER7_REORDERED' in text:continue
    if lang=='zh':
        chapter='# 第 7 章 神经元与结构路径的行为评价'
        next_chapter='# 第 8 章 外部验证、讨论与结论'
        heads=['## 7.1 本章问题','## 7.2 方法','## 7.3 辅助诊断：参照位移感知泰勒量的多幅度近似','## 7.4 结果：完整成分排序与原联合判据','## 7.5 结果：区域选择','## 7.6 前瞻实验 E3：神经元替换与目标选择性','## 7.7 讨论','## 7.8 本章小结']
        newheads=['## 7.1 本章问题','## 7.2 方法','## 7.6 辅助诊断：参照位移感知泰勒量的多幅度近似','## 7.3 结果：完整成分排序与原联合判据','## 7.4 结果：区域选择','## 7.5 前瞻实验 E3：神经元替换与目标选择性','## 7.7 讨论','## 7.8 本章小结']
        prefix='表';fig='图'
    else:
        chapter='# Chapter 7 Behavioral Evaluation of Neurons and Structural Paths'
        next_chapter='# Chapter 8 External Validation, Discussion, and Conclusions'
        heads=['## 7.1 Chapter Question','## 7.2 Method','## 7.3 Auxiliary Diagnosis: Reference-Displacement Taylor across Amplitudes','## 7.4 Results: Full-Component Ranking and the Original Joint Criterion','## 7.5 Results: Regional Selection','## 7.6 Prospective E3: Neuron Replacement and Target Selectivity','## 7.7 Discussion','## 7.8 Chapter Summary']
        newheads=['## 7.1 Chapter Question','## 7.2 Method','## 7.6 Auxiliary Diagnosis: Reference-Displacement Taylor across Amplitudes','## 7.3 Results: Full-Component Ranking and the Original Joint Criterion','## 7.4 Results: Regional Selection','## 7.5 Prospective E3: Neuron Replacement and Target Selectivity','## 7.7 Discussion','## 7.8 Chapter Summary']
        prefix='Table';fig='Figure'
    body=section(text,chapter,next_chapter)
    pieces={}
    for i,h in enumerate(heads):
        end=heads[i+1] if i+1<len(heads) else None
        value=body.split(h,1)[1]
        if end:value=value.split(end,1)[0]
        pieces[i]=newheads[i]+value
    # Question, method, component, region, neuron, Taylor, discussion, summary.
    newbody='\n\n'.join(pieces[i].strip() for i in (0,1,3,4,5,2,6,7))+'\n\n'
    text=text.split(chapter,1)[0]+chapter+'\n\n'+newbody+next_chapter+text.split(next_chapter,1)[1]
    mapping={1:7,2:8,3:1,4:2,5:3,6:4,7:5,8:6}
    for old in mapping:text=text.replace(f'{prefix} 7-{old}',f'{prefix} TMP-{old}')
    for old,new in mapping.items():text=text.replace(f'{prefix} TMP-{old}',f'{prefix} 7-{new}')
    text=text.replace(f'{fig} 7-1',f'{fig} TMP-1').replace(f'{fig} 7-2',f'{fig} 7-1').replace(f'{fig} TMP-1',f'{fig} 7-2')
    if lang=='zh':
        text=text.replace('本章回答 RQ4 的行为评价部分：一阶局部量对注意力头输出插值的误差怎样随干预幅度变化；在完整成分替换和固定面积区域替换中，原输入局部图与参照状态感知的泰勒估计分别具有怎样的排序能力。','本章回答 RQ4 的行为评价部分：先检验路径图在完整成分、区域和神经元替换中的排序表现，再以参照位移感知泰勒量分析局部线性化的适用幅度。')
    else:
        text=text.replace('This chapter addresses the behavioral part of RQ4: how first-order local quantities behave as the amplitude of head-output interpolation changes, and how clean-input local maps and reference-aware Taylor estimates rank full component and fixed-area regional replacements.','This chapter addresses the behavioral part of RQ4. It first evaluates path-map ranking for complete-component, regional, and neuron replacement, then uses reference-displacement Taylor as an auxiliary analysis of local linearization across amplitudes.')
    p.write_text(text+'\n<!-- REVISION_V5_CHAPTER7_REORDERED -->\n')
print('CHAPTER7_REORDERED')
