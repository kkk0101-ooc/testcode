#!/usr/bin/env python3
"""Bilingual corrections whose numbers are already verified in sealed evidence."""
from pathlib import Path
import csv,re
P=Path(__file__).resolve().parents[1];R=P.parents[2];D=P/'evidence/revision_v5_existing';PUB=R/'papers/nns_vit_endpoint/publication_r2_20260920'
def rows(p):return list(csv.DictReader(p.open()))
def ci(r,n=4):return f"{float(r['median']):.{n}f} [{float(r['ci95_low']):.{n}f}, {float(r['ci95_high']):.{n}f}]"
def table(headers,body):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in body)
def insert(s,num,z,e,lang):
 text=z if lang=='zh' else e
 return re.sub(r'(^## '+re.escape(num)+r' [^\n]*\n)',lambda m:m[0]+'\n'+text+'\n\n',s,count=1,flags=re.M)
spatial=r'''$$
H(q)=-\frac{\sum_{j=1}^{196}q_j\log q_j}{\log196},\qquad
L_t=\sum_{j:\|r_j-r_t\|_\infty\le1}q_j-\frac{|\{j:\|r_j-r_t\|_\infty\le1\}|}{196},\qquad
D_t=\sum_jq_j\frac{\|r_j-r_t\|_2}{13\sqrt2}.
$$

$$
S_Q=\frac{\|E^Q\|_1}{\|E^Q\|_1+\|E^K\|_1+\|E^V\|_1},\quad S_K,S_V\text{ analogously}.
$$'''
invariance=r'''$$
z'=Rz,\quad g_{z'}=R^{-\mathsf T}g_z,\quad
J_{z',x}^{\mathsf T}g_{z'}=J_{z,x}^{\mathsf T}R^{\mathsf T}R^{-\mathsf T}g_z=J_{z,x}^{\mathsf T}g_z.
$$'''
score=r'''$$
m_k=\sum_i|E_{k,i}|,\qquad b_k=|\langle g_k,z_k(x)\rangle|,\qquad
y_k=|F(x)-F(x;z_k\leftarrow z_k^{\rm ref})|.
$$

$$
\operatorname{capture@}K=\frac{\sum_{j=1}^{K}y_{\pi(j)}}{\sum_{j=1}^{K}y_{(j)}},\qquad
\operatorname{regret@}1=\max_ky_k-y_{\pi(1)}.
$$'''
regional=r'''$$
x^{(R)}=(1-M_R)\odot x+M_R\odot x^{\rm ref},\qquad
e_{h,R}=F(x)-F\bigl(x;z_h\leftarrow z_h(x^{(R)})\bigr),
$$

$$
m_{h,R}=\sum_{i\in R}|E_{h,i}(x)|,\qquad
t_{h,R}=\left|\left\langle g_h,z_h(x)-z_h(x^{(R)})\right\rangle\right|.
$$'''
aggregation=r'''$$
\left\|\sum_hE_h\right\|_1,\qquad\sum_h\|E_h\|_1,\qquad\operatorname{median}_h\|E_h\|_1.
$$'''
for lang in ['zh','en']:
 p=P/lang/'manuscript.md';s=p.read_text();zh=lang=='zh'
 if 'V5_FOUNDATIONS_APPLIED' in s:continue
 word='表' if zh else 'Table'
 for a,b in [('3-1','3-2'),('6-3','6-4')]:s=s.replace(word+' '+a,word+' TMP').replace(word+' '+b,word+' '+a).replace(word+' TMP',word+' '+b)
 s=insert(s,'4.7','令 $r_j$ 为 14×14 图像块网格中心，$t$ 为目标空间标记。\n\n'+spatial+'\n\n局部性扣除邻域均匀质量；CLS 的局部性和距离为 NA，零总质量的归一化指标为 NA。Q/K/V 分母为各项绝对质量之和；路由比例先合并 Q、K 的带符号图，二者不同。','Let $r_j$ be patch centers on the 14×14 grid and $t$ the spatial target token.\n\n'+spatial+'\n\nLocality subtracts uniform neighborhood mass. CLS locality/distance and zero-mass normalized metrics are NA. Q/K/V shares use the sum of separate absolute masses; routing instead first combines signed Q and K maps.',lang)
 s=insert(s,'4.6','完整向量边对边内可逆线性坐标变换保持不变：\n\n'+invariance+'\n\n坐标子分项可随混合而改变；跨边混合及切面粒度改变也会改变分组。','A complete vector edge is invariant to invertible linear coordinate changes within the edge:\n\n'+invariance+'\n\nCoordinate terms can change under mixing, as can groupings that mix different edges or change cut granularity.',lang)
 s=insert(s,'5.2',score+'\n\n$\pi$ 按评分递减排列，$y_{(j)}$ 按实测绝对效应递减排列，零分母捕获率为 NA。捕获率不是选对头的概率。原 C4a 每图跨层候选为 144/72 个头、24 个路由/取值成分或 24 个残差分支。M1 与多幅度比较先计算每层跨头的绝对值相关，再平均 12 层，最后跨图取中位数。',score+'\n\n$\pi$ orders evaluated scores and $y_{(j)}$ orders measured magnitudes; zero-denominator capture is NA. Capture is not the probability of selecting the best head. Original C4a pools 144/72 heads, 24 routing/value components, or 24 residual branches within each image. M1 and multi-amplitude comparisons instead rank absolute values within each block, average 12 blocks, and take image medians.',lang)
 s=insert(s,'5.2','区域任务先形成混合输入，再只把它产生的一个头输出插回原输入计算：\n\n'+regional+'\n\n其他头和旁路保持原值，后续计算重新执行。区域泰勒需要每个候选混合输入的前向；不是输入梯度乘像素差，也不能从一次整体参照前向得到所有区域评分。$N_R$ 个候选需 $N_R$ 次候选输入前向和对应隔离后缀，另计算干净梯度。本文不据此主张实测效率优势。每个预算独立取最佳子集时，候选上界曲线可不对应一个嵌套排序。','Regional evaluation first creates a mixed input and inserts its single-head output into the clean computation:\n\n'+regional+'\n\nOther heads and bypasses retain clean values, and downstream computation is repeated. Regional Taylor needs a forward for each mixed-input candidate; it is neither input gradient times pixel displacement nor obtainable for all regions from one full-reference forward. $N_R$ candidates require $N_R$ input forwards and corresponding isolated suffixes, plus the clean gradient. This does not establish a measured efficiency advantage. Budget-specific best subsets need not form a nested ranking.',lang)
 s=insert(s,'6.4',aggregation+'\n\n以上三种汇总通常不同。表 6-2 先合并各头带符号图再取范数；表 6-4 先计算每头范数，再取图内头中位数。M4 路由尺度同样先求每头比例，再取图内中位数，不能读作整层因果份额。跨头抵消和中位数操作使两表量级不同；仅比例下降不能推出路径作用减弱。',aggregation+'\n\nThese aggregations generally differ. Table 6-2 combines signed head maps before taking norms; Table 6-4 takes each head norm and then its within-image median. M4 routing scale likewise takes the median of head-specific ratios and is not a whole-layer causal share. Cancellation and the median operator change scale; a declining ratio alone does not establish a weaker path effect.',lang)
 s=insert(s,'5.1','原权重为 vit_base_patch16_224.augreg2_in21k_ft_in1k（ImageNet-21K 预训练、ImageNet-1K 微调）和 deit_small_patch16_224.fb_in1k（DeiT ImageNet-1K）。RGB 均值/标准差分别为 (0.5,0.5,0.5)/(0.5,0.5,0.5) 与 (0.485,0.456,0.406)/(0.229,0.224,0.225)。Pillow 10.2.0 解码 RGB，不作 EXIF 转置；torchvision 0.17.2 双三次短边缩放至 248、antialias=True、中心裁剪 224。模糊在 [0,1] RGB 上使用 31×31 高斯核、σ=10，再标准化。训练来源不同，模型间差异不单独归因于宽度或头数。\n\n原样本排除 1098 个已暴露文件，990 类仍有合格图像，确定性选择其中 528 类各一图。图像自助区间近似将入选类别/图像视为可交换样本，不估计类内变异。三个初始化状态的区间仅描述这些状态；增加自助次数不会增加独立状态数。','The original weights are vit_base_patch16_224.augreg2_in21k_ft_in1k (ImageNet-21K pretraining, ImageNet-1K fine-tuning) and deit_small_patch16_224.fb_in1k (DeiT ImageNet-1K). RGB means/standard deviations are (0.5,0.5,0.5)/(0.5,0.5,0.5) and (0.485,0.456,0.406)/(0.229,0.224,0.225). Pillow 10.2.0 decodes RGB without EXIF transposition; torchvision 0.17.2 uses bicubic short-side resize to 248, antialias=True, and center crop 224. Blur uses a 31×31 Gaussian kernel with σ=10 on [0,1] RGB before normalization. Training sources differ, preventing attribution of model differences to width or head count alone.\n\nThe original exclusion list contains 1,098 exposed files; 990 classes retain eligible images, of which 528 classes contribute one image each. Image bootstrap approximates exchangeability of selected classes/images and does not estimate within-class variation. Initialization intervals describe the three sampled states; additional bootstrap draws do not add independent states.',lang)
 sums=rows(D/'m1_head_summary.csv');paired=rows(D/'m1_head_paired.csv');body=[]
 for model in ['vit_b16','deit_s16']:
  for ref in ['normalized_zero','blur_rgb']:
   for metric in ['spearman','capture_at_1','regret_at_1']:
    body.append([model,ref,metric]+[ci(next(r for r in sums if r['model']==model and r['reference']==ref and r['metric']==metric and r['predictor']==method)) for method in ['endpoint','activation_gradient','taylor','random']])
 headings=['模型','参照','指标','路径图','激活乘梯度','位移泰勒','随机'] if zh else ['Model','Reference','Metric','Path map','Activation × gradient','Displacement Taylor','Random']
 m1=('**表 7-4 头内比较：图像中位数与 95% 区间（400 图，补充分析）**' if zh else '**Table 7-4 Within-block head comparison: image medians and 95% intervals (400 images, supplementary)**')+'\n\n'+table(headings,body)
 body=[[r['model'],r['reference'],ci(r)] for r in paired if r['metric']=='spearman' and r['baseline']=='activation_gradient']
 m1+='\n'+('**表 7-5 路径图减激活乘梯度的配对 Spearman 差**' if zh else '**Table 7-5 Paired Spearman difference: path map minus activation × gradient**')+'\n\n'+table(['模型','参照','配对差 [95% 区间]'] if zh else ['Model','Reference','Paired difference [95% interval]'],body)
 m1+='\n'+('激活乘梯度不使用参照状态，其排序在三组配对区间中优于路径图；ViT-B 模糊组跨零。路径图弱项不能只归因于缺少参照信息，输入坐标拉回、绝对聚合、尺度与抵消的相对作用尚未分离。' if zh else 'Activation × gradient uses no reference state and exceeds path-map ranking in three paired intervals; the ViT-B blur interval crosses zero. Missing reference information does not fully explain the deficit. Pullback, absolute aggregation, scale, and cancellation remain unseparated factors.')
 s=insert(s,'7.4',m1,m1,lang)
 s=s.replace('## 7.3 结果：多幅度一阶边界','## 7.3 辅助诊断：参照位移感知泰勒量的多幅度近似').replace('## 7.3 Results: Multi-Amplitude First-Order Boundary','## 7.3 Auxiliary Diagnosis: Reference-Displacement Taylor across Amplitudes')
 s=insert(s,'7.3','本节预测量为式（5-2）的参照位移感知泰勒量，计算不需要生成路径图。结果检验沿实际替换方向的局部线性化，Spearman 使用预测和实测效应的绝对值。','The predictor is reference-displacement Taylor in Equation (5-2) and requires no path map. This auxiliary analysis evaluates linearization along the actual replacement direction; Spearman compares absolute predictions and absolute effects.',lang)
 s=s.replace('多幅度注意力头输出插值','参照位移感知泰勒量的多幅度近似').replace('multi-amplitude head-output interpolation','multi-amplitude reference-displacement Taylor approximation')
 amp=rows(PUB/'evidence/revision_v3_step2/multi_amplitude_summary.csv')
 for tid,model in [('7-1','vit_b16'),('7-2','deit_s16')]:
  match=re.search(r'(\*\*(?:表|Table) '+tid+r'[^\n]*\*\*\n\n)((?:\|[^\n]*\n)+)',s)
  if match:
   lines=match[2].splitlines();out=lines[:2]
   for line in lines[2:]:
    cells=[x.strip() for x in line.strip('|').split('|')];ref='normalized_zero' if cells[0] in ['归一化零输入','Normalized zero'] else 'blur_rgb';alpha=float(cells[1]);new=cells[:2]
    for metric in ['median_absolute_finite_effect','median_absolute_taylor_prediction','median_absolute_error','median_relative_error','mean_within_block_spearman']:
     rr=next(r for r in amp if r['model_id']==model and r['reference_id']==ref and abs(float(r['alpha'])-alpha)<1e-7 and r['metric']==metric);new.append(ci(rr,4 if 'relative' in metric or 'spearman' in metric else 6))
    out.append('| '+' | '.join(new)+' |')
   s=s[:match.start(2)]+'\n'.join(out)+'\n'+s[match.end(2):]
 if zh:
  s=s.replace('小幅头输出替换时，一阶数值误差较低、头间排序较稳定；完整替换时数值误差明显增加，排序仍保留部分信息。','辅助诊断中，参照位移感知泰勒量的小幅替换误差较低，完整替换时误差增加；这一预测不需要路径图。')
  s=re.sub(r'绝对曲线下面积相对原目标逻辑值很小：[^\n]+\n','',s)
  s=s.replace('四种模型—参照组合方向一致，降低了结论由某一输入原点或模糊算子单独驱动的可能性。','两种参照下，泰勒近似误差均随替换幅度增加。').replace('第 5 个编码块','零起始编号块 5').replace('第 2、5、10 个编码块','零起始编号块 2、5、10')
  s=s.replace('**图 4-2 残差路径。**','**图 4-2 残差路径。** 蓝底正值显示不具有可加性；等式只适用于原始带符号数组。')
 else:
  s=re.sub(r'The absolute AUC relative to the original target logit[^\n]+\n','',s)
  s=s.replace('**Figure 4-2 Residual paths.**','**Figure 4-2 Residual paths.** Blue positive displays are not additive; equalities apply to raw signed arrays.')
 s+='\n<!-- V5_FOUNDATIONS_APPLIED -->\n';p.write_text(s)
print('BILINGUAL_V5_FOUNDATIONS_WRITTEN')
