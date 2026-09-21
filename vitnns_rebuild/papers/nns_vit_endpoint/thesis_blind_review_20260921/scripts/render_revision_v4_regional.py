#!/usr/bin/env python3
"""Render corrected regional estimands from published image-level evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];S=P.parent/'publication_r2_20260920/evidence/supplement_v1/baseline_image_values.csv'
with S.open() as f:rows=list(csv.DictReader(f))
plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(10,6.6))
for ax,(m,r) in zip(axes.flat,[(m,r) for m in ['vit_b16','deit_s16'] for r in ['normalized_zero','blur_rgb']]):
 part=[x for x in rows if x['model']==m and x['reference']==r];assert len(part)==400
 vals=[[float(x[k]) for x in part] for k in ['endpoint','taylor','random_legacy','random_corrected']]
 bp=ax.boxplot(vals,showfliers=False,patch_artist=True,labels=['Path map','Regional\nTaylor','Random\noriginal','Random\ncorrected'])
 for box,col in zip(bp['boxes'],['#2166ac','#4c956c','#b7bcc3','#d7a35e']):box.set_facecolor(col);box.set_alpha(.7)
 ax.set(title=f'{m} | {r}',ylabel='Absolute-effect AUC (logit)');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
fig.suptitle('Regional replacement: original and corrected random estimands\n400 images per panel; median, quartiles, 1.5 IQR whiskers; post-hoc sensitivity analysis',fontsize=11)
fig.tight_layout(rect=(0,0,1,.92))
for ext in ['png','pdf']:fig.savefig(P/f'figures/revision_v4/regional_corrected.{ext}',dpi=180)
(P/'evidence/revision_v4/regional_render_receipt.json').write_text(json.dumps(dict(source=str(S.relative_to(P.parent)),source_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),n_per_panel=400,estimator='corrected random absolute AUC per repeat before averaging; original shown separately'),indent=2)+'\n')
