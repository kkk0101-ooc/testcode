"""
生成所有可视化分析 - 基于提取的数据
需要先运行 run_20images_analysis.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

print("开始生成可视化...")

# 1. Attention演化分析
print("  生成attention演化...")
num_layers = len(all_data[0]['attention'])
print(f"  检测到 {num_layers} 层")

# 集中度演化
fig, ax = plt.subplots(figsize=(12, 6))
for data in all_data:
    conc = []
    for attn in data['attention']:
        cls_attn = attn[0, :, 0, 1:].mean(0).cpu().numpy()
        conc.append(cls_attn.std())
    ax.plot(conc, alpha=0.3, color='blue')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Concentration (Std)', fontsize=12)
ax.set_title(f'Attention Concentration Evolution ({len(all_data)} images)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "visualize_attention/attention_evolution/concentration.png", dpi=150)
plt.close()

# 熵值演化
fig, ax = plt.subplots(figsize=(12, 6))
for data in all_data:
    entropy = []
    for attn in data['attention']:
        cls_attn = attn[0, :, 0, 1:].mean(0).cpu().numpy()
        cls_attn = cls_attn / cls_attn.sum()
        ent = -np.sum(cls_attn * np.log(cls_attn + 1e-10))
        entropy.append(ent)
    ax.plot(entropy, alpha=0.3, color='green')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Entropy (bits)', fontsize=12)
ax.set_title(f'Attention Entropy Evolution ({len(all_data)} images)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "visualize_attention/attention_evolution/entropy.png", dpi=150)
plt.close()

# 2. Head专业化分析
print("  生成head专业化分析...")
representative_layers = [0, min(4, num_layers-1), min(8, num_layers-1), num_layers-1]

for layer_idx in representative_layers:
    if layer_idx >= num_layers:
        continue
    # 收集该层所有head的attention
    all_heads = []
    for data in all_data:
        if layer_idx >= len(data['attention']):
            continue
        attn = data['attention'][layer_idx]
        cls_attn = attn[0, :, 0, 1:].cpu().numpy()  # (12, 196)
        all_heads.append(cls_attn)
    
    if len(all_heads) == 0:
        continue
    
    mean_heads = np.stack(all_heads).mean(0)  # (12, 196)
    num_heads = mean_heads.shape[0]
    
    # 可视化所有head
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Layer {layer_idx} - All Heads', fontsize=16)
    
    for h in range(num_heads):
        ax = axes[h//4, h%4]
        attn_map = mean_heads[h].reshape(14, 14)
        im = ax.imshow(attn_map, cmap='viridis')
        ax.set_title(f'Head {h}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"visualize_attention/head_specialization/layer{layer_idx:02d}_heads.png", dpi=150)
    plt.close()
    
    # Head相似度
    sim = cosine_similarity(mean_heads)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_title(f'Layer {layer_idx} - Head Similarity', fontsize=14)
    plt.colorbar(im, ax=ax)
    
    for i in range(num_heads):
        for j in range(num_heads):
            ax.text(j, i, f'{sim[i,j]:.2f}', ha="center", va="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"visualize_attention/head_specialization/layer{layer_idx:02d}_similarity.png", dpi=150)
    plt.close()

# 3. 空间分布
print("  生成空间分布...")
for layer_idx in [0, min(6, num_layers-1), num_layers-1]:
    all_spatial = []
    for data in all_data:
        attn = data['attention'][layer_idx]
        cls_attn = attn[0, :, 0, 1:].mean(0).cpu().numpy().reshape(14, 14)
        all_spatial.append(cls_attn)
    
    mean_spatial = np.mean(all_spatial, 0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mean_spatial, cmap='hot', interpolation='bilinear')
    ax.set_title(f'Layer {layer_idx} - Spatial Distribution', fontsize=14)
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"visualize_attention/spatial_distribution/layer{layer_idx:02d}.png", dpi=150)
    plt.close()

# 4. 量化统计
print("  生成量化统计...")
stats = {'entropy': [], 'concentration': [], 'sparsity': [], 'layer_changes': []}

for layer_idx in range(num_layers):
    layer_ent, layer_conc, layer_spar = [], [], []
    
    for data in all_data:
        attn = data['attention'][layer_idx]
        cls_attn = attn[0, :, 0, 1:].mean(0).cpu().numpy()
        cls_attn_norm = cls_attn / cls_attn.sum()
        
        ent = -np.sum(cls_attn_norm * np.log(cls_attn_norm + 1e-10))
        conc = cls_attn_norm.std()
        spar = (cls_attn_norm < 0.01).sum() / len(cls_attn_norm)
        
        layer_ent.append(ent)
        layer_conc.append(conc)
        layer_spar.append(spar)
    
    stats['entropy'].append(np.mean(layer_ent))
    stats['concentration'].append(np.mean(layer_conc))
    stats['sparsity'].append(np.mean(layer_spar))

# 层间变化
for layer_idx in range(num_layers-1):
    changes = []
    for data in all_data:
        a1 = data['attention'][layer_idx][0, :, 0, 1:].mean(0).cpu().numpy()
        a2 = data['attention'][layer_idx+1][0, :, 0, 1:].mean(0).cpu().numpy()
        changes.append(np.linalg.norm(a1 - a2))
    stats['layer_changes'].append(np.mean(changes))

# 保存统计 - 转换为Python原生类型
import json
stats_serializable = {
    'entropy': [float(x) for x in stats['entropy']],
    'concentration': [float(x) for x in stats['concentration']],
    'sparsity': [float(x) for x in stats['sparsity']],
    'layer_changes': [float(x) for x in stats['layer_changes']]
}
with open(OUTPUT_DIR / "visualize_attention/quantitative/statistics.json", 'w') as f:
    json.dump(stats_serializable, f, indent=2)

# 可视化统计
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(stats['entropy'], 'o-')
axes[0,0].set_title('Entropy Evolution')
axes[0,0].set_xlabel('Layer')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(stats['concentration'], 'o-', color='orange')
axes[0,1].set_title('Concentration Evolution')
axes[0,1].set_xlabel('Layer')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(stats['sparsity'], 'o-', color='green')
axes[1,0].set_title('Sparsity Evolution')
axes[1,0].set_xlabel('Layer')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(stats['layer_changes'], 'o-', color='red')
axes[1,1].set_title('Layer-to-Layer Changes')
axes[1,1].set_xlabel('Layer Transition')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "visualize_attention/quantitative/summary.png", dpi=150)
plt.close()

# 5. 生成HTML报告
print("  生成HTML报告...")
html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>ViT Analysis - 20 Images</title>
<style>
body {{font-family: Arial; margin: 20px; background: #f5f5f5;}}
.container {{max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px;}}
h1 {{color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px;}}
h2 {{color: #666; margin-top: 30px; border-left: 4px solid #4CAF50; padding-left: 10px;}}
.grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;}}
.item {{text-align: center; background: #f9f9f9; padding: 10px; border-radius: 5px;}}
img {{max-width: 100%; border-radius: 5px;}}
table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
th, td {{border: 1px solid #ddd; padding: 12px;}}
th {{background: #4CAF50; color: white;}}
</style></head><body><div class="container">

<h1>🔬 ViT-Base-16 分析报告</h1>
<p><strong>图片数量:</strong> {len(all_data)}</p>
<p><strong>模型:</strong> vit_base_patch16_224.augreg2_in21k_ft_in1k</p>

<h2>📊 统计摘要</h2>
<table>
<tr><th>指标</th><th>最小值</th><th>最大值</th><th>平均值</th></tr>
<tr><td>熵值</td><td>{min(stats['entropy']):.4f}</td><td>{max(stats['entropy']):.4f}</td><td>{np.mean(stats['entropy']):.4f}</td></tr>
<tr><td>集中度</td><td>{min(stats['concentration']):.4f}</td><td>{max(stats['concentration']):.4f}</td><td>{np.mean(stats['concentration']):.4f}</td></tr>
<tr><td>稀疏度</td><td>{min(stats['sparsity']):.4f}</td><td>{max(stats['sparsity']):.4f}</td><td>{np.mean(stats['sparsity']):.4f}</td></tr>
</table>

<h2>🖼️ 图片列表</h2>
<div style="background:#f9f9f9; padding:15px; border-radius:5px; font-family:monospace;">
"""

for i, data in enumerate(all_data, 1):
    html += f"{i}. {data['name']}<br>\n"

html += f"""</div>

<h2>📈 Attention演化</h2>
<div class="grid">
<div class="item"><img src="visualize_attention/attention_evolution/concentration.png"><p>集中度演化</p></div>
<div class="item"><img src="visualize_attention/attention_evolution/entropy.png"><p>熵值演化</p></div>
</div>

<h2>🎯 Head专业化</h2>
<div class="grid">
<div class="item"><img src="visualize_attention/head_specialization/layer00_heads.png"><p>Layer 0 - All Heads</p></div>
<div class="item"><img src="visualize_attention/head_specialization/layer00_similarity.png"><p>Layer 0 - Similarity</p></div>
</div>

<h2>📊 量化统计</h2>
<div class="item"><img src="visualize_attention/quantitative/summary.png"></div>

<h2>📁 输出结构</h2>
<pre style="background:#f9f9f9; padding:15px;">
analysis_results_20images/
├── visualize_attention/
│   ├── attention_evolution/
│   ├── head_specialization/
│   ├── spatial_distribution/
│   └── quantitative/
├── visualize_v/
│   └── value_evolution/
└── raw_data/
    ├── attention/
    └── values/
</pre>

</div></body></html>"""

with open(OUTPUT_DIR / "analysis_report.html", 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n[OK] 所有可视化已生成")
print(f"[OK] HTML报告: {OUTPUT_DIR}/analysis_report.html")
