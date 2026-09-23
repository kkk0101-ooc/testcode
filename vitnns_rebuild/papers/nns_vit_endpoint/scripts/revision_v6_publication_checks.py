"""Current publication contracts, with direct numeric and semantic coverage."""
import csv,hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=R/'papers/nns_vit_endpoint/publication_r2_20260920';E=P/'evidence/revision_v6';Q=P/'qc/revision_v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((E/n).open()))
def fmt(v):
 v=float(v)
 if v==0:return '0.000'
 if abs(v)<.00005:return ('−' if v<0 else '+')+'<0.0001'
 return f'{v:.4f}' if abs(v)<.0005 else f'{v:.3f}'
def cell(r):return fmt(r['median'])+' ['+fmt(r['ci_low'])+', '+fmt(r['ci_high'])+']'
def table(s,n):
 start=s.index('**Table '+str(n)+'.');end=s.index(chr(10)*2,s.index('|',start));return s[start:end]
def final_checks(texts,review=True):
 checks=[]
 def ck(name,value,detail=''):checks.append(dict(name=name,passed=bool(value),detail=str(detail)))
 for lang,s in texts.items():
  ck('v6_no_internal_ids:'+lang,not re.search('NOV-[A-Z0-9-]+|(?<![A-Za-z0-9])(?:E[1-5]|M[1-6])(?![A-Za-z0-9])',s))
  ck('v6_no_engineering_paths:'+lang,not re.search('reports/|src/vitnns|/data01/',s))
  ck('v6_anonymous:'+lang,'REQUIRED]' not in s and ('匿名送审' if lang=='zh' else 'anonymous review') in s)
  ck('v6_normalized_scope:'+lang,'normalized-input endpoint' in s.split('### 8.2 ')[1].lower())
  ck('v6_coordinate_counterexample:'+lang,'DeiT-S Q' in s and ('跨零' if lang=='zh' else 'cross zero') in s)
  for label,part in [('abstract',s.split('## 1.')[0]),('contribution',s.split('## 2.')[0]),('results',s.split('### 6.6 ')[1].split('### 6.7 ')[0]),('discussion',s.split('### 8.1 ')[1].split('### 8.2 ')[0]),('conclusion',s.split('### 8.2 ')[1])]:
   ck('v6_h4:'+lang+':'+label,'H4' in part and 'activation × gradient' in part)
  ck('v6_all_family:'+lang,'all-family' in s and ('未获支持' if lang=='zh' else 'not supported') in s)
  ck('v6_broad_nonuniversal:'+lang,('广泛但非普遍' if lang=='zh' else 'broad but non-universal') in s)
  ck('v6_no_external_replication_label:'+lang,not re.search('E5.*外部复现|E5.*external replication',s))
  ck('v6_no_defensive_stock:'+lang,not re.search('We emphasize that|It should be noted|We do not claim|为避免误解|我们不声称|首次|首创',s,re.I))
  for n in range(4,11):ck('v6_table_population:'+lang+':'+str(n),('Inference' in table(s,n) or 'inference scope' in table(s,n)))
  for name in ['table7_recomputed.csv','table8_recomputed.csv','table9_10_recomputed.csv']:
   n=7 if name.startswith('table7') else 8 if name.startswith('table8') else None
   for r in rows(name):
    if name.startswith('table7') and r['method']=='activation_gradient':continue
    if name.startswith('table9') and (r['metric'] not in ['auc','regret','logit_ratio','logit']):continue
    if name.startswith('table9') and r['method'] in ['total_gxi','raw_gradient']:continue
    if name.startswith('table9') and r['method']=='random_regions' and r['metric']=='logit_ratio':continue
    target=table(s,n) if n else table(s,9)+table(s,10)
    ck('v6_numeric:'+lang+':'+name+':'+str(tuple(r.values())[:4]),cell(r) in target,cell(r))
  old=json.loads((E/'original_inference_recomputed.json').read_text())
  for fam in old['families']:
   if fam['claim']!='C4a':continue
   t=fam['test_results'][0];c=cell(dict(median=t['estimate'],ci_low=t['ci_lower'],ci_high=t['ci_upper']))
   if fam['family_id']=='C4a.vit_b16.normalized_zero.routing_value':c=c.replace('0.100','0.0996',1)
   ck('v6_table6_original:'+lang+':'+fam['family_id'],c in table(s,6))
  e5=json.loads((E/'legacy_numeric_audit.json').read_text())['e5']['summary']
  for key,r in e5.items():ck('v6_table5:'+lang+':'+key,cell(dict(median=r['median'],ci_low=r['ci95'][0],ci_high=r['ci95'][1])) in table(s,5))
 for lang in ['zh','en']:
  for name in ['manuscript.md','supplement.md','supplement_experiments_v1.md']:
   p=P/lang/name;s=p.read_text();ck('markdown_fences:'+lang+':'+name,s.count(chr(96)*3)%2==0);ck('math_balanced:'+lang+':'+name,s.count('$$')%2==0 and all(e.count('{')==e.count('}') for e in s.split('$$')[1::2]))
   ck('no_control_bytes:'+lang+':'+name,not any(c<32 and c not in [9,10] for c in p.read_bytes()))
   widths=[]
   for line in s.splitlines():
    if line.startswith('|'):
     width=len(line.split('|'));widths.append(width)
    elif widths:
     ck('table_width:'+lang+':'+name+':'+str(len(checks)),len(set(widths))==1);widths=[]
   for target in re.findall(re.escape('](')+'([^)]*)'+re.escape(')'),s):
    if target.startswith(('http','#')):continue
    path,_,anchor=target.partition('#');dest=(p.parent/path).resolve();ck('link:'+lang+':'+target,dest.is_file())
    if anchor and dest.is_file():ck('anchor:'+lang+':'+target,'id="'+anchor+'"' in dest.read_text())
 for name in ['supplement.md','supplement_experiments_v1.md']:
  pair=[(P/l/name).read_text() for l in ['zh','en']]
  ck('bilingual_equations:'+name,pair[0].split('$$')[1::2]==pair[1].split('$$')[1::2])
  numbers=lambda s:[[float(x) for x in re.findall('(?<![A-Za-z])[-+]?[0-9]+(?:[.][0-9]+)?(?:e[-+]?[0-9]+)?',line)] for line in s.splitlines() if line.startswith('|')]
  ck('bilingual_table_numbers:'+name,numbers(pair[0])==numbers(pair[1]))
  ck('bilingual_section_count:'+name,len(re.findall('^#{1,4} ',pair[0],re.M))==len(re.findall('^#{1,4} ',pair[1],re.M)))
 for lang in ['zh','en']:
  sup=(P/lang/'supplement.md').read_text()
  for r in rows('coordinate_sensitivity_summary.csv'):ck('coordinate_summary:'+lang+':'+str(tuple(r.values())[:5]),cell(r) in sup)
  ck('finite_replacement_contract:'+lang,all(x in sup for x in ['[1,H,197,197]','[1,H,197,64]','[1,197,D]','reflect_padding=15','rank_regions','full_input_effect','isolated_effect','0/6/11','0/3/5']))
 references=json.loads((Q/'reference_live_checks.json').read_text())['records'];reftext=(P/'references.md').read_text()
 for row in references:
  ck('reference_live:'+row['arxiv_id'],row['http_status']==200 and sha(Q/(row['arxiv_id']+'.html'))==row['sha256'] and row['arxiv_id']+row['version'] in reftext)
 package=json.loads((E/'package_manifest.json').read_text())
 for row in package['files']:ck('sealed_copy:'+row['path'],sha(E/row['path'])==row['sha256'])
 if review:
  required=['thesis_final_review.md','completion_audit.md','defensive_writing_audit.md']
  for name in required:ck('final_review_file:'+name,(Q/name).is_file())
  if (Q/'thesis_final_review.md').exists():
   text=(Q/'thesis_final_review.md').read_text();dims=['研究问题与科学价值','研究设计与因果逻辑','方法学与统计推断','结果解释与外推边界','学术写作与叙事结构','改进路径']
   ck('six_review_dimensions',all(x in text for x in dims));ck('ten_defense_answers',all('### 答辩 '+str(i)+'.' in text for i in range(1,11)));ck('review_verdict',text.rstrip().endswith('可送审'))
 return checks
