#!/usr/bin/env python3
"""Revision-v6 publication gates. Stage 1 is independent of new outcomes."""
import argparse,ast,hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[3]
P=R/'papers/nns_vit_endpoint/publication_r2_20260920'
Q=P/'qc/revision_v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mirror(s,lang):
 return s.replace('](../figures/','](../publication_r2_20260920/figures/').replace('](../references.md)','](../publication_r2_20260920/references.md)').replace('](supplement.md)','](../publication_r2_20260920/'+lang+'/supplement.md)').replace('](supplement_experiments_v1.md)','](../publication_r2_20260920/'+lang+'/supplement_experiments_v1.md)').replace('](supplement.md#','](../publication_r2_20260920/'+lang+'/supplement.md#').replace('](../evidence/','](../publication_r2_20260920/evidence/')
def contracts(texts):
 checks=[]
 def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
 source=(R/'src/vitnns/endpoint_vit.py').read_text();tree=ast.parse(source)
 expected={'residual1':'L_F','residual2':'L_F','heads_branch':'A_F','heads_with_bypass':'L_F','qkv_branch':'A_F','routing_value_branch':'A_F','qkv_with_bypass':'L_F','routing_value_with_bypass':'L_F'}
 actual=None
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='cut_reconstruction' for x in node.targets):actual=ast.literal_eval(node.value)
 check('source_reconstruction_contract',actual==expected)
 for lang,s in texts.items():
  for token in ['heads_branch','qkv_branch','routing_value_branch','heads_with_bypass','qkv_with_bypass','routing_value_with_bypass','A_F','L_F','Total Endpoint','branch contribution','branch-internal differential partitions','routing-induced differential','value-induced differential']:
   check(lang+':term:'+token,token in s)
  for line in s.splitlines():
   if line.startswith(('| heads_branch |','| qkv_branch |','| routing_value_branch |')):check(lang+':branch_target:'+line.split('|')[1].strip(),'A_F' in line and 'L_F' not in line)
  section=s.split('### 4.6 ',1)[1].split('## 5.',1)[0]
  check(lang+':replacement_variables',all(t in section for t in ['attention probability $A$','value projection $V$','head output','residual branch activation']))
  check(lang+':five_unique_names',all(t.lower() in section.lower() for t in ['Total Endpoint','Edge-conditioned local map','Branch contribution map','Regional map score','Reference-aware Taylor score']))
  check(lang+':ranking_formulas',all(t in section for t in ['R_k(P)','R_{'+chr(92)+'mathrm{total}}(P)','T_k(P)','m_k=']))
  check(lang+':regional_table_forward_names','| A-probability / V-projection replacement |' in s and '| vs Total Endpoint regional score |' in s)
  check(lang+':no_routing_actual_edge_claim','routing/value actual edges' not in s and 'routing-induced actual edge' not in s)
  check(lang+':figure2_bypass',all(t in next(x for x in s.splitlines() if x.startswith('**Figure 2.**')) for t in ['bypass','A_F','L_F']))
  check(lang+':figure3_branch','attention branch contribution' in next(x for x in s.splitlines() if x.startswith('**Figure 3.**')))
  check(lang+':equation_delimiters',s.count('$$')%2==0)
 eq=lambda s:s.split('$$')[1::2]
 check('bilingual_display_equations',eq(texts['zh'])==eq(texts['en']))
 check('bilingual_table_rows',[x for x in texts['zh'].splitlines() if x.startswith('|')]==[x for x in texts['en'].splitlines() if x.startswith('|')])
 digits=lambda s:re.findall('^#{2,3} ([0-9]+[.][0-9]+|[0-9]+)[. ]',s,re.M)
 check('bilingual_sections',digits(texts['zh'])==digits(texts['en']))
 return checks

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['1','final'],default='1');a=parser.parse_args()
 texts={lang:(P/lang/'manuscript.md').read_text() for lang in ['zh','en']};checks=contracts(texts)
 for lang,s in texts.items():checks.append(dict(name=lang+':mirror',passed=(P.parent/lang/'full_draft.md').read_text()==mirror(s,lang)))
 f=json.loads((P/'figure_manifest.json').read_text())
 for row in f['figures']:
  for ext,rec in row['files'].items():checks.append(dict(name='figure_hash:'+str(row['number'])+':'+ext,passed=sha(P/rec['path'])==rec['sha256']))
 for main,pack in [(6,12),(8,16),(10,18)]:
  row=next(r for r in f['figures'] if r['number']==pack)
  for lang,s in texts.items():
   cap=next(x for x in s.splitlines() if x.startswith('**Figure '+str(main)+'.**'))
   checks.append(dict(name='caption_fields:'+str(main)+':'+lang,passed=all(t in cap for t in ['panel','10,000','95%','100' if main==6 else '400'])))
   checks.append(dict(name='caption_manifest:'+str(main)+':'+lang,passed=cap.split('** ',1)[1]==row['caption_'+lang]))
 if a.stage=='final':
  from revision_v6_publication_checks import final_checks
  checks.extend(final_checks(texts))
 errors=[x for x in checks if not x['passed']]
 out=dict(schema_version=1,stage=a.stage,status='PASS' if not errors else 'FAIL',checks=checks,errors=errors,coverage='theory, object classification, replacement variables, formula/section/table parity, figures and mirrors; no experiment or final-review claim')
 Q.mkdir(parents=True,exist_ok=True);(Q/('stage1_validation.json' if a.stage=='1' else 'final_validation.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+chr(10));print(json.dumps(dict(status=out['status'],checks=len(checks),errors=errors),ensure_ascii=False))
 return bool(errors)
if __name__=='__main__':raise SystemExit(main())
