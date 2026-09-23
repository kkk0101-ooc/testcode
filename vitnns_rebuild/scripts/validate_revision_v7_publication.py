#!/usr/bin/env python3
"""Validate bilingual Markdown, LaTeX, PDF, anonymity, and v7 evidence links."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import fitz

REPO=Path(__file__).resolve().parents[1]
PUB=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
REPORT=REPO/'reports/nns_vit_endpoint_revision_v7_r4_20260923'


def sha(path:Path)->str:
 h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()


def rec(path:Path)->dict[str,Any]:return {'path':path.relative_to(REPO).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}


def figures(text:str)->list[str]:return re.findall(r'!\[[^]]*\]\(([^)]+)\)',text)
def table_titles(text:str)->list[str]:return re.findall(r'^\*\*(?:Table|表)\s+([^*]+)\*\*',text,re.M)
def equations(text:str)->list[str]:return re.findall(r'\$\$\n(.*?)\n\$\$',text,re.S)
def normalize_math(value:str)->str:return re.sub(r'\s+','',value)


def csv_rows(name:str)->list[dict[str,str]]:
 with (REPORT/name).open() as stream:return list(csv.DictReader(stream))


def main()->int:
 errors=[];texts={};pdf_texts={};pdfs=[]
 for language in ('zh','en'):
  for kind in ('manuscript','supplement'):
   md=PUB/language/f'{kind}.md';tex=PUB/language/f'{kind}.tex';pdf=PUB/language/f'{kind}.pdf'
   for path in (md,tex,pdf):
    if not path.is_file() or path.stat().st_size==0:errors.append('MISSING:'+str(path))
   text=md.read_text()
   if kind=='supplement':
    text += '\n\n' + (PUB/language/'supplement_experiments_v1.md').read_text()
   texts[language,kind]=text;pdfs.append(pdf)
   with fitz.open(pdf) as doc:
    extracted='\n'.join(page.get_text() for page in doc);pdf_texts[language,kind]=extracted
    metadata=doc.metadata or {}
    if metadata.get('author') or metadata.get('subject') or metadata.get('keywords'):errors.append('PDF_METADATA_IDENTITY:'+str(pdf))
    if any(token in json.dumps(metadata,ensure_ascii=False) for token in ('yeziqi','/data01/','github.com')):errors.append('PDF_METADATA_PATH:'+str(pdf))
    if len(doc)<1 or any(page.rect.width<=0 or page.rect.height<=0 for page in doc):errors.append('PDF_PAGE_GEOMETRY:'+str(pdf))
    if sum(len(page.get_text('blocks')) for page in doc)<10:errors.append('PDF_TEXT_BLOCKS:'+str(pdf))
 zh_levels=[len(m.group(1)) for m in re.finditer(r'^(#{2,4})\s+',texts['zh','manuscript'],re.M)]
 en_levels=[len(m.group(1)) for m in re.finditer(r'^(#{2,4})\s+',texts['en','manuscript'],re.M)]
 if zh_levels!=en_levels:errors.append('MANUSCRIPT_SECTION_STRUCTURE')
 if figures(texts['zh','manuscript'])!=figures(texts['en','manuscript']):errors.append('MANUSCRIPT_FIGURES')
 if len(table_titles(texts['zh','manuscript']))!=len(table_titles(texts['en','manuscript'])):errors.append('MANUSCRIPT_TABLE_COUNT')
 zh_s_levels=[len(m.group(1)) for m in re.finditer(r'^(#{2,4})\s+',texts['zh','supplement'],re.M)]
 en_s_levels=[len(m.group(1)) for m in re.finditer(r'^(#{2,4})\s+',texts['en','supplement'],re.M)]
 if zh_s_levels!=en_s_levels:errors.append('SUPPLEMENT_SECTION_STRUCTURE')
 if figures(texts['zh','supplement'])!=figures(texts['en','supplement']):errors.append('SUPPLEMENT_FIGURES')
 for kind in ('manuscript','supplement'):
  zh_eq=[normalize_math(x) for x in equations(texts['zh',kind])];en_eq=[normalize_math(x) for x in equations(texts['en',kind])]
  if zh_eq!=en_eq:errors.append('EQUATION_PARITY:'+kind)
  for language in ('zh','en'):
   tex=(PUB/language/f'{kind}.tex').read_text()
   if 'Generated from Markdown by scripts/build_revision_v7_publication.py' not in tex:errors.append('TEX_PROVENANCE:'+language+kind)
   if '\\label{' not in tex or '\\ref{' not in tex:errors.append('TEX_CROSSREF:'+language+kind)
 common=['0.049','0.224','-0.157','0.378','-0.186','-0.136','11/12','all-family']
 for token in common:
  for language in ('zh','en'):
   for representation in ('manuscript','pdf'):
    source=texts[language,'manuscript'] if representation=='manuscript' else pdf_texts[language,'manuscript']
    if token not in source:errors.append(f'CRITICAL_TOKEN:{language}:{representation}:{token}')
 language_tokens={'zh':['结构合法切面','点态微分闭合','候选集合遗憾率','未获支持'],'en':['structurally valid cut','pointwise differential closure','candidate-set','not supported']}
 for language,tokens in language_tokens.items():
  for token in tokens:
   if token not in texts[language,'manuscript']:errors.append(f'MD_CLAIM:{language}:{token}')
   if token not in pdf_texts[language,'manuscript']:errors.append(f'PDF_CLAIM:{language}:{token}')
  compact_pdf=re.sub(r'\s+','',pdf_texts[language,'manuscript'])
  for token in ('Dou', 'Wrobel', '2602.06613', '2604.14477'):
   if token not in compact_pdf:errors.append(f'PDF_REFERENCE:{language}:{token}')
 rows=csv_rows('table6_stratified_summary.csv')
 metrics=['A_only_across_blocks','V_only_across_blocks','attention_branch_only_across_blocks','mlp_branch_only_across_blocks','A_V_adjusted_operation_and_block_rank_correlation','residual_adjusted_operation_and_block_rank_correlation']
 for model in ('vit_b16','deit_s16'):
  for ref in ('normalized_zero','blur_rgb'):
   for metric in metrics:
    row=next(r for r in rows if r['model']==model and r['reference']==ref and r['metric']==metric and r['block']=='')
    for key in ('median','ci95_low','ci95_high'):
     token=f"{float(row[key]):.3f}"
     if token not in texts['zh','manuscript'] or token not in texts['en','manuscript']:errors.append(f'TABLE6_VALUE:{model}:{ref}:{metric}:{key}:{token}')
 for language,kind in texts:
  base=PUB/language
  for target in figures(texts[language,kind]):
   if not (base/target).resolve().is_file():errors.append(f'FIGURE_LINK:{language}:{kind}:{target}')
 if not (PUB/'references.bib').is_file():errors.append('BIB_MISSING')
 bib=(PUB/'references.bib').read_text()
 for arxiv in ('2602.06613','2603.13652','2604.14477','2605.23393','2607.10365','2609.01282'):
  if arxiv not in bib:errors.append('BIB_ARXIV:'+arxiv)
 forbidden=[r'/data01/',r'yeziqi',r'@bytedance\.',r'github\.com[:/][^ ]+/',r'\b学号\b',r'\b导师\b']
 for language in ('zh','en'):
  for kind in ('manuscript','supplement'):
   for location,value in (('md',texts[language,kind]),('tex',(PUB/language/f'{kind}.tex').read_text()),('pdf',pdf_texts[language,kind])):
    for pattern in forbidden:
     if re.search(pattern,value,re.I):errors.append(f'ANONYMITY:{language}:{kind}:{location}:{pattern}')
 transient=[p for p in PUB.rglob('*') if p.suffix in {'.aux','.log','.fls','.fdb_latexmk','.synctex'}]
 if transient:errors.extend('TRANSIENT:'+str(p) for p in transient)
 checks=[{'id':'bilingual_structure','status':'PASS' if not any('STRUCTURE' in e or 'FIGURES' in e or 'TABLE_COUNT' in e for e in errors) else 'FAIL'},{'id':'markdown_latex_pdf_content','status':'PASS' if not any(e.startswith(('CRITICAL_TOKEN','MD_CLAIM','PDF_CLAIM','PDF_REFERENCE','TABLE6_VALUE','EQUATION_PARITY')) for e in errors) else 'FAIL'},{'id':'cross_references_and_assets','status':'PASS' if not any(e.startswith(('TEX_CROSSREF','FIGURE_LINK','BIB_')) for e in errors) else 'FAIL'},{'id':'anonymity_and_metadata','status':'PASS' if not any(e.startswith(('ANONYMITY','PDF_METADATA')) for e in errors) else 'FAIL'},{'id':'pdf_parse_and_geometry','status':'PASS' if not any(e.startswith(('PDF_PAGE','PDF_TEXT_BLOCKS')) for e in errors) else 'FAIL'}]
 report={'schema_version':1,'status':'PASS' if not errors else 'FAIL','checks':checks,'errors':errors,'files':[rec(PUB/language/f'{kind}.{suffix}') for language in ('zh','en') for kind in ('manuscript','supplement') for suffix in ('md','tex','pdf')]+[rec(PUB/'references.bib')],'pdfs':[{'path':p.relative_to(REPO).as_posix(),'pages':len(fitz.open(p)),'text_characters':len(pdf_texts[p.parent.name,p.stem]),'sha256':sha(p)} for p in pdfs],'pymupdf':fitz.__doc__.splitlines()[0]}
 (REPORT/'publication_validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'checks':len(checks),'errors':errors,'pdfs':report['pdfs']},ensure_ascii=False));return 0 if not errors else 1


if __name__=='__main__':raise SystemExit(main())
