#!/usr/bin/env python3
"""Validate revision-v8 manuscript, supplement, PDFs, and new evidence."""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path

import fitz

REPO=Path(__file__).resolve().parents[1]
PUB=REPO/'papers/nns_vit_endpoint/publication_r2_20260920'
ROOT=REPO/'reports/nns_vit_endpoint_revision_v8_r2_20260923'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rec(path):return {'path':path.relative_to(REPO).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}
def rows(name):
    with (ROOT/name).open() as stream:return list(csv.DictReader(stream))


def main():
    errors=[];texts={};pdf_texts={};pdf_rows=[]
    for language in ('zh','en'):
        for kind in ('manuscript','supplement'):
            md=PUB/language/f'{kind}.md';tex=PUB/language/f'{kind}.tex';pdf=PUB/language/f'{kind}.pdf'
            for path in (md,tex,pdf):
                if not path.is_file() or path.stat().st_size==0:errors.append('MISSING:'+str(path))
            source=md.read_text()
            if kind=='supplement':source=source+'\n'+(PUB/language/'supplement_experiments_v1.md').read_text()
            texts[language,kind]=source
            with fitz.open(pdf) as document:
                extracted='\n'.join(page.get_text() for page in document);pdf_texts[language,kind]=extracted;metadata=document.metadata or {}
                if metadata.get('author') or metadata.get('subject') or metadata.get('keywords'):errors.append('PDF_METADATA:'+language+kind)
                if any(token in json.dumps(metadata,ensure_ascii=False) for token in ('yeziqi','/data01/','github.com')):errors.append('PDF_IDENTITY:'+language+kind)
                if len(document)<1 or sum(len(page.get_text('blocks')) for page in document)<10:errors.append('PDF_PARSE:'+language+kind)
                pdf_rows.append({'language':language,'kind':kind,'pages':len(document),'sha256':sha(pdf),'bytes':pdf.stat().st_size})
    zh=texts['zh','manuscript'];en=texts['en','manuscript'];zhs=texts['zh','supplement'];ens=texts['en','supplement']
    if re.search(r'\bRQ[1-4]\b',zh) or re.search(r'\bRQ[1-4]\b',en):errors.append('RQ_LABEL_REMAINS')
    zh_abstract=zh.split('## 1. 引言')[0];en_abstract=en.split('## 1. Introduction')[0]
    for token in ('H4','B11−B0','all-family','wrong-slot','fan-out','CPU float64'):
        if token in zh_abstract or token in en_abstract:errors.append('ABSTRACT_INTERNAL_TOKEN:'+token)
    for token in ('80,832','26,094','15,660','457/385','−0.729','−0.760','−0.226'):
        if token not in zh or token not in en or token not in zhs or token not in ens:errors.append('BILINGUAL_EVIDENCE:'+token)
    for token in ('scalar target','local-differential decomposition','finite-replacement intervention'):
        if token not in zhs or token not in ens:errors.append('REGISTRY_TERM:'+token)
    if [len(x) for x in re.findall(r'^(#{2,4}) ',zh,re.M)]!=[len(x) for x in re.findall(r'^(#{2,4}) ',en,re.M)]:errors.append('HEADING_PARITY')
    if [len(x) for x in re.findall(r'^(#{2,4}) ',zhs,re.M)]!=[len(x) for x in re.findall(r'^(#{2,4}) ',ens,re.M)]:errors.append('SUPPLEMENT_HEADING_PARITY')
    for language in ('zh','en'):
        for kind in ('manuscript','supplement'):
            tex=(PUB/language/f'{kind}.tex').read_text()
            if 'Generated from Markdown by scripts/build_revision_v8_publication.py' not in tex:errors.append('TEX_PROVENANCE:'+language+kind)
            for pattern in (r'/data01/',r'yeziqi',r'@bytedance\.'):
                if re.search(pattern,texts[language,kind],re.I) or re.search(pattern,tex,re.I) or re.search(pattern,pdf_texts[language,kind],re.I):errors.append('ANONYMITY:'+language+kind)
    for row in rows('entropy_mass_sensitivity.csv'):
        if row['metric'].startswith('trim_'):
            token=f"{float(row['median']):.3f}"
            if token not in zh or token not in en:errors.append('ENTROPY_VALUE:'+row['model']+token)
    for row in rows('h4_matched_summary.csv'):
        if row['metric']=='matched_aggregation_h4':
            token=f"{float(row['median']):.3f}"
            if token not in zh.replace('−','-') or token not in en.replace('−','-'):errors.append('H4_VALUE:'+row['model']+token)
    transient=[path for path in PUB.rglob('*') if path.suffix in {'.aux','.log','.fls','.fdb_latexmk','.synctex'}]
    errors.extend('TRANSIENT:'+str(path) for path in transient)
    report={'schema_version':1,'status':'PASS' if not errors else 'FAIL','errors':errors,'checks':['bilingual structure','abstract terminology','RQ-label removal','v8 evidence values','three-registry terminology','PDF parse and anonymity','LaTeX provenance'],'pdfs':pdf_rows,'files':[rec(PUB/language/f'{kind}.{suffix}') for language in ('zh','en') for kind in ('manuscript','supplement') for suffix in ('md','tex','pdf')]}
    (ROOT/'publication_validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False));return 0 if not errors else 1


if __name__=='__main__':raise SystemExit(main())
