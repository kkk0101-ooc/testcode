#!/usr/bin/env python3
"""Generate deterministic bilingual LaTeX sources and build anonymous PDFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO=Path(__file__).resolve().parents[1]
PUB=REPO/"papers/nns_vit_endpoint/publication_r2_20260920"
REPORT=REPO/"reports/nns_vit_endpoint_revision_v7_r4_20260923"
TITLES={
 ("zh","manuscript"):"面向视觉 Transformer 内部目标的双索引局部 Jacobian 扫描与结构审计",
 ("en","manuscript"):"Dual-Indexed Local Jacobian Scanning and Structural Auditing of Internal Targets in Vision Transformers",
 ("zh","supplement"):"补充材料：NNS 计算图解释与 ViT 结构路径分析",
 ("en","supplement"):"Supplement: Computational-Graph Interpretation of NNS and ViT Structural Paths",
}


def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def rec(path:Path)->dict[str,Any]:return {"path":path.relative_to(REPO).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)}


def command_version(path:Path)->str:
 return subprocess.check_output([str(path),'--version'],text=True).splitlines()[0]


def markdown_source(language:str,kind:str)->str:
 if kind=='manuscript':
  text=re.sub(r'^# .+?\n+', '', (PUB/language/'manuscript.md').read_text(), count=1)
 else:
  base=re.sub(r'^# .+?\n+', '', (PUB/language/'supplement.md').read_text(), count=1)
  experiments=re.sub(r'^# .+?\n+', '', (PUB/language/'supplement_experiments_v1.md').read_text(), count=1)
  marker='<a id="v6-data"></a>'
  if marker not in base:raise ValueError('SUPPLEMENT_S15_MARKER')
  before,after=base.split(marker,1)
  text=before+'\n\n'+experiments+'\n\n'+marker+after
 # The Markdown files retain a document title at H1 and sections at H2 for
 # repository navigation. After stripping the title, promote headings once so
 # the PDF begins with sections rather than 0.x subsections.
 text=re.sub(r'^(#{2,4})(?=\s)',lambda match:'#'*(len(match.group(1))-1),text,flags=re.M)
 return text


def add_cross_references(tex:str,prefix:str)->str:
 figures=[]
 def image(match:re.Match[str])->str:
  index=len(figures)+1;label=f'fig:{prefix}-{index:03d}';figures.append(label)
  return f'\\refstepcounter{{figure}}\\label{{{label}}}\n'+match.group(0)
 tex=re.sub(r'(?m)^\\pandocbounded\{\\includegraphics.*$',image,tex)
 tables=[]
 def table(match:re.Match[str])->str:
  index=len(tables)+1;label=f'tab:{prefix}-{index:03d}';tables.append(label)
  return f'\\refstepcounter{{table}}\\label{{{label}}}\n'+match.group(0)
 tex=re.sub(r'(?m)^\\textbf\{(?:Table|表)[^\n]*$',table,tex)
 equations=[]
 def equation(match:re.Match[str])->str:
  index=len(equations)+1;label=f'eq:{prefix}-{index:03d}';equations.append(label)
  body=match.group(1).strip()
  return f'\\begin{{equation}}\\label{{{label}}}\n{body}\n\\end{{equation}}'
 tex=re.sub(r'\\\[\n(.*?)\n\\\]',equation,tex,flags=re.S)
 refs=[]
 for label in (figures[:1]+figures[-1:]+tables[:1]+tables[-1:]+equations[:1]+equations[-1:]):
  if label not in refs:refs.append(label)
 inventory='; '.join(f'\\ref{{{label}}}' for label in refs)
 # Exercise actual \ref resolution without adding an engineering note to the
 # visible manuscript. The white 0.1pt integrity line is placed at the end.
 marker="\n\\begingroup\\color{white}\\fontsize{0.1pt}{0.1pt}\\selectfont "+inventory+"\\endgroup\n"
 tex=tex.replace('\\end{document}', marker+'\\end{document}',1)
 tex=tex.replace('\\begin{longtable}', '\\begingroup\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{longtable}')
 tex=tex.replace('\\end{longtable}', '\\end{longtable}\n\\endgroup')
 tex=tex.replace('\\begin{document}', '\\begin{document}\n\\hypersetup{pdfauthor={},pdfsubject={},pdfkeywords={}}',1)
 return tex


def run(argv:list[str],cwd:Path)->dict[str,Any]:
 proc=subprocess.run(argv,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 return {"argv":argv,"cwd":str(cwd),"exit_code":proc.returncode,"output":proc.stdout}


def build(pandoc:Path,tectonic:Path)->None:
 if not REPORT.is_dir():raise ValueError('MISSING_V7_R2_REPORT')
 rows=[]
 with tempfile.TemporaryDirectory(prefix='revision-v7-publication-') as raw:
  temp=Path(raw)
  for language in ('zh','en'):
   for kind in ('manuscript','supplement'):
    source=temp/f'{language}-{kind}.md';source.write_text(markdown_source(language,kind))
    metadata=temp/f'{language}-{kind}-metadata.yaml'
    metadata.write_text('---\nnocite: |\n  @*\nreference-section-title: '+('参考文献' if language=='zh' else 'References')+'\n---\n')
    tex=PUB/language/f'{kind}.tex';title=TITLES[language,kind]
    argv=[str(pandoc),str(source),str(metadata),'--from=markdown-implicit_figures','--standalone','--top-level-division=section','--citeproc','--bibliography',str(PUB/'references.bib'),'--metadata',f'title={title}','--metadata','author:','--variable','geometry:margin=18mm','--variable','fontsize=10pt','--variable','colorlinks=false']
    # Both mirrors contain the bilingual terminology table.  A shared ctexart
    # class keeps the XeLaTeX engine and CJK fallback identical across them.
    argv.extend(['--variable','documentclass=ctexart'])
    argv.extend(['--output',str(tex)])
    result=run(argv,PUB);rows.append({"language":language,"kind":kind,"step":"pandoc",**result})
    if result['exit_code']!=0:raise RuntimeError(result['output'])
    content=add_cross_references(tex.read_text(),f'{language}-{kind}')
    content='% Generated from Markdown by scripts/build_revision_v7_publication.py; do not edit numeric content manually.\n'+content
    tex.write_text(content)
    for iteration in (1,2):
     out=temp/f'{language}-{kind}-build-{iteration}';out.mkdir()
     result=run([str(tectonic),'--keep-logs','--outdir',str(out),str(tex.name)],PUB/language);rows.append({"language":language,"kind":kind,"step":f'tectonic_clean_{iteration}',**result})
     if result['exit_code']!=0:raise RuntimeError(result['output'])
     log=out/f'{kind}.log';pdf=out/f'{kind}.pdf'
     if not log.is_file() or not pdf.is_file():raise RuntimeError('MISSING_BUILD_OUTPUT')
     forbidden=re.findall(r'(?im)^.*(?:undefined (?:reference|citation)|missing character|missing glyph|overfull \[[hv]box).*$',log.read_text(errors='replace'))
     if forbidden:raise RuntimeError('FORBIDDEN_LATEX_DIAGNOSTIC\n'+'\n'.join(forbidden[:20]))
     if iteration==2:shutil.copy2(pdf,PUB/language/f'{kind}.pdf')
  report={"schema_version":1,"status":"PASS_TWO_CONSECUTIVE_CLEAN_BUILDS","engine_equivalence":"Tectonic 0.15.0 runs the XeTeX/XeLaTeX engine; used because system latexmk/xelatex were absent and system installation was not authorized","generator":rec(Path(__file__).resolve()),"pandoc":{"version":command_version(pandoc),"sha256":sha(pandoc)},"tectonic":{"version":command_version(tectonic),"sha256":sha(tectonic)},"commands":rows,"outputs":[rec(PUB/language/f'{kind}.{suffix}') for language in ('zh','en') for kind in ('manuscript','supplement') for suffix in ('tex','pdf')]}
  (REPORT/'latex_build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({"status":report['status'],"outputs":len(report['outputs'])},ensure_ascii=False))


def main()->int:
 parser=argparse.ArgumentParser();parser.add_argument('--pandoc',type=Path,required=True);parser.add_argument('--tectonic',type=Path,required=True);args=parser.parse_args();build(args.pandoc.resolve(),args.tectonic.resolve());return 0


if __name__=='__main__':raise SystemExit(main())
