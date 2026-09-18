#!/usr/bin/env python3
"""Build the manuscript-facing manifest for the sealed Endpoint figure package."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/"reports/nns_vit_endpoint_unified_paper_figures_r2_20260918"
OUTPUT=REPO/"manuscript/endpoint_figure_manifest.json"
MAIN={1,2,5,10,12,14,16,20,21,24,26,28}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(path):return {"path":path.relative_to(REPO).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size}
def main():
 manifest=json.loads((ROOT/"render_manifest.json").read_text());terminal=json.loads((ROOT/"terminal_receipt.json").read_text())
 if terminal["status"]!="PASS_PRESENTATION_RENDER" or manifest["figure_count"]!=28:raise ValueError("UNSEALED_OR_INCOMPLETE")
 rows=[]
 for f in manifest["figures"]:
  formats={o["format"].lower():record(ROOT/o["path"]) for o in f["outputs"]}
  rows.append({"figure":f["number"],"title":f["title"],"placement":"main_text_recommended" if f["number"] in MAIN else "supplement_recommended",**formats})
 value={"schema_version":1,"source_revision":"ENDPOINT-R2","source_root":ROOT.relative_to(REPO).as_posix(),"source_terminal_receipt":record(ROOT/"terminal_receipt.json"),"support_files":[record(ROOT/name) for name in ("README.md","figure_index.md","captions_en.md","captions_zh.md","figure_analysis.md","render_manifest.json","terminal_receipt.json")],"figure_count":28,"main_text_recommended":sorted(MAIN),"supplement_recommended":[n for n in range(1,29) if n not in MAIN],"figures":rows}
 OUTPUT.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":"PASS","output":OUTPUT.relative_to(REPO).as_posix(),"figures":28,"main":len(MAIN),"supplement":28-len(MAIN)}))
if __name__=="__main__":main()
