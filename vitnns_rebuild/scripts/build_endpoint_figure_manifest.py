#!/usr/bin/env python3
"""Build the manuscript-facing manifest for the organized Endpoint R2 package."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/"reports/nns_vit_endpoint_all_figures_r2_20260919"
OUTPUT=REPO/"papers/nns_vit_endpoint/figure_manifest.json"
MAIN={1,2,5,10,11,14,16,18,20,26,28,29,30,31,32,33}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(path):return {"path":path.relative_to(REPO).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size}
def main():
 manifest=json.loads((ROOT/"manifest.json").read_text());terminal=json.loads((ROOT/"terminal_receipt.json").read_text())
 if terminal["status"]!="PASS_PRESENTATION_RENDER" or manifest["main_figure_count"]!=40:raise ValueError("UNSEALED_OR_INCOMPLETE")
 rows=[]
 for f in manifest["main_figures"]:
  formats={o["format"].lower():record(ROOT/o["path"]) for o in f["outputs"]}
  rows.append({"figure":f["number"],"title":f["title"],"placement":"main_text_recommended" if f["number"] in MAIN else "supplement_recommended",**formats})
 supplemental=[]
 for source in manifest["supplemental_records"]:
  item={k:v for k,v in source.items() if k not in {"path","sha256","bytes","outputs"}}
  if "outputs" in source:
   item["outputs"]=[]
   for o in source["outputs"]:
    output_record=record(ROOT/o["path"]);output_record["format"]=o["format"];item["outputs"].append(output_record)
  else:
   item.update(record(ROOT/source["path"]))
  supplemental.append(item)
 value={"schema_version":3,"source_revision":"ENDPOINT-ALL-FIGURES-R2","source_root":ROOT.relative_to(REPO).as_posix(),"source_terminal_receipt":record(ROOT/"terminal_receipt.json"),"support_files":[record(ROOT/name) for name in ("README.md","figure_index.md","captions_en.md","captions_zh.md","figure_analysis.md","manifest.json","validation_report.json","manual_visual_qc.json","display_receipt.json","terminal_receipt.json")],"figure_count":40,"supplemental_record_count":len(supplemental),"main_text_recommended":sorted(MAIN),"supplement_recommended":[n for n in range(1,41) if n not in MAIN],"display_sample_binding":manifest["display_sample_binding"],"figures":rows,"supplemental_records":supplemental}
 OUTPUT.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":"PASS","output":OUTPUT.relative_to(REPO).as_posix(),"figures":40,"main":len(MAIN),"supplement":40-len(MAIN),"supplemental_records":len(supplemental)}))
if __name__=="__main__":main()
