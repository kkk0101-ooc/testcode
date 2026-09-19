#!/usr/bin/env python3
import json,re
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"reports/nns_vit_endpoint_m3_sup_r2_20260919";PAPER=REPO/"papers/nns_vit_endpoint"
def require(x,c):
 if not x:raise ValueError(c)
def main():
 docs=[REPO/"docs/endpoint_m3_supplement_r2_20260919.md",PAPER/"en/endpoint_m3_supplement_r2.md",PAPER/"zh/endpoint_m3_supplement_r2.md"];texts=[p.read_text() for p in docs];en_full=(PAPER/"en/full_draft.md").read_text();zh_full=(PAPER/"zh/full_draft.md").read_text()
 tokens=("5,000","1.602e-7","2.427e-7","0.0625%","6.94e-18","1.840e-06","0.4752","0.4712","0.0086","0.0238","0.3082","0.3084","0.0592","0.0874","0.0283","0.0406")
 for token in tokens:
  require(all(token in text for text in texts),"SUPPLEMENT_SYNC:"+token)
 full_tokens=("5,000","0.0625%","0.4752","0.4712","0.0086","0.0238","0.3082","0.3084","0.0592","0.0874","0.0283","0.0406","1.602\\times10^{-7}","2.427\\times10^{-7}","6.94\\times10^{-18}","1.84\\times10^{-6}")
 for token in full_tokens:require(token in en_full and token in zh_full,"FULL_DRAFT_SYNC:"+token)
 for text in texts:
  require("not independent statistical samples" in text or "不是相互独立的统计样本" in text,"NUMERIC_CASE_SCOPE");require("spatial broadening of local input dependence" in text or "局部一阶输入依赖的空间扩展" in text,"BROADENING_TERM");require("not a causal measure" in text or "不是因果信息整合量" in text,"CLS_CAUSAL_BOUNDARY");require("do not identify a unique" in text or "不能把失败唯一归因" in text or "不能把这些失败唯一归因" in text,"FD_ATTRIBUTION_BOUNDARY")
 manifest=json.loads((ROOT/"figure_manifest.json").read_text());require(manifest["stage"]=="ENDPOINT-M3-SUP-R2" and [x["paper_number"] for x in manifest["figures"]]==[29,30,31,32,33],"FIGURES")
 for figure in manifest["figures"]:
  for a,b in zip(figure["files"],figure["paper_files"]):require(a["sha256"]==b["sha256"] and (REPO/a["path"]).is_file() and (REPO/b["path"]).is_file(),"FIGURE_MIRROR")
 require((PAPER/"figure_manifest_m3_sup_r2.json").read_text()==(ROOT/"figure_manifest.json").read_text(),"MANIFEST_SYNC")
 report={"status":"PASS","supplement_documents":3,"full_drafts":2,"numeric_tokens_checked":len(tokens),"figures":5,"paper_numbers":[29,30,31,32,33]};(ROOT/"document_validation.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
if __name__=="__main__":main()
