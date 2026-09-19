#!/usr/bin/env python3
import json,re,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/"reports/nns_vit_endpoint_m3_r1_20260919"
def require(x,c):
 if not x:raise ValueError(c)
def main():
 zh=(REPO/"papers/nns_vit_endpoint/zh/endpoint_m3_supplement.md").read_text();en=(REPO/"papers/nns_vit_endpoint/en/endpoint_m3_supplement.md").read_text();doc=(REPO/"docs/endpoint_m3_results_20260919.md").read_text();tests=json.loads((ROOT/"atlas_tests.json").read_text())["tests"]
 for text in (zh,en,doc):
  require("1,228,800" in text,"TARGET_COUNT");require("5,000" in text and "88" in text,"M3A_COUNTS");require("0.338%" in text and "0.677%" in text,"COVERAGE");require("stratified audit covers" in text or "分层审计覆盖" in text,"POSITIVE_SCOPE");require("completeness" in text,"COMPLETENESS_BOUNDARY")
 for t in tests:
  for text in (zh,en):require(t["contrast"] in text and f"{t['holm_p']:.3e}" in text,"TEST_SYNC")
 banned=("complete vit neuron scan","all vit neurons","causally important","universal intervention")
 require(not any(x in zh.lower() or x in en.lower() for x in banned),"OVERCLAIM")
 require((REPO/"papers/nns_vit_endpoint/figure_manifest_m3.json").read_text()==(ROOT/"figure_manifest.json").read_text(),"FIGURE_MANIFEST_SYNC")
 # Full-map examples are raw signed float32 maps and are not display-normalized paper figures.
 full=json.loads((ROOT/"full_map_manifest.json").read_text());require(full["file_count"]==20 and full["map_count"]==120,"FULL_MAP_MANIFEST")
 report={"status":"PASS","documents":3,"tests_crosschecked":len(tests),"scope_boundaries":4,"figure_manifest_sync":True,"full_map_files":20,"full_maps":120};(ROOT/"document_validation.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
if __name__=="__main__":main()
