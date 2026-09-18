#!/usr/bin/env python3
"""Validate and seal the complete Endpoint Scanner paper figure package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from PIL import Image
import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/"reports/nns_vit_endpoint_unified_paper_figures_r2_20260918"
R1=REPO/"reports/nns_vit_endpoint_unified_paper_figures_r1_20260918"
MATH=REPO/"reports/nns_vit_endpoint_math_comparison_r1_20260918"
EXCLUSIONS={"validation_report.json","whole_root_inventory.json","terminal_receipt.json"}


def sha(path:Path)->str:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):value.update(block)
    return value.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def signed(value):return {**value,"canonical_digest":canonical(value)}
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def verify(value):value=dict(value);observed=value.pop("canonical_digest",None);return observed==canonical(value)
def inventory(root,exclude=()):return [{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.name not in exclude]


def sections(path:Path,marker:str)->dict[int,str]:
    output={}
    for part in path.read_text().split(marker)[1:]:
        head,body=part.split("\n",1);number=int(head.split(":",1)[0].strip());output[number]=body.strip()
    return output


def validate(draft:bool=False)->list[str]:
    errors=[]
    def check(ok,msg):
        if not ok:errors.append(msg)
    def checked(path):
        value=read(path);check(verify(value),"canonical digest "+str(path));return value
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CUDA_VISIBLE_DEVICES")
    check(not set(sys.modules).intersection({"torch","timm"}),"forbidden imports")
    manifest=checked(ROOT/"render_manifest.json");figures=manifest["figures"]
    check(manifest["revision"]=="ENDPOINT-R2" and manifest["figure_count"]==28 and manifest["artifact_count"]==84 and manifest["r1_preserved"],"manifest identity")
    check([f["number"] for f in figures]==list(range(1,29)),"figure roster")
    check(manifest["inherited_figures"]==list(range(2,11)) and manifest["new_figures"]==[1,*range(11,29)],"revision scope")
    outputs=[]
    old_names=["01_method_identity","02_forward_endpoint_path","03_cnn_endpoint_gates","04_attention_qkv","05_residual_edges","06_unit_sum_energy","07_path_seed_coordinates","08_layernorm_gelu","09_structured_cuts"]
    for figure in figures:
        number=figure["number"];check({o["format"] for o in figure["outputs"]}=={"PNG","PDF","SVG"},f"formats {number}")
        for record in figure["outputs"]:
            path=ROOT/record["path"];outputs.append(path);check(path.is_file() and not path.is_symlink(),"output exists "+str(path))
            if not path.is_file():continue
            check(path.stat().st_size==record["bytes"] and sha(path)==record["sha256"],"output identity "+str(path))
            if record["format"]=="PNG":
                with Image.open(path) as image:image.load();check(image.width>=1200 and image.height>=700,f"PNG size {number}: {image.size}")
            elif record["format"]=="PDF":check(path.read_bytes().startswith(b"%PDF"),"PDF signature "+str(path))
            else:check(b"<svg" in path.read_bytes()[:4096],"SVG signature "+str(path))
            if 2<=number<=10:
                source=MATH/("figures_"+record["format"].lower())/(old_names[number-2]+"."+record["format"].lower())
                check(record.get("inherited_from")==source.relative_to(REPO).as_posix() and sha(source)==record["sha256"],"analytic inheritance "+str(number)+record["format"])
            else:check("inherited_from" not in record,"new figure inheritance marker "+str(number))
    actual=[p for folder in ("figures_png","figures_pdf","figures_svg") for p in sorted((ROOT/folder).iterdir())]
    check(set(actual)==set(outputs) and len(actual)==84,"exact output roster")
    sources=checked(ROOT/"source_inventory.json");check(sources["sources"]==manifest["source_files"] and sources["sources_digest"]==canonical(sources["sources"])==manifest["source_files_digest"],"source inventory parity")
    for record in sources["sources"]:
        path=REPO/record["path"];check(path.is_file() and path.stat().st_size==record["bytes"] and sha(path)==record["sha256"],"source drift "+record["path"])
    numeric=checked(ROOT/"numeric_display_data.json");check(not numeric["new_scientific_inference"],"new inference");check(numeric["rq3_interpretation"]=="OVERALL_SUPPORTED_WITH_ONE_REGISTERED_BOUNDARY_EXCEPTION" and numeric["omnibus_c4a"]==numeric["rq4"]=="NOT_SUPPORTED","RQ interpretation")
    check(numeric["figures"]["figure_21"]["supported"]==11 and numeric["figures"]["figure_21"]["total"]==12,"RQ3 counts")
    exception=numeric["figures"]["figure_21"]["exception"];check(abs(exception["test_results"][0]["estimate"]-.09956521739130435)<1e-15,"RQ3 exception value")
    matrix=np.array(numeric["figures"]["figure_27"]["matrix"]);check(matrix.shape==(4,4) and np.all(matrix[:,2]<0) and np.max(matrix)<.05,"RQ4 matrix")
    for filename,marker in (("captions_zh.md","## Figure "),("captions_en.md","## Figure "),("figure_analysis.md","## Image #")):
        check(set(sections(ROOT/filename,marker))==set(range(1,29)),"text coverage "+filename)
    check(re.search(r"[\u3400-\u9fff]",(ROOT/"captions_en.md").read_text()) is None,"English captions contain CJK")
    check(all(re.fullmatch(r"figure_\d{2}_[a-z0-9_]+\.(png|pdf|svg)",p.name) for p in outputs),"portable output slugs")
    r1_terminal=checked(R1/"terminal_receipt.json");check(r1_terminal["status"]=="PASS_PRESENTATION_RENDER" and r1_terminal["revision"]=="ENDPOINT-R1","R1 predecessor")
    analysis=(ROOT/"figure_analysis.md").read_text();check("Image #21" in analysis and "0.099565" in analysis and "RQ3" in analysis,"RQ3 analysis");check("Image #27" in analysis and "regional Taylor" in analysis,"RQ4 analysis")
    for path in [*ROOT.glob("*.md"),ROOT/"gallery.html"]:
        text=path.read_text();check(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]",text) is None,"control character "+str(path));links=re.findall(r'(?:href|src)="([^"]+)"',text) if path.suffix==".html" else re.findall(r'\]\(([^)]+)\)',text)
        for target in links:
            if target.startswith("#"):check('id="'+target[1:]+'"' in text,"anchor "+target)
            elif "://" not in target:check((path.parent/target).exists(),"broken link "+str(path)+" "+target)
    layout=checked(ROOT/"layout_audit.json");check(layout["status"]=="PASS" and len(layout["records"])==28 and all(not r["text_outside_canvas"] for r in layout["records"]),"layout audit")
    qc=checked(ROOT/"manual_visual_qc.json");
    if draft:check(qc["status"] in {"PENDING","PASS"},"draft QC status")
    else:
        check(qc["status"]=="PASS" and [r["number"] for r in qc["records"]]==list(range(1,29)),"visual QC coverage")
        for row,figure in zip(qc["records"],figures):check(row["status"]=="PASS" and row["png_sha256"]==next(o["sha256"] for o in figure["outputs"] if o["format"]=="PNG"),"visual QC binding "+str(figure["number"]))
    if (ROOT/"terminal_receipt.json").exists():
        terminal=checked(ROOT/"terminal_receipt.json");whole=checked(ROOT/"whole_root_inventory.json");check(terminal["status"]=="PASS_PRESENTATION_RENDER" and terminal["figure_count"]==28 and terminal["artifact_count"]==84,"terminal status");check(terminal["validator_sha256"]==sha(Path(__file__)),"validator binding");check(whole["files"]==inventory(ROOT,EXCLUSIONS) and whole["files_digest"]==canonical(whole["files"]),"whole root inventory")
        for name,value in terminal["bindings"].items():check(sha(ROOT/name)==value,"terminal binding "+name)
    return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--draft",action="store_true");parser.add_argument("--record-visual-pass",action="store_true");parser.add_argument("--seal",action="store_true");args=parser.parse_args()
    if args.record_visual_pass:
        manifest=read(ROOT/"render_manifest.json");records=[]
        for f in manifest["figures"]:png=next(o for o in f["outputs"] if o["format"]=="PNG");records.append({"number":f["number"],"png_sha256":png["sha256"],"status":"PASS","review_mode":"direct_visual_review"})
        write(ROOT/"manual_visual_qc.json",signed({"status":"PASS","records":records}));print(json.dumps({"status":"VISUAL_PASS_RECORDED","figures":28}));return 0
    sealed=(ROOT/"terminal_receipt.json").exists();errors=validate(args.draft);report=signed({"status":"FAIL" if errors else "PASS_DRAFT" if args.draft else "PASS","errors":errors,"revision":"ENDPOINT-R2","figure_count":28,"artifact_count":84,"cpu_only":True})
    if not sealed:write(ROOT/"validation_report.json",report)
    if args.seal and not errors and not sealed:
        files=inventory(ROOT,EXCLUSIONS);write(ROOT/"whole_root_inventory.json",signed({"files":files,"files_digest":canonical(files),"exclusions":sorted(EXCLUSIONS)}));write(ROOT/"terminal_receipt.json",signed({"status":"PASS_PRESENTATION_RENDER","revision":"ENDPOINT-R2","figure_count":28,"artifact_count":84,"validator_sha256":sha(Path(__file__)),"bindings":{name:sha(ROOT/name) for name in ("render_manifest.json","validation_report.json","manual_visual_qc.json","whole_root_inventory.json")},"new_scientific_inference":False,"rq3_interpretation":"OVERALL_SUPPORTED_WITH_ONE_REGISTERED_BOUNDARY_EXCEPTION","omnibus_c4a":"NOT_SUPPORTED","rq4":"NOT_SUPPORTED"}));report["sealed"]=True
    print(json.dumps(report,ensure_ascii=False));return bool(errors)


if __name__=="__main__":raise SystemExit(main())
