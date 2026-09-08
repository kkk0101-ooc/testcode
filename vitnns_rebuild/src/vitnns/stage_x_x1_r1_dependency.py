"""Validate and materialize the source-only X1-R1 dependencies."""
from __future__ import annotations
import argparse, fnmatch, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from jsonschema import Draft202012Validator
from .constants import REPO_ROOT

CONFIG=Path("config/stage_x_x1_r1_dependency.json")
SCHEMA=Path("config/stage_x_x1_r1_dependency.schema.json")
UPSTREAM_REF="refs/archive/stage-x-nns-v2-x1-blocked"
UPSTREAM_COMMIT="6e7dfc90b5c059f1432f21c0533afae9ea302506"
MANIFEST=Path("third_party/stage_x/dependency_manifest.json")

class DependencyError(ValueError):pass
def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise DependencyError(f"duplicate key: {key}")
        result[key]=value
    return result
def read(path):return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=_pairs,parse_constant=lambda token:(_ for _ in ()).throw(DependencyError(token)))
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def signed(value):result=dict(value);result["canonical_digest"]=digest(result);return result
def sha(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""):result.update(chunk)
    return result.hexdigest()
def git(repo,*args):
    process=subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True,check=False)
    if process.returncode:raise DependencyError(process.stderr.strip() or "git failed")
    return process.stdout.strip()
def load(repo):
    config=read(repo/CONFIG);schema=read(repo/SCHEMA);Draft202012Validator.check_schema(schema);errors=list(Draft202012Validator(schema).iter_errors(config))
    if errors:raise DependencyError(errors[0].message)
    payload=dict(config);observed=payload.pop("canonical_digest")
    if observed!=digest(payload):raise DependencyError("config digest mismatch")
    if git(repo,"rev-parse",UPSTREAM_REF+"^{commit}")!=UPSTREAM_COMMIT:raise DependencyError("upstream blocker ref drifted")
    if any(config["access_boundary"].values()) or config["limits"]["automatic_waiver"]:raise DependencyError("amendment access boundary drifted")
    return config
def _allowed(relative,patterns):return any(fnmatch.fnmatch(relative.as_posix(),pattern) for pattern in patterns)
def materialize(repo_root=REPO_ROOT):
    repo=Path(repo_root).resolve();config=load(repo)
    if (repo/MANIFEST).exists() or any((repo/item["destination"]).exists() for item in config["repositories"]):raise DependencyError("dependency destination exists; refusing overwrite")
    forbidden={value.lower() for value in config["forbidden_suffixes"]};downloaded=0;retained=0;records=[]
    with tempfile.TemporaryDirectory(prefix="stage-x-x1-r1-") as raw:
        temporary=Path(raw)
        for item in config["repositories"]:
            clone=temporary/item["id"]
            process=subprocess.run(["git","clone","--filter=blob:none","--no-checkout",item["url"],str(clone)],text=True,capture_output=True,check=False)
            if process.returncode:raise DependencyError(process.stderr.strip() or "clone failed")
            process=subprocess.run(["git","-C",str(clone),"checkout","--detach",item["commit"],"--",*item["allowlist"]],text=True,capture_output=True,check=False)
            if process.returncode:raise DependencyError(process.stderr.strip() or "allowlisted checkout failed")
            if git(clone,"rev-parse","HEAD")!=item["commit"]:raise DependencyError("dependency commit drifted")
            regular=[path for path in clone.rglob("*") if path.is_file() and ".git" not in path.parts];downloaded+=sum(path.stat().st_size for path in regular);selected=[]
            for path in regular:
                relative=path.relative_to(clone)
                if not _allowed(relative,item["allowlist"]):continue
                if any(relative.name.lower().endswith(suffix) for suffix in forbidden):raise DependencyError(f"forbidden dependency suffix: {relative}")
                selected.append(path)
            destination=repo/item["destination"]
            for source in selected:
                relative=source.relative_to(clone);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);retained+=target.stat().st_size;records.append({"engine":item["id"],"path":target.relative_to(repo).as_posix(),"bytes":target.stat().st_size,"sha256":sha(target),"source_commit":item["commit"]})
            licenses=[row for row in records if row["engine"]==item["id"] and Path(row["path"]).name.lower() in {"license","license.txt","copying"}]
            if not licenses:raise DependencyError(f"license not retained: {item['id']}")
        if downloaded>config["limits"]["download_bytes_max"] or retained>config["limits"]["retained_bytes_max"]:raise DependencyError("dependency byte cap exceeded")
        manifest=signed({"schema_version":1,"document_type":"stage_x_x1_r1_dependency_manifest","status":"PASS","repositories":[{key:item[key] for key in ("id","url","commit","license","destination")} for item in config["repositories"]],"files":sorted(records,key=lambda row:row["path"]),"file_count":len(records),"downloaded_regular_file_bytes":downloaded,"retained_bytes":retained,"download_cap":config["limits"]["download_bytes_max"],"retained_cap":config["limits"]["retained_bytes_max"],"dataset_or_checkpoint_downloaded":False,"image_or_model_file_retained":False,"torch_or_timm_imported":False,"cuda_or_gpu_accessed":False,"next_authorization":config["next_authorization"]})
        (repo/MANIFEST).parent.mkdir(parents=True,exist_ok=True);descriptor=os.open(repo/MANIFEST,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(descriptor,"wb") as stream:stream.write(canonical(manifest)+b"\n");stream.flush();os.fsync(stream.fileno())
    return manifest
def validate_materialized(repo_root=REPO_ROOT):
    repo=Path(repo_root).resolve();config=load(repo);manifest=read(repo/MANIFEST);payload=dict(manifest);observed=payload.pop("canonical_digest",None)
    if observed!=digest(payload) or manifest["status"]!="PASS":raise DependencyError("manifest invalid")
    expected=set()
    for item in config["repositories"]:
        files=[row for row in manifest["files"] if row["engine"]==item["id"]]
        if not files or any(row["source_commit"]!=item["commit"] for row in files):raise DependencyError("engine inventory drifted")
        expected.update(row["path"] for row in files)
    actual={path.relative_to(repo).as_posix() for item in config["repositories"] for path in (repo/item["destination"]).rglob("*") if path.is_file()}
    if actual!=expected:raise DependencyError("dependency file set drifted")
    for row in manifest["files"]:
        path=repo/row["path"]
        if sha(path)!=row["sha256"] or path.stat().st_size!=row["bytes"]:raise DependencyError("dependency file drifted")
    return {"status":"PASS","file_count":manifest["file_count"],"retained_bytes":manifest["retained_bytes"],"next_authorization":manifest["next_authorization"]}
def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    for name in ("validate","materialize","validate-materialized"):
        child=sub.add_parser(name);child.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    args=parser.parse_args(argv)
    try:result=load(Path(args.repo_root).resolve()) if args.command=="validate" else materialize(args.repo_root) if args.command=="materialize" else validate_materialized(args.repo_root)
    except (DependencyError,OSError,ValueError,KeyError) as error:print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True),file=sys.stderr);return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
