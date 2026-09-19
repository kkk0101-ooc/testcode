#!/usr/bin/env python3
"""Freeze and execute Endpoint M3 with a published producer and atomic queue."""
import argparse,fcntl,hashlib,json,os,shutil,sqlite3,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns.endpoint_ep1_queue import create_queue,digest
from vitnns.endpoint_m3_protocol import select_shared_cohort,selected_tokens,selected_channels
SPEC=REPO/"config/nns_vit_endpoint/endpoint_m3.json";ROOT=REPO/"reports/nns_vit_endpoint_m3_r1_20260919"
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
 return h.hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def git(*args):return subprocess.check_output(["git",*args],cwd=REPO,text=True).strip()

def prepare():
 spec=load(SPEC)
 if ROOT.exists():raise ValueError("FRESH_ROOT_REQUIRED")
 for p,s in spec["source_bindings"].items():
  if sha(REPO/p)!=s:raise ValueError("SOURCE_DRIFT:"+p)
 subprocess.run([sys.executable,"scripts/seal_endpoint_m2.py","--verify"],cwd=REPO,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1"))
 source=load(REPO/"reports/nns_vit_endpoint_m1_r1_20260918/task_manifest.json")["tasks"];by_model={}
 for t in source:by_model.setdefault(t["payload"]["model_id"],{})[t["payload"]["sample_id"]]=t["payload"]
 common=set.intersection(*(set(x) for x in by_model.values()));base=[]
 for sample_id in common:
  identities={(rows[sample_id]["image_sha256"],rows[sample_id]["class_id"]) for rows in by_model.values()}
  if len(identities)!=1:raise ValueError("SHARED_IMAGE_IDENTITY")
  image_sha256,class_id=identities.pop();base.append({"sample_id":sample_id,"image_sha256":image_sha256,"class_id":class_id})
 cohort=select_shared_cohort(base,spec["cohort"]["selected_images_per_model"],spec["cohort"]["namespace"]);tokens=selected_tokens();tasks=[];registry={}
 for model_id,model_spec in spec["models"].items():
  channels={str(block):list(selected_channels(model_id,block,model_spec["embedding_dim"],spec["targets"]["channels_per_block"],spec["targets"]["channel_selection"]["namespace"])) for block in spec["targets"]["blocks"]};registry[model_id]=channels
  for order,image in enumerate(cohort):
   payload=dict(by_model[model_id][image["sample_id"]]);payload.update(stage="ENDPOINT-M3",cohort_order=order,tokens=list(tokens),channels=channels,gradient_chunk_size=spec["numeric"]["gradient_chunk_size"],save_full_maps=order<spec["full_maps"]["images_per_model"],directions=spec["validation_subset"]["directions"],direction_namespace=spec["validation_subset"]["direction_namespace"],finite_difference_epsilons_float32=spec["validation_subset"]["finite_difference_epsilons_float32"]);tasks.append({"task_id":f"{model_id}__{order:03d}","payload":payload})
 tasks.sort(key=lambda x:x["task_id"]);ROOT.mkdir();write(ROOT/"cohort.json",{"namespace":spec["cohort"]["namespace"],"images":cohort,"digest":digest(cohort)});write(ROOT/"target_registry.json",{"tokens":list(tokens),"channels":registry,"targets_per_model_image":6144,"digest":digest({"tokens":list(tokens),"channels":registry})});write(ROOT/"task_manifest.json",{"tasks":tasks,"digest":digest(tasks)})
 paths=[SPEC,REPO/"docs/endpoint_m3_execution_20260919.md",Path(__file__),REPO/"scripts/run_endpoint_m3_worker.py",REPO/"scripts/run_endpoint_m3_fidelity.py",REPO/"scripts/run_endpoint_m3_operator_validation.py",REPO/"scripts/run_endpoint_m3_float64_review.py",REPO/"scripts/run_endpoint_m3_analysis.py",REPO/"scripts/validate_endpoint_m3.py",REPO/"src/vitnns/endpoint_m3_protocol.py",REPO/"src/vitnns/endpoint_m3_fidelity.py",REPO/"src/vitnns/endpoint_m3_operators.py",REPO/"src/vitnns/endpoint_m3_atlas.py",REPO/"src/vitnns/endpoint_m3_statistics.py",REPO/"tests/test_endpoint_m3.py"]
 write(ROOT/"freeze.json",{"time_ns":time.time_ns(),"sources":{p.relative_to(REPO).as_posix():sha(p) for p in paths},"task_manifest_sha256":sha(ROOT/"task_manifest.json"),"cohort_sha256":sha(ROOT/"cohort.json"),"target_registry_sha256":sha(ROOT/"target_registry.json"),"predecessor_head":git("rev-parse","HEAD")})
 write(ROOT/"predecessor_validation.json",{"status":"PASS","command":"scripts/seal_endpoint_m2.py --verify","terminal_status":load(REPO/"reports/nns_vit_endpoint_m2_r1_20260919/terminal_receipt.json")["status"],"requirements":17,"files":2436,"source_bindings":spec["source_bindings"]})
 create_queue(ROOT/"queue.sqlite3",tasks)
 subprocess.run([sys.executable,"scripts/run_endpoint_m3_fidelity.py","--output",str(ROOT/"m3a_fidelity.json")],cwd=REPO,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1"))
 subprocess.run([sys.executable,"scripts/run_endpoint_m3_operator_validation.py","--output",str(ROOT/"m3a_operator_validation.json")],cwd=REPO,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1"));print("M3_PREPARED")

def run():
 freeze=load(ROOT/"freeze.json")
 for p,s in freeze["sources"].items():
  if sha(REPO/p)!=s:raise ValueError("FROZEN_SOURCE:"+p)
 if sha(ROOT/"task_manifest.json")!=freeze["task_manifest_sha256"]:raise ValueError("TASK_MANIFEST_DRIFT")
 head=git("rev-parse","HEAD");branch=git("symbolic-ref","--short","HEAD");remote=git("ls-remote","origin","refs/heads/"+branch).split()[0]
 if head!=remote:raise ValueError("PUBLISHED_PRODUCER")
 for p,s in freeze["sources"].items():
  data=subprocess.check_output(["git","show",head+":vitnns_rebuild/"+p],cwd=REPO)
  if hashlib.sha256(data).hexdigest()!=s:raise ValueError("PUBLISHED_SOURCE:"+p)
 query=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True);pids=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader,nounits"],text=True);locks=[];selected=[];lock_root=REPO.parent/".runtime/endpoint_shared_gpu_locks";lock_root.mkdir(parents=True,exist_ok=True)
 spec=load(SPEC)
 for line in query.splitlines():
  index,uuid,name,free,util=[x.strip() for x in line.split(",")]
  if name!="NVIDIA H20" or int(free)<spec["runtime"]["min_free_mib"] or int(util)>spec["runtime"]["max_utilization_percent"]:continue
  stream=(lock_root/(uuid+".lock")).open("a+")
  try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:stream.close();continue
  locks.append(stream);selected.append({"index":index,"uuid":uuid,"free_mib":int(free),"util":int(util)})
  if len(selected)==spec["runtime"]["gpus_max"]:break
 if not selected:raise ValueError("NO_ELIGIBLE_GPU")
 write(ROOT/"startup.json",{"producer_commit":head,"remote_commit":remote,"freeze_sha256":sha(ROOT/"freeze.json"),"selected":selected,"gpu_snapshot":query,"existing_pids":pids,"sys_executable_realpath":str(Path(sys.executable).resolve()),"CUBLAS_WORKSPACE_CONFIG":":4096:8"})
 processes=[];started=time.monotonic()
 for gpu in selected:
  worker="gpu"+gpu["index"];log=(ROOT/(worker+".log")).open("xb");env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu["index"],CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1");cmd=[sys.executable,"scripts/run_endpoint_m3_worker.py","--root",str(ROOT),"--worker-id",worker,"--physical-index",gpu["index"],"--gpu-uuid",gpu["uuid"]];processes.append((subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT),log))
 exits=[p.wait() for p,_ in processes]
 for _,log in processes:log.close()
 with sqlite3.connect(ROOT/"queue.sqlite3") as c:states=dict(c.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state"))
 passed=all(x==0 for x in exits) and states=={"completed":200};write(ROOT/"acquisition_receipt.json",{"status":"PASS" if passed else "FAIL","states":states,"exits":exits,"wall_seconds":time.monotonic()-started,"gpu_count":len(selected)});print("M3_ACQUISITION_"+("PASS" if passed else "FAIL"));
 if not passed:raise ValueError("ACQUISITION_FAILED")

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--run",action="store_true");a=p.parse_args();prepare() if a.prepare else run() if a.run else p.error("choose --prepare or --run")
