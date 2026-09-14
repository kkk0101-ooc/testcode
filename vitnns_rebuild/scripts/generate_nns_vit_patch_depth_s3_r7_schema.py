#!/usr/bin/env python3
"""Generate a recursively closed S3-R7 publication schema from R6 shapes."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/"evidence/nns_vit_patch_depth/s3_r6_20260915"
OUTPUT=REPO/"config/nns_vit_patch_depth/s3_r7_publication.schema.json"
def infer(value:Any)->dict[str,Any]:
    if value is None:return {"type":"null"}
    if isinstance(value,bool):return {"type":"boolean"}
    if isinstance(value,int):return {"type":"integer"}
    if isinstance(value,float):return {"type":"number"}
    if isinstance(value,str):return {"type":"string"}
    if isinstance(value,list):
        if not value:return {"type":"array","maxItems":0}
        item=infer(value[0])
        for other in value[1:]:item=merge(item,infer(other))
        return {"type":"array","items":item}
    if isinstance(value,dict):return {"type":"object","additionalProperties":False,"required":sorted(value),"properties":{key:infer(item) for key,item in sorted(value.items())}}
    raise TypeError(type(value))
def merge(left:dict[str,Any],right:dict[str,Any])->dict[str,Any]:
    if left==right:return left
    lt,rt=left.get("type"),right.get("type")
    if {lt,rt}<={"integer","number"}:return {"type":"number"}
    if lt==rt=="object":
        keys=set(left["properties"])|set(right["properties"]);required=set(left["required"])&set(right["required"]);properties={}
        for key in sorted(keys):
            if key in left["properties"] and key in right["properties"]:properties[key]=merge(left["properties"][key],right["properties"][key])
            else:properties[key]=(left["properties"] if key in left["properties"] else right["properties"])[key]
        return {"type":"object","additionalProperties":False,"required":sorted(required),"properties":properties}
    if lt==rt=="array":
        if "items" not in left:return right
        if "items" not in right:return left
        return {"type":"array","items":merge(left["items"],right["items"])}
    alternatives=[]
    for schema in (left,right):
        if "anyOf" in schema:alternatives.extend(schema["anyOf"])
        elif schema not in alternatives:alternatives.append(schema)
    unique=[]
    for schema in alternatives:
        if schema not in unique:unique.append(schema)
    return {"anyOf":unique}
def assert_closed(schema:Any,path:str="$")->None:
    if isinstance(schema,dict):
        if schema.get("type")=="object" and schema.get("additionalProperties") is not False:raise ValueError(f"open object at {path}")
        for key,value in schema.items():assert_closed(value,f"{path}.{key}")
    elif isinstance(schema,list):
        for index,value in enumerate(schema):assert_closed(value,f"{path}[{index}]")
def main()->int:
    docs=[json.loads(path.read_text()) for path in sorted(SOURCE.glob("*.json"))]
    definitions={doc["document_type"]:infer(doc["payload"]) for doc in docs}
    schema={"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"s3_r7_publication.schema.json","type":"object","additionalProperties":False,"required":["$schema","schema_version","study_id","stage_id","run_id","document_type","status","producer_commit","payload","canonical_digest"],"properties":{"$schema":{"const":"s3_r7_publication.schema.json"},"schema_version":{"const":1},"study_id":{"const":"NNS_VIT_PATCH_DEPTH_SEMANTICS"},"stage_id":{"const":"S3"},"run_id":{"const":"s3_r7_20260915"},"document_type":{"enum":sorted(definitions)},"status":{"const":"PASS"},"producer_commit":{"type":"string","pattern":"^[0-9a-f]{40}$"},"payload":{},"canonical_digest":{"type":"string","pattern":"^[0-9a-f]{64}$"}},"allOf":[{"if":{"properties":{"document_type":{"const":kind}}},"then":{"properties":{"payload":{"$ref":f"#/$defs/{kind}"}}}} for kind in sorted(definitions)],"$defs":definitions}
    assert_closed(definitions);OUTPUT.write_text(json.dumps(schema,sort_keys=True,indent=2)+"\n");print(json.dumps({"status":"PASS","document_type_count":len(definitions),"recursive_closed":True,"output":OUTPUT.relative_to(REPO).as_posix()}));return 0
if __name__=="__main__":raise SystemExit(main())
