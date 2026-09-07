from __future__ import annotations

import ast
from pathlib import Path

from vitnns import stage_f_f3c4_r3 as r3

REPO=Path(__file__).resolve().parents[1]

def test_r3_config_and_all_r1_table_parity():
    assert r3.load_config(REPO)["allowed_input_boundary"]=="R1_ROOTS_ONLY"
    assert all(r3.table_parity(REPO,name)["status"]=="PASS" for name in r3.TABLES)

def test_r3_source_has_no_upstream_array_jpeg_model_or_render_access():
    source=(REPO/"src/vitnns/stage_f_f3c4_r3.py").read_text(); tree=ast.parse(source); imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    assert not imports.intersection({"torch","timm","PIL","matplotlib"})
    assert not any(token in source for token in ("np.load","Image.open","reconcile_terminal(","render_formal(","load_canonical_model","run_f4"))

def test_r1_inventory_is_complete_and_stable_shape():
    inventory=r3.r1_inventory(REPO); assert inventory["file_count"]==134 and inventory["total_bytes"]>90_000_000 and len(inventory["file_inventory_digest"])==64
