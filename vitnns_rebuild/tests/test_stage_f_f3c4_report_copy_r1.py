from __future__ import annotations

import ast
from pathlib import Path

from vitnns import stage_f_f3c4_report_copy_r1 as r1

REPO=Path(__file__).resolve().parents[1]

def test_v1_payload_validates_byte_identical():
    result=r1.validate_v1(REPO); assert result=={"status":"PASS","review_png_count":23,"all_byte_identical":True}

def test_r1_is_terminal_seal_only():
    source=(REPO/"src/vitnns/stage_f_f3c4_report_copy_r1.py").read_text(); tree=ast.parse(source); imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    assert not imports.intersection({"numpy","PIL","matplotlib","torch","timm"})
    assert not any(token in source for token in ("execute(","savefig","Image.open","np.load","run_f4"))
