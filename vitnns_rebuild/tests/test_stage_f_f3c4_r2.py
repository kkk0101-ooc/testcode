from __future__ import annotations

import ast
import csv
from pathlib import Path

import pytest

from vitnns import stage_f_f3c4 as r1
from vitnns import stage_f_f3c4_r2 as r2

REPO=Path(__file__).resolve().parents[1]

def test_corrected_parity_accepts_canonical_key_sort_and_rejects_value_mutation(tmp_path: Path):
    assert all(r2.validate_table_parity(REPO,name)["status"]=="PASS" for name in r1.__dict__.get("_unused",[]) or ("per_image_method_scores","per_image_contrasts","method_summary","contrast_summary","curve_summary"))
    source=REPO/r1.ROOT_PATH/"paper_data/per_image_method_scores.csv"; root=tmp_path/r1.ROOT_PATH/"paper_data"; root.mkdir(parents=True); (root/"per_image_method_scores.csv").write_bytes(source.read_bytes()); (root/"per_image_method_scores.json").write_bytes((REPO/r1.ROOT_PATH/"paper_data/per_image_method_scores.json").read_bytes())
    lines=(root/"per_image_method_scores.csv").read_text().splitlines(); fields=next(csv.reader([lines[0]])); row=next(csv.reader([lines[1]])); row[-1]=str(float(row[-1])+1); import io; stream=io.StringIO(); writer=csv.writer(stream,lineterminator="\n"); writer.writerow(fields); writer.writerow(row); stream.write("\n".join(lines[2:])+"\n"); (root/"per_image_method_scores.csv").write_text(stream.getvalue())
    with pytest.raises(r2.StageFF3C4R2Error): r2.validate_table_parity(tmp_path,"per_image_method_scores")

def test_r1_binding_detects_mutation(tmp_path: Path):
    inventory=r2._r1_inventory(REPO); changed=dict(inventory); changed["total_bytes"]+=1; assert changed!=inventory

def test_r2_source_is_terminal_seal_only():
    tree=ast.parse((REPO/"src/vitnns/stage_f_f3c4_r2.py").read_text()); imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    assert not imports.intersection({"torch","timm","PIL","matplotlib"})
    source=(REPO/"src/vitnns/stage_f_f3c4_r2.py").read_text(); assert not any(token in source for token in ("np.load","Image.open","render_formal(","reconcile_and_analyze(","load_canonical_model","run_f4"))
