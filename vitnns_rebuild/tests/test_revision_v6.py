"""Adversarial coverage for the new analytic and publication contracts."""
import ast
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts'),str(R/'papers/nns_vit_endpoint/scripts')]
from vitnns.revision_v6_analytic import fanout_fixture,validate_use_maps
from validate_revision_v6 import contracts

def test_reference_has_no_scanner_or_autograd_dependency():
 tree=ast.parse((R/'src/vitnns/revision_v6_analytic.py').read_text())
 imports=[x.module for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]+[a.name for x in ast.walk(tree) if isinstance(x,ast.Import) for a in x.names]
 assert set(imports)=={'numpy','math'}

def test_real_scanner_matches_independent_reference_componentwise():
 from run_revision_v6_analytic import run
 rows,arrays,mutations,toy=run()
 assert len(rows)==41 and all(r['passed'] for r in rows)
 assert {'query','key','value','routing','head_0','head_1','identity','attention','branch_reconstruction','full','with_bypass'} <= {r['component'] for r in rows}
 assert len(mutations)==4 and all(r['correct_pass'] and r['rejected'] for r in mutations)

def test_sum_closure_does_not_hide_wrong_slots():
 expected={('z','f',0,0):np.array([1.,2.]),('z','f',1,0):np.array([4.,-1.])}
 rows=[dict(use=list(k),map=v) for k,v in expected.items()]
 assert validate_use_maps(rows,expected)
 bad=[dict(use=rows[0]['use'],map=rows[1]['map']),dict(use=rows[1]['use'],map=rows[0]['map'])]
 np.testing.assert_equal(sum(r['map'] for r in bad),sum(expected.values()))
 with pytest.raises(ValueError,match='USE_COMPONENT_MISMATCH'):validate_use_maps(bad,expected)

def test_merged_and_duplicated_fanout_rejected():
 rows,_=fanout_fixture();assert {r['mutation'] for r in rows}=={'wrong_slot','merged_fanout','merged_fanout_duplicated'}
 assert all(r['rejected'] for r in rows)

def test_publication_checker_rejects_wrong_reconstruction():
 texts={lang:(R/'papers/nns_vit_endpoint/publication_r2_20260920'/lang/'manuscript.md').read_text() for lang in ['zh','en']}
 assert all(r['passed'] for r in contracts(texts))
 bad=dict(texts);bad['en']=bad['en'].replace('| heads_branch | All head maps | Attention branch contribution', '| heads_branch | All head maps | Full target contribution').replace('($A_F$ in code)','($L_F$ in code)')
 assert any(not r['passed'] and 'branch_target' in r['name'] for r in contracts(bad))
