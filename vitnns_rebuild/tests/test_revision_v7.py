import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('v7',ROOT/'scripts/run_revision_v7_analysis.py')
v7=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(v7)


def test_average_ranks_and_spearman_ties():
    assert v7.average_ranks([3,1,1,2]).tolist()==[4.,1.5,1.5,3.]
    value,reason=v7.spearman([1,2,3,4],[4,3,2,1])
    assert value==pytest.approx(-1.) and reason==''
    assert v7.spearman([1,1,1,1],[1,2,3,4])==(None,'NA_CONSTANT_VECTOR')


def test_two_sided_exact_sign_and_ties():
    assert v7.exact_two_sided_sign([1,2,3,0])=={'positive':3,'negative':0,'ties':1,'raw_p':.25}
    assert v7.exact_two_sided_sign([1,-1,0])['raw_p']==1.


def test_holm_is_monotone_and_family_local():
    rows=[{'metric':'a','raw_p':.01},{'metric':'b','raw_p':.02},{'metric':'c','raw_p':.5}]
    v7.holm(rows)
    assert [r['holm_p'] for r in rows]==pytest.approx([.03,.04,.5])


def test_residualized_rank_correlation_removes_group_scale():
    rows=[]
    for block in range(4):
        for operation,shift in [('A',0.),('V',100.)]:
            rows.append({'block':block,'operation':operation,'m':shift+block,'absolute_effect':shift-block})
    raw,_=v7.spearman([r['m'] for r in rows],[r['absolute_effect'] for r in rows])
    adjusted,df,condition,reason=v7.residualized_rank_correlation(rows,('operation',))
    assert raw>0 and adjusted<0 and df==6 and condition<1e12 and reason==''


def test_closed_schema_contract():
    schema=v7.analysis_schema()
    assert schema['all_json_objects_closed']
    for value in schema['json'].values():
        assert value['additionalProperties'] is False
        assert set(value['required'])==set(value['properties'])
    assert set(schema['csv'])==set(v7.CSV_HEADERS)


def test_counterexample_logic():
    x=0.0
    full_derivative=1+2*x
    retained_derivative=1.0
    assert full_derivative==retained_derivative
    assert {'x->linear_branch','x->square_branch'}!={'x->linear_branch'}


def test_stage1_manuscripts_distinguish_structural_and_pointwise_closure():
    expected={
        'zh':('结构合法切面','点态微分闭合','s(x)=x+x^2','use-specific cotangent'),
        'en':('structurally valid cut','pointwise differential closure','s(x)=x+x^2','use-specific cotangent'),
    }
    for language,tokens in expected.items():
        text=(ROOT/f'papers/nns_vit_endpoint/publication_r2_20260920/{language}/manuscript.md').read_text()
        assert all(token in text for token in tokens)
        assert 'with nonzero local derivative at the current point' not in text
        assert '具有非零当前点局部导数' not in text
