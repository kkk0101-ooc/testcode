import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('v8',ROOT/'scripts/run_revision_v8_analysis.py')
v8=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(v8)


def test_rank_and_spearman_contract():
    assert v8.average_ranks([3,1,1,2]).tolist()==[4.,1.5,1.5,3.]
    value,reason=v8.spearman([1,2,3,4],[4,3,2,1])
    assert value==pytest.approx(-1.) and reason==''
    assert v8.spearman([1,1,1,1],[1,2,3,4])==(None,'NA_CONSTANT')


def test_structural_cut_dp_distinguishes_bypass_and_branch():
    edges,cuts=v8.build_graph(2)
    assert v8.path_cut_counts(edges,'b00.attn_proj',cuts[0,'heads_branch'])==({1},True)
    assert v8.path_cut_counts(edges,'b00.output',cuts[0,'heads_branch'])==({0,1},True)
    assert v8.path_cut_counts(edges,'b00.output',cuts[0,'heads_total'])==({1},True)
    assert v8.path_cut_counts(edges,'b00.output',cuts[0,'residual2'])==({1},True)
    assert v8.path_cut_counts(edges,'b00.norm1',cuts[0,'residual1'])==({0},False)
    analyzer=v8.GraphAnalyzer(edges)
    assert analyzer.cut_counts('b00.output',cuts[0,'heads_total'])==({1},True)
    assert analyzer.cut_counts('b00.norm1',cuts[0,'residual1'])==({0},False)


def test_target_classes_reconstruct_published_counts():
    source=json.loads(v8.SOURCE_TARGETS.read_text())
    expected={'vit_b16':{'post_block_token_coordinate':192,'post_block_patch_sum':48,'post_block_patch_energy':48,'attention_head_coordinate':144,'module_patch_aggregate':18,'patch_embedding_aggregate':4,'final_cls_coordinate':1,'class_logit':2},'deit_s16':{'post_block_token_coordinate':192,'post_block_patch_sum':48,'post_block_patch_energy':48,'attention_head_coordinate':72,'module_patch_aggregate':18,'patch_embedding_aggregate':4,'final_cls_coordinate':1,'class_logit':2}}
    for model,rows in source['targets_by_model'].items():
        observed=Counter(v8.classify_target(row) for row in rows)
        assert dict(observed)==expected[model]
        assert sum(observed.values())==source['target_count_by_model'][model]


def test_h4_matched_aggregation_is_distinct_from_l1():
    path=[2.,-1.]
    assert sum(abs(x) for x in path)==3.
    assert abs(sum(path))==1.
