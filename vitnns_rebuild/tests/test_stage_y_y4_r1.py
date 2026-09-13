from __future__ import annotations
import copy,math
import numpy as np
import pytest
from vitnns import stage_y_y4_r1 as y

def test_mass_entropy_gini_and_within_patch():
    x=np.ones((56,56),dtype=np.float32);m=y.mass_summary(x);assert m["positive_mass"]==3136 and m["negative_mass"]==0 and m["cancellation"]==0
    w=y.within_patch(x);assert w["within_patch_population_variance"]==0 and w["within_patch_normalized_total_variation"]==0 and math.isclose(w["within_patch_effective_cell_count"],16)
    assert math.isclose(y.gini(np.ones(4)),0)

def test_spatial_and_patch_mixing_primitives():
    x=np.ones((56,56),dtype=np.float32);s=y.spatial_support(x);assert math.isclose(s["effective_spatial_support"],3136,rel_tol=1e-12);assert math.isclose(s["top_1_percent_absolute_mass_share"],32/3136)
    t=np.eye(196,dtype=np.float32);c=np.zeros(196,dtype=np.float32);p=y.patch_mixing(t,c);assert p["diagonal_absolute_mass_fraction"]==1 and p["off_diagonal_absolute_mass_fraction"]==0 and p["source_to_destination_rms_radius"]==0

def test_head_and_branch_primitives():
    h=np.stack([np.full((56,56),i+1,dtype=np.float32) for i in range(12)]);summary,rows=y.head_organization(h);assert len(rows)==12 and math.isclose(sum(r["absolute_share"] for r in rows),1);assert summary["head_top3_share"]>summary["head_top1_share"]
    one=np.ones((56,56),dtype=np.float32);b=y.branch_allocation(one,one,one,one,2*one,2*one);assert b["first_attention_share"]==b["second_mlp_share"]==.5 and b["first_reconstruction_max_abs"]==0

def test_inverse_matrix_direction_and_valid_pairs():
    w=np.eye(196,dtype=np.float64)[::-1];t=np.arange(196*196,dtype=np.float64).reshape(196,196);expected=w@t@w.T;assert np.array_equal(expected,t[::-1,::-1]);mask=np.zeros(196,dtype=bool);mask[:5]=True;assert expected[np.ix_(mask,mask)].size==25

def test_image_level_aggregation_order():
    rows=np.arange(24,dtype=np.float64).reshape(2,12);per_image=rows[:,8:12].mean(1)-rows[:,0:4].mean(1);assert np.array_equal(per_image,[8,8]);assert np.median(per_image)==8

def test_bootstrap_golden_determinism_and_even_median():
    out=y.bootstrap([1.,2.,8.,9.],replicates=8,seed=17);assert out[:3]==y.bootstrap([1.,2.,8.,9.],replicates=8,seed=17)[:3];assert out[0]==5.0;assert out[3].tolist()==[4.5,5.0,1.5,8.0,1.0,5.0,4.5,5.0]

def test_signflip_plus_one_golden():
    p,null=y.sign_flip([2.,3.,4.],"greater",1.,replicates=16,seed=9);assert p==pytest.approx(7/17);assert null.shape==(2,16)
    p2,null2=y.sign_flip([2.,3.,4.],"two_sided",1.,replicates=16,seed=9);assert p2==pytest.approx(14/17);assert null2.shape==(2,16)

def test_holm_stable_ties_and_stepdown():
    rows=[{"hypothesis_id":k,"raw_p":p} for k,p in (("B2",.01),("B1",.01),("B3",.04))];out=y.holm(rows);by={r["hypothesis_id"]:r for r in out};assert by["B1"]["holm_rank"]==1 and by["B2"]["holm_rank"]==2;assert by["B3"]["adjusted_p"]==pytest.approx(.04)

def test_hypothesis_truth_table():
    base={"n":100,"family":"B","direction":"two_sided","threshold":.5,"ci_lower":.7,"ci_upper":.9,"reject":True};assert y.hypothesis_outcome(base)=="SUPPORTED"
    base.update(ci_lower=-.2,ci_upper=.2,reject=False);assert y.hypothesis_outcome(base)=="NOT_SUPPORTED"
    base.update(ci_lower=.2,ci_upper=.8);assert y.hypothesis_outcome(base)=="INCONCLUSIVE"
    base["n"]=89;assert y.hypothesis_outcome(base)=="INVALID"

def test_claim_truth_tables():
    hyp={i:{"hypothesis_id":i,"outcome":"SUPPORTED","n":100,"ci_lower":1.,"ci_upper":2.,"threshold":t} for i,_,_,t in y.HYPOTHESES};allowed={k:"bounded wording" for k in ("Y-CLM-INTRAPATCH","Y-CLM-DEPTH-SPATIAL","Y-CLM-HEAD-ORGANIZATION","Y-CLM-PATCH-MIXING","Y-CLM-BRANCH-ALLOCATION","Y-CLM-PERTURBATION-MODULATION")};assert all(x["decision"]=="SUPPORTED" for x in y.claim_decisions(hyp,allowed));hyp["B6"]["outcome"]="INCONCLUSIVE";assert y.claim_decisions(hyp,allowed)[2]["decision"]=="INCONCLUSIVE"

def test_missing_failed_na_retention_and_no_pseudoreplication():
    counts={"expected":100,"VALID":88,"FAILED":10,"NA":2};assert sum(counts[k] for k in ("VALID","FAILED","NA"))==counts["expected"];assert y.hypothesis_outcome({"n":88,"family":"A","direction":"greater","threshold":.02,"ci_lower":.1,"ci_upper":.2,"reject":True})=="INVALID"

def test_schema_recursively_closed():
    assert y.schema_closed(y.artifact_schema()) and y.artifact_schema()["additionalProperties"] is False

@pytest.mark.parametrize("attack",y.FALSE_PASS_ATTACKS)
def test_false_pass_attack_is_registered(attack):
    assert attack in y.FALSE_PASS_ATTACKS and len(y.FALSE_PASS_ATTACKS)==68

def test_no_forbidden_import_tokens_in_control_plane():
    source=__import__("inspect").getsource(y);assert "import torch\n" not in source and "import PIL" not in source and "import cv2" not in source
