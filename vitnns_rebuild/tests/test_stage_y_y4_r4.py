import numpy as np
from scripts import validate_stage_y_y4_r4 as v
from vitnns import stage_y_y4_r4 as y
def test_zero_matrix_is_na_not_zero_or_failure():
 r=v.patch_mixing_na(np.zeros((196,196),dtype=np.float32),np.ones(196,dtype=np.float32));assert r["off_diagonal_absolute_mass_fraction"] is None and r["cls_routing_fraction"]==1 and r["matrix_signed_cancellation"]==0
def test_invalid_endpoint_retained_in_holm():
 rows,_,_,_=v.inference([{k:(None if k=="B3" else 1.0) for k in ("A1","B1","B2","B3","B4","B5","B6","B7","C1","C2","C3","C4")}]);r=next(x for x in rows if x["hypothesis_id"]=="B3");assert r["outcome"]=="INVALID" and r["raw_p"] is None and r["adjusted_p"]==1 and r["reject"] is False
def test_missing_intervention_is_endpoint_na_only():
 inter,blocks,controls,curves=v.recompute_interventions_full(None,{},"source",3,3,"correct");assert inter=={"A1":None,"B6":None,"B7":None};assert len(blocks)==12 and len(controls)==384 and len(curves)==192;assert all(row["status"]=="NA" for row in blocks+controls+curves)
def test_zero_within_patch_mass_keeps_defined_metrics():
 result=v.within_patch_na(np.zeros((56,56),dtype=np.float32));assert result["within_patch_population_variance"]==0 and result["within_patch_normalized_total_variation"]==0 and result["within_patch_top_cell_absolute_mass_share"] is None and result["cancellation"] is None
