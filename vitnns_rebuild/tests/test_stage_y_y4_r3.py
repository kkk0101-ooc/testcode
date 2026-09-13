import numpy as np
from scripts import validate_stage_y_y4_r3 as v
from vitnns import stage_y_y4_r3 as y
def test_attention_cut_is_explicit_and_nonselective():
 arrays={}
 for b in y.BLOCKS:
  p=f"b{b:02d}";arrays[p+"_first_attention_cell"]=np.ones((56,56),dtype=np.float32);arrays[p+"_first_attention_matrix"]=np.eye(196,dtype=np.float32);arrays[p+"_first_attention_cls"]=np.ones(196,dtype=np.float32);arrays[p+"_first_heads_cell"]=np.ones((12,56,56),dtype=np.float32);arrays[p+"_first_identity_cell"]=np.ones((56,56),dtype=np.float32);arrays[p+"_second_residual_cell"]=np.ones((56,56),dtype=np.float32);arrays[p+"_second_mlp_cell"]=np.ones((56,56),dtype=np.float32);arrays[p+"_input_cell"]=2*np.ones((56,56),dtype=np.float32)
 rows,heads=v.primitive(arrays,0,0,"x","clean","clean",None);assert len(rows)==12 and len(heads)==144 and all(r["diagonal_absolute_mass_fraction"]==1 for r in rows)
def test_r3_identity():assert y.STAGE_ID=="Y4-R3" and y.SPEC.name=="y4_r3_run_spec.json"
