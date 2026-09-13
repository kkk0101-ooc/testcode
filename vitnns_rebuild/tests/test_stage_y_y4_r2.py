from vitnns import stage_y_y4_r2 as y
def test_control_plane_recovery_identity():
 assert y.STAGE_ID=="Y4-R2" and y.ROOT.name=="formal" and y.R1_PRODUCER_REF in y.EXPECTED_REFS and y.PRODUCER_REF!=y.R1_PRODUCER_REF
