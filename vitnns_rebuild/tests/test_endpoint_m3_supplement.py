import numpy as np
from vitnns.endpoint_m3_supplement import cancellation_metrics

def test_cancellation_nested_survival_identity_and_scales():
    x=np.zeros((3,224,224),np.float64);x[0,0,0]=3;x[1,0,0]=-1;x[0,1,1]=-1;x[0,32,32]=-1
    m=cancellation_metrics(x)
    assert m["coordinate_absolute_mass"]==6 and m["pixel_signed_mass"]==4
    assert m["patch_signed_mass"]==2 and m["absolute_net"]==0
    assert m["within_rgb_channels"]==1-4/6
    assert m["within_patch_after_rgb_sum"]==1-2/4
    assert m["across_patch_signed"]==1 and m["whole_image_cancellation"]==1
    assert m["survival_identity_error"]==0

def test_zero_map_propagates_na():
    assert all(v is None for v in cancellation_metrics(np.zeros((3,224,224))).values())
