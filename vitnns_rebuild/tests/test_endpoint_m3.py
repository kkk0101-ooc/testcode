import torch
from vitnns.endpoint_m3_fidelity import FidelityNetwork,compare_learning_images
from vitnns.endpoint_m3_operators import cases
from vitnns.endpoint_m3_protocol import selected_tokens,selected_channels,similarity_pairs,validation_targets
from vitnns.endpoint_m3_atlas import pack_rows,summarize_maps
from vitnns.endpoint_m3_statistics import image_metric_values,summarize_image_values

def test_frozen_target_selection_counts_and_identity():
    tokens=selected_tokens();assert len(tokens)==16 and len(set(tokens))==16 and tokens[0]==0
    assert all(1<=x<=196 for x in tokens[1:])
    for model,width in (("vit_b16",768),("deit_s16",384)):
        all_channels=[selected_channels(model,b,width,32,"ENDPOINT_M3_CHANNEL_20260919") for b in range(12)]
        assert all(len(x)==len(set(x))==32 for x in all_channels)
        assert len(validation_targets(tokens,all_channels[0]))==8
        assert len(similarity_pairs(model,0,tokens,all_channels[0],"same_token_different_channel",256,"ENDPOINT_M3_SIMILARITY_20260919"))==256
        assert len(similarity_pairs(model,0,tokens,all_channels[0],"different_token",256,"ENDPOINT_M3_SIMILARITY_20260919"))==256

def test_original_nns_explicit_propagation_matches_autograd_endpoint():
    for kind in ("fcn","cnn","cnn_maxpool","residual_mlp","residual_cnn"):
        model=FidelityNetwork(kind,torch.float64);x=torch.randn(model.input_shape,dtype=torch.float64)
        assert max(r["max_tolerance_ratio"] for r in compare_learning_images(model,x,1e-10,1e-9))<=1

def test_all_operator_analytic_jvps_match_autograd():
    assert [x[0] for x in cases()]==["layer_norm","gelu","softmax","qk_scaled","av","head_concat","output_projection","residual_add","mlp","final_layer_norm","classifier"]
    for _,primals,function,analytic in cases():
        directions=tuple(torch.randn_like(x) for x in primals);observed=analytic(primals,directions);expected=torch.autograd.functional.jvp(function,primals,directions,strict=True)[1]
        torch.testing.assert_close(observed,expected,atol=1e-10,rtol=1e-9)

def test_streaming_summary_zero_na_and_image_level_statistics():
    maps=torch.zeros(2,3,224,224);maps[0,:,32:48,48:64]=torch.tensor([1.,-2.,3.]).reshape(3,1,1)
    rows=summarize_maps(maps,(0,1),(2,),0);assert rows[0]["absolute_mass"]>0 and rows[0]["normalized_spatial_entropy"]==0
    assert rows[1]["zero_gradient"] and rows[1]["cancellation"] is None and rows[1]["normalized_spatial_entropy"] is None
    packed=pack_rows(rows);assert packed["patch_absolute_mass_196"].shape==(2,196) and torch.isnan(torch.tensor(packed["cancellation"][1]))
    # Expand to all phases and both token types to exercise all five frozen tests.
    synthetic=[]
    for block in range(12):
        for token in (0,1):
            row=dict(rows[0],block=block,token=token,zero_gradient=False,cancellation=.2+block/100,normalized_spatial_entropy=.3+block/100)
            synthetic.append(row)
    data=pack_rows(synthetic);values=image_metric_values(data);assert values["late_minus_early_entropy"]>0
    records=[dict(values,model_id="vit_b16",sample_id=str(i),image_sha256=str(i)) for i in range(7)]
    summaries,tests=summarize_image_values(records,100);assert len(tests)==5 and all(t["n"]==7 for t in tests) and summaries
