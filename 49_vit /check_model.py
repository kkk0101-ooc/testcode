"""检查模型文件内容"""
import torch

model_path = r"d:\project\vit_114\NNS_model_and_dataset\models\vit_3channel_mutlihead\tiny200ILSVRC_VisionTransformer_32imgsize_10classes_45threshold_3heads_12depth_4patch_size_VisionTransformer_200_final.pth"

print(f"Loading model from: {model_path}")

try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"Model type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print("Model is a dictionary with keys:")
        for key in checkpoint.keys():
            print(f"  - {key}: {type(checkpoint[key])}")
            if key == 'model' and hasattr(checkpoint[key], '__class__'):
                print(f"    Model class: {checkpoint[key].__class__}")
    else:
        print(f"Model class: {checkpoint.__class__}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
