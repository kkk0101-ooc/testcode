import sys
sys.path.append(".")
from cams.vit_gradcam_1 import *
import os

# 修改参数
args = get_args()
args.image_dir = r"D:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111"
args.output_dir = r"D:\project\vit_114\vitnns\49_vit\cams\results_n01443537_24"
args.method = "gradcam"
args.use_cuda = True

# 修改模型路径
model_weight_path = r"D:\project\vit_114\NNS_model_and_dataset\models\vit_3channel_mutlihead\tiny200ILSVRC_VisionTransformer_32imgsize_10classes_0threshold_3heads_12depth_4patch_size_VisionTransformer_50_final.pth"

# 运行GradCAM
if __name__ == "__main__":
    model = VisionTransformer(img_size=[32], patch_size=4, num_classes=10, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=False)
    checkpoint = torch.load(model_weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if args.use_cuda:
        model = model.cuda()
    
    target_layers = [model.blocks[-1].attn.qkv]
    
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
    }
    
    cam = methods[args.method](model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    # 只处理n01443537_24.JPEG
    image_path = os.path.join(args.image_dir, "n01443537", "n01443537_24.JPEG")
    if os.path.exists(image_path):
        print(f"Processing {image_path} ...")
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (32, 32))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        targets = None
        cam.batch_size = 1
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=args.eigen_smooth, aug_smooth=args.aug_smooth)
        grayscale_cam = grayscale_cam[0, :]
        
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "n01443537_24_gradcam.png")
        cv2.imwrite(save_path, cam_image)
        print(f"Saved to {save_path}")
