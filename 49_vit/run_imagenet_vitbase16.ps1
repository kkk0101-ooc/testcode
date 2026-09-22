# 使用ImageNet和ViT-base-16生成12层可视化
# 这将生成更多层的注意力图（Layer 00, 02, 04, 06, 08, 10, 12, 14, 16, 18, 20, 22）

Write-Host "================================================================" -ForegroundColor Green
Write-Host "使用ImageNet + ViT-base-16 生成12层可视化" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green

$pythonPath = "D:\Program Files\miniconda3\envs\vit_114\python.exe"
$imagePath = "D:\project\vit_114\vitnns\49_vit\datas\imagenet-scanner"

Write-Host "`n步骤1: 处理图像（使用64x64降采样以节省内存）" -ForegroundColor Cyan
& $pythonPath train_scanner_cls.py `
    --device 0 `
    --count 1 `
    --dataset ImageNet `
    --classes 1000 `
    --imagenet_path $imagePath `
    --process_wrong_classification True `
    --nns_resize 64

Write-Host "`n步骤2: 生成层注意力网格图" -ForegroundColor Cyan
& $pythonPath combine_layer_attention.py `
    --base_dir "scan_results\scan_vit\114VisionTransformer_ImageNet" `
    --grid_size 3 4

Write-Host "`n================================================================" -ForegroundColor Green
Write-Host "完成！查看结果:" -ForegroundColor Green
Write-Host "scan_results\scan_vit\114VisionTransformer_ImageNet\" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Green

