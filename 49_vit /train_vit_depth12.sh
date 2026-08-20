#!/bin/bash
# 训练 depth=12 的 ViT 模型

cd D:\project\vit_114\vitnns\49_vit

python train_net_simple.py \
    --device 0 \
    --model VisionTransformer \
    --dataset tiny200ILSVRC \
    --classes 10 \
    --patch_size 4 \
    --num_heads 3 \
    --depth 12 \
    --img_size 32 \
    --epochs 200 \
    --threshold 45 \
    --batch_size 100 \
    --lr 0.001

