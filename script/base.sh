#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=1 python main.py \
#    --phase=train \
#    --model=coco_pretrain \
#    --config=hyli_default

#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --phase=evaluate \
#    --model=last \
#    --config=hyli_default_old

CUDA_VISIBLE_DEVICES=1 python main.py \
    --phase=evaluate \
    --model=logs/hyli_default_old20180327T0234/mask_rcnn_hyli_default_0030.pth \
    --config=hyli_default_old