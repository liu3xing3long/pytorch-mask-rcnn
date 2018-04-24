#!/usr/bin/env bash

DEVICE_ID=0
#CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
#    --device_id=$DEVICE_ID \
#    --phase=train \
#    --model=last \
#    --config=hyli_default_old

#CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
#    --device_id=$DEVICE_ID \
#    --phase=evaluate \
#    --model=last \
#    --config=hyli_default_old

#results/hyli_default_old_20180327T0234/mask_rcnn_hyli_default_0048.pth
CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=evaluate \
    --model=results/hyli_default_old_20180327T0234/mask_rcnn_hyli_default_0025.pth \
    --config=hyli_default_old

CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=evaluate \
    --model=results/hyli_default_old_20180327T0234/mask_rcnn_hyli_default_0030.pth \
    --config=hyli_default_old

CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=evaluate \
    --model=results/hyli_default_old_20180327T0234/mask_rcnn_hyli_default_0040.pth \
    --config=hyli_default_old

CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=evaluate \
    --model=results/hyli_default_old_20180327T0234/mask_rcnn_hyli_default_0048.pth \
    --config=hyli_default_old