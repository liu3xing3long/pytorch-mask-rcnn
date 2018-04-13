#!/usr/bin/env bash

DEVICE_ID=1
CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=evaluate \
    --model=logs/hyli_default_old20180327T0234/mask_rcnn_hyli_default_0030.pth \
    --config=hyli_default_old