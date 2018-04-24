#!/usr/bin/env bash

DEVICE_ID=4,5,6,7
CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=train \
    --debug=1 \
    --config_name=all_new_2

##results/hyli_default_old_20180327T0234/mask_rcnn_hyli_default_0048.pth
#CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
#    --device_id=$DEVICE_ID \
#    --phase=inference \
#    --config_name=all_new_2
