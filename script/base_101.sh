#!/usr/bin/env bash

DEVICE_ID=1,2,3,4
CUDA_VISIBLE_DEVICES=$DEVICE_ID python -u main.py \
    --device_id=$DEVICE_ID \
    --phase=train \
    --debug=0 \
    --config_name=base_101

