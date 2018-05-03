#!/usr/bin/env bash

DEVICE_ID=0,1,2,3

CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=train \
    --debug=0 \
    --config_name=base_102_new

