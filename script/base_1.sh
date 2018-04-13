#!/usr/bin/env bash

DEVICE_ID=1
CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=train \
    --model=last \
    --config=hyli_default_old