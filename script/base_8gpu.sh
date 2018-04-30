#!/usr/bin/env bash

DEVICE_ID=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=train \
    --config_name=None \
    --debug=0 \
    --config_file=configs/base_105.yaml  # change config file here


