#!/usr/bin/env bash


DEVICE_ID=4,5,6,7
CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=train \
    --config_name=None \
    --debug=0 \
    --config_file=configs/105/meta_105_quick_7.1.yaml


CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=train \
    --config_name=None \
    --debug=0 \
    --config_file=configs/105/meta_105_quick_7.2.yaml

