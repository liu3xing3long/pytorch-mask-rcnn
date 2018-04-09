#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py \
    --phase=train \
    --model=coco_pretrain \
    --config=hyli_default