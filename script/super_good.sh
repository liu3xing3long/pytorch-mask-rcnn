#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "No device id provided; set all devices available."
    DEVICE_ID=0,1,2,3,4,5,6,7
else
    DEVICE_ID=$1
fi
echo "device id:"
echo $DEVICE_ID

CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=train \
    --config_name=base_102_new_super_good \
    --debug=0 \
    TRAIN.BATCH_SIZE 8 \
    TRAIN.DO_VALIDATION True \
    TRAIN.SCHEDULE 10,10,10


