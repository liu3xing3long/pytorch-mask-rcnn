#!/usr/bin/env bash
gpus=4

srun -p Med -n1 -w BJ-IDC1-10-10-15-74 --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=pytorch_maskrcnn \
--kill-on-bad-exit=1 \
python coco.py train --dataset=/mnt/lustre/liuxinglong/data/coco --model=./mask_rcnn_coco.pth


