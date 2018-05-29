#!/usr/bin/env bash

echo 'setup coco eval ...'
cd datasets/eval/PythonAPI
make
cd ../../../

echo 'setup NMS and RoI pooling ...'
cd lib/nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../..
python build.py
cd ../

cd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py
cd ../../../
