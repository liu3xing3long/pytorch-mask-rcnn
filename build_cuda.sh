#!/usr/bin/env bash

PATH=/usr/local/cuda/bin:$PATH
arch=sm_50

cd nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=${arch}
cd ../../
python build.py
cd ../

cd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=$arch
cd ../../
python build.py
cd ../../