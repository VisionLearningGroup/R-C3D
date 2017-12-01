#!/usr/bin/env sh

# get the last model (latest)
LASTMODEL=/home/gpuadmin/Documents/segmentation/tdcnn/pretrain/ucf101.caffemodel

echo "[Info] The caffemodel to be used: ${LASTMODEL}"

# check the # test samples and batch_size: 41822/30=1395
NUMITERS=838
echo "[Info] Tested for ${NUMITERS} iterations"

if [ -z "${LASTMODEL}" ]; then
  echo "[Error] Can not find the model. Check the caffemodel name."
else
  build/tools/caffe \
    test \
  --model=./examples/c3d_finetune/c3d_ucf101_test.prototxt \
  --weights=${LASTMODEL} \
  --iterations=${NUMITERS} \
  --gpu=0 \
  2>&1 | tee ./examples/c3d_finetune/c3d_ucf101_test.log
fi
