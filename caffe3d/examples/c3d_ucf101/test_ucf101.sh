#!/usr/bin/env sh

# get the last model (latest)
LASTMODEL=$(ls -1t examples/c3d_ucf101/c3d_ucf101_iter_*.caffemodel | head -n 1)
echo "[Info] The caffemodel to be used: ${LASTMODEL}"

# check the # test samples and batch_size: 41822/30=1395
NUMITERS=1395
echo "[Info] Tested for ${NUMITERS} iterations"

if [ -z "${LASTMODEL}" ]; then
  echo "[Error] Can not find the model. Check the caffemodel name."
else
  build/tools/caffe \
    test \
  --model=examples/c3d_ucf101/c3d_ucf101_test.prototxt \
  --weights=${LASTMODEL} \
  --iterations=${NUMITERS} \
  --gpu=0 \
  2>&1 | tee examples/c3d_ucf101/c3d_ucf101_test.log
fi
