#!/bin/bash

GPU_ID=0
EX_DIR=thumos14

export PYTHONUNBUFFERED=true

iter=52


LOG="experiments/${EX_DIR}/test/test_log_${i}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"


time python ./experiments/${EX_DIR}/test/test_net.py --gpu ${GPU_ID} \
  --def ./experiments/${EX_DIR}/test/test.prototxt \
  --net ./experiments/${EX_DIR}/snapshot/thumos14_iter_${iter}000.caffemodel \
  --cfg ./experiments/${EX_DIR}/td_cnn_end2end.yml \
  ${EXTRA_ARGS} \
  2>&1 | tee $LOG


