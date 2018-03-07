#!/bin/bash

GPU_ID=1
EX_DIR=charades

export PYTHONUNBUFFERED=true

i=162


LOG="experiments/${EX_DIR}/test/test_log_${i}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"


time python ./experiments/${EX_DIR}/test/test_net.py --gpu ${GPU_ID} \
  --def ./experiments/${EX_DIR}/test/test.prototxt \
  --net ./experiments/${EX_DIR}/snapshot/charades_iter_${i}000.caffemodel \
  --cfg ./experiments/${EX_DIR}/td_cnn_end2end.yml \
  ${EXTRA_ARGS} \
  2>&1 | tee $LOG


