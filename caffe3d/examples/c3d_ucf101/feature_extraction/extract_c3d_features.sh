#!/usr/bin/env sh

## params
# test.prototxt
# model file
# id of gpu
# batch_size
# mini_batch_num
# prefix file
# target feature

./build/tools/predict.bin \
  examples/c3d_ucf101/c3d_ucf101_test.prototxt \
  examples/c3d_ucf101/c3d_iter_25.caffemodel \
  3 \
  16 \
  1 \
  examples/c3d_ucf101/videos_output_prefix.txt \
  fc8
