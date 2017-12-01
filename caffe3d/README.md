# Video-Caffe: Caffe with C3D implementation and video reader

[![Build Status](https://travis-ci.org/chuckcho/video-caffe.svg?branch=master)](https://travis-ci.org/chuckcho/video-caffe)

This is 3D convolution (C3D) and video reader implementation in the latest Caffe (Nov 2016). The original [Facebook C3D implementation](https://github.com/facebook/C3D/) is branched out from Caffe on July 17, 2014 with git commit [b80fc86](https://github.com/BVLC/caffe/tree/b80fc862952ba4e068cf74acc0823785ce1cc0e9), and has not been rebased with the original Caffe, hence missing out quite a few new features in the recent Caffe. I therefore pulled in C3D concept and an accompanying video reader and applied to the latest Caffe, and will try to rebase this repo with the upstream whenever there is a new important feature. This repo is rebased on [eb4ba30](https://github.com/BVLC/caffe/commit/eb4ba30e3c4899edc7a9713158d61503fa8ecf90), on Nov 15 2016.
Please reach [me](https://github.com/chuckcho) for any feedback or question.

Check out the [original Caffe readme](README-original.md) for Caffe-specific information.

## Branches

[`refactor` branch](https://github.com/chuckcho/video-caffe/tree/refactor) is a recent re-work, based on the [original Caffe](https://github.com/BVLC/caffe) and [Nd convolution and pooling with cuDNN PR](https://github.com/BVLC/caffe/pull/3983). This is a cleaner, less-hacky implementation of 3D convolution/pooling than the `master` branch, and is supposed to more stable than the `master` branch. So, feel free to try this branch. One missing feature in the `refactor` branch (yet) is the python wrapper.

## Requirements

In addition to [prerequisites for Caffe](http://caffe.berkeleyvision.org/installation.html#prerequisites), video-caffe depends on cuDNN. It is known to work with CuDNN verson 4 and 5, but it may need some efforts to build with v3.

* If you use "make" to build make sure `Makefile.config` point to the right paths for CUDA and CuDNN.
* If you use "cmake" to build, double-check `CUDNN_INCLUDE` and `CUDNN_LIBRARY` are correct. If not, you may want something like `cmake -DCUDNN_INCLUDE="/your/path/to/include" -DCUDNN_LIBRARY="/your/path/to/lib" ${video-caffe-root}`.

## Building video-caffe

Key steps to build video-caffe are:

1. `git clone git@github.com:chuckcho/video-caffe.git`
2. `cd video-caffe`
3. `mkdir build && cd build`
4. `cmake ..`
5. Make sure CUDA and CuDNN are detected and their paths are correct.
6. `make all -j8`
7. `make install`
8. (optional) `make runtest`

## Usage

Look at [`${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_train_test.prototxt`](examples/c3d_ucf101/c3d_ucf101_train_test.prototxt) for how 3D convolution and pooling are used. In a nutshell, use `Convolution` or `Pooling` layer with 3 repeated `{kernel_size,stride,pad}` fields that specify 3D shape (L x H x W) where `L` is the temporal length (usually 16).
```
...
# ----- video/label input -----
layer {
  name: "data"
  type: "VideoData"
  top: "data"
  top: "label"
  video_data_param {
    source: "examples/c3d_ucf101/c3d_ucf101_train_split1.txt"
    batch_size: 50
    new_height: 128
    new_width: 171
    new_length: 16
    shuffle: true
  }
  include {
    phase: TRAIN
  }
  transform_param {
    crop_size: 112
    mirror: true
    mean_value: 90
    mean_value: 98
    mean_value: 102
  }
}
...
# ----- 1st group -----
layer {
  name: "conv1a"
  type: "Convolution"
  bottom: "data"
  top: "conv1a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 3
    kernel_size: 3
    kernel_size: 3
    pad: 1
    pad: 1
    pad: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
...
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1a"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 1
    kernel_size: 2
    kernel_size: 2
    stride: 1
    stride: 2
    stride: 2
  }
}
...
```

## UCF-101 training demo

Scripts and training files for C3D training on UCF-101 are located in [examples/c3d_ucf101/](examples/c3d_ucf101/).
Steps to train C3D on UCF-101:

1. Download UCF-101 dataset from [UCF-101 website](http://crcv.ucf.edu/data/UCF101.php).
2. Unzip the dataset: e.g. `unrar x UCF101.rar`
3. (Optional) video reader works more stably with extracted frames than directly with video files. Extract frames from UCF-101 videos by revising and running a helper script, `${video-caffe-root}/examples/c3d_ucf101/extract_UCF-101_frames.sh`.
4. Change `${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_{train,test}_split1.txt` to correctly point to UCF-101 videos or directories that contain extracted frames.
5. Modify `${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_train_test.prototxt` to your taste or HW specification. Especially `batch_size` may need to be adjusted for the GPU memory.
6. Run training script: e.g. `cd ${video-caffe-root} && examples/c3d_ucf101/train_ucf101.sh` (optionally use --gpu to use multiple GPU's)
7. (Optional) Occasionally run [`${video-caffe-root}/tool/extra/plot_training_loss.sh`](tools/extra/plot_training_loss.sh) to get training loss / validation accuracy (top1/5) plot. It's pretty hacky, so look at the file to meet your need.
8. At 7 epochs of training, clip accuracy should be around 45%.

## Pretrained model

[Jimmy](https://github.com/lood339) provided a pretrained model ([downloadable link](https://dl.dropboxusercontent.com/u/54750216/C3D_models/c3d_ucf101_iter_38000.caffemodel)) for UCF101 (trained from scratch), achieving top-1 accuracy of 47% (as reported in https://github.com/chuckcho/video-caffe/issues/46).

## To-do's

1. Make a C3D-compatible python wrapper available.
2. Check CuDNN ND max-pooling works as intended in all conditions (https://github.com/BVLC/caffe/pull/3574).

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
