## UCF-101 training demo

Follow these steps to train C3D on UCF-101.

1. Download UCF-101 dataset from [UCF-101 website](http://crcv.ucf.edu/data/UCF101.php).
2. Unzip the dataset: e.g. `unrar x UCF101.rar`
3. (Optional) video reader works more stably with extracted frames than directly with video files. Extract frames from UCF-101 videos by revising and running a helper script, `${video-caffe-root}/examples/c3d_ucf101/extract_UCF-101_frames.sh`.
4. Change `${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_{train,test}_split1.txt` to correctly point to UCF-101 videos or directories that contain extracted frames.
5. Modify `${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_train_test.prototxt` to your taste or HW specification. Especially `batch_size` may need to be adjusted for the GPU memory.
6. Run training script: e.g. `cd ${video-caffe-root} && examples/c3d_ucf101/train_ucf101.sh`
7. After ~7 epochs of training, check if you have about 45% clip accuracy. (See [original paper](https://arxiv.org/pdf/1412.0767.pdf) Figure 2 -- ~45% clip accuracy around 6th epoch.)

A typical training will yield the following loss and top-1 accuracy: ![iter-loss-accuracy plot](c3d_ucf101_train_loss_accuracy.png?raw=true "Iteration vs Training loss and top-1 accuracy")

## Files in this directory

* `train_ucf101.sh`: a main script to run for training C3D on UCF-101 data
* `c3d_ucf101_solver.prototxt`: a solver specifications -- SGD parameters, testing parametesr, etc
* `c3d_ucf101_test_split1.txt`, `c3d_ucf101_train_split1.txt`: lists of testing/training video clips in ("video directory", "starting frame num", "label") format
* `c3d_ucf101_train_test.prototxt`: training/testing network model
* `ucf101_train_mean.binaryproto`: a mean cube calculated from UCF101 training set
* `c3d_ucf101_train_loss_accuracy.png`: a sample plot of training iteration vs loss and accuracy
