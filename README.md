# R-C3D: Region Convolutional 3D Network for Temporal Activity Detection

By Huijuan Xu, Abir Das and Kate Saenko (Boston University).

### Introduction

We propose a fast end-to-end Region Convolutional 3D Network (R-C3D) for activity detection in continuous video streams. The network encodes the frames with fully-convolutional 3D filters, proposes activity segments, then classifies and refines them based on pooled features within their boundaries.

### License

R-C3D is released under the MIT License (refer to the LICENSE file for details).

### Citing R-C3D

If you find R-C3D useful in your research, please consider citing:

    @inproceedings{Xu2017iccv,
        title = {R-C3D: Region Convolutional 3D Network for Temporal Activity Detection},
        author = {Huijuan Xu and Abir Das and Kate Saenko},
        booktitle = {Proceedings of the International Conference on Computer Vision (ICCV)},
        year = {2017}
    }

We build this repo based on Fater R-CNN, C3D and ActivityNet dataset. Please cite the following papers as well:

Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In Advances in neural information processing systems, pp. 91-99. 2015.

Tran, Du, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. "Learning spatiotemporal features with 3d convolutional networks." In Proceedings of the IEEE international conference on computer vision, pp. 4489-4497. 2015. 

Caba Heilbron, Fabian, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. "Activitynet: A large-scale video benchmark for human activity understanding." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 961-970. 2015.

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Training](#training)
4. [Testing](#testing)

### Installation:

1. Clone the R-C3D repository.
  	```Shell
  	git clone --recursive git@github.com:VisionLearningGroup/R-C3D.git
  	```
  
2. Build `Caffe3d` with `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)).

	**Note:** Caffe must be built with Python support!
  
	```Shell
	cd ./caffe3d
	
	# If have all of the requirements installed and your Makefile.config in place, then simply do:
	make -j8 && make pycaffe
	```

3. Build R-C3D lib folder.

	```Shell
	cd ./lib    
	make
	```

### Preparation:

1. Download the ground truth annatations and videos in ActivityNet dataset.

	```Shell
	cd ./preprocess/activityNet/
	
	# Download the groud truth annotations in ActivityNet dataset.
	wget http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json
	
	# Download the videos in ActivityNet dataset into ./preprocess/activityNet/videos.
	python download_video.py
	```

2. Extract frames from downloaded videos in 25 fps.

	```Shell
	# training video frames are saved in ./preprocess/activityNet/frames/training/
	# validation video frames are saved in ./preprocess/activityNet/frames/validation/ 
	python generate_frames.py
	```

3. Generate the pickle data for training R-C3D model.

	```Shell
  	# generate training data
	python generate_roidb_training.py
  	# generate validation data
	python generate_roidb_validation.py
  	```

### Training:
	
1. Download C3D classification pretrain model to ./pretrain/:

The C3D model weight pretrained on Sports1M and finetuned on ActivityNet dataset is provided in: [caffemodel](https://drive.google.com/file/d/131Cpuq1FndydeHzu38TY0baiS-uyN71w/view)

2. In R-C3D root folder, run:
	```Shell
	./experiments/activitynet/script_train.sh
  	```

### Testing

1. In R-C3D root folder, run to generate the log file
	```Shell
	./experiments/activitynet/test/script_test.sh
  	```
	
2. In ./experiments/activitynet/test folder, run to get the results.json file
	```Shell
	python activitynet_log_analysis.py test_log_<iters>.txt.*
  	```

3. In ./experiments/activitynet/test/Evaluation folder, run to get the evaluation result
	```Shell
	python get_detection_performance.py ../results.json
  	```

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments/scripts/faster_rcnn_alt_opt.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
