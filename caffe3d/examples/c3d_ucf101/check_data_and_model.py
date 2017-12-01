#!/usr/bin/env python

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

import os
model_weights = './c3d_ucf101_iter_5000.caffemodel'
if not os.path.isfile(model_weights):
    print "[Error] model weights can't be found."
    sys.exit(-1)

print 'model found.'
caffe.set_mode_gpu()
caffe.set_device(1)
model_def = './c3d_ucf101_deploy.prototxt'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# convert from HxWxCxL to CxLxHxW (L=temporal length)
length = 16
transformer.set_transpose('data', (2,3,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          length,        # length of a clip
                          112, 112)  # image size

clip = np.tile(
        caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'),
        (16,1,1,1)
        )
clip = np.transpose(clip, (1,2,3,0))
print "clip.shape={}".format(clip.shape)

transformed_image = transformer.preprocess('data', clip)

#plt.imshow(image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image
for l in range(0, length):
    print "net.blobs['data'].data[0,:,{},:,:]={}".format(
            l,
            net.blobs['data'].data[0,:,l,:,:]
            )

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()
