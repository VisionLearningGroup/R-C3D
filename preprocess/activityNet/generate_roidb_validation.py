# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
import copy
import json
import cPickle
import numpy as np
import cv2
from util import *

FPS = 25
LENGTH = 768
STEP = LENGTH / 4
WINS = [LENGTH * 8]
META_FILE = './activity_net.v1-3.min.json'
data = json.load(open(META_FILE))

USE_FLIPPED = False

print 'Generate Classes'
classes = generate_classes(data)

path = './preprocess/activityNet/frames/'
def generate_roi(video, start, end, stride, split):
  tmp = {}
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['bg_name'] = path + split + '/' + video
  tmp['fg_name'] = path + split + '/' + video
  if not os.path.isfile('../../' + tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'):
    print '../../' + tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'
    raise
  return tmp

def generate_roidb(split):
  VIDEO_PATH = 'frames/%s/' % split
  video_list = os.listdir(VIDEO_PATH)
  roidb = []
  for i,vid in enumerate(video_list):
    print i
    length = len(os.listdir('./frames/' + split + '/' + vid))
    if length ==0:
       print('length =0 in' + str(vid))
       continue
    for win in WINS:
      stride = win / LENGTH
      step = stride * STEP
      # Forward Direction
      for start in xrange(0, max(1, length - win + 1), step):
        end = min(start + win, length)
        assert end <= length

        # Add data
        tmp = generate_roi(vid, start, end, stride, split)
        roidb.append(tmp)

        if USE_FLIPPED:
          flipped_tmp = copy.deepcopy(tmp)
          flipped_tmp['flipped'] = True
          roidb.append(flipped_tmp)

      # Backward Direction
      # for end in xrange(length, win, - step):
      for end in xrange(length, win-1, - step):
        start = end - win
        assert start >= 0

        # Add data
        tmp = generate_roi(vid, start, end, stride, split)
        roidb.append(tmp)

        if USE_FLIPPED:
          flipped_tmp = copy.deepcopy(tmp)
          flipped_tmp['flipped'] = True
          roidb.append(flipped_tmp)

  return roidb
      
val_roidb = generate_roidb('validation')
print len(val_roidb)
  
print "Save dictionary"
cPickle.dump(val_roidb, open('val_data_3fps.pkl','w'), cPickle.HIGHEST_PROTOCOL)
