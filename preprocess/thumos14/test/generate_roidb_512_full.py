# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

### please change the corresponding path prefix ${PATH} 

import scipy.io as sio
import os
import cPickle
import subprocess
import numpy as np

USE_FLIPPED = False
CLASS_INDEX_FILE = '../../data/val/Class_Index_Detection.txt'

classes = {'Background': 0}
with open(CLASS_INDEX_FILE) as f:
  for idx, line in enumerate(f):
    classes[line.split()[1]] = idx+1

print "Get test dictionary"
META_FILE='${PATH}/data/test/test_set_meta.mat'
data = sio.loadmat(META_FILE)['test_videos'][0]

ANNOTATION_DIR = '../../data/test/TH14_Temporal_Annotations_Test/annotations/annotation/'
annotation_files = os.listdir(ANNOTATION_DIR)

video_list = os.listdir('./frames/')
video_db = {}
for video in video_list:
  video_db[video] = []

for fl in annotation_files:
  with open(ANNOTATION_DIR + fl) as f:
    for annotation in f:
      video = annotation.split()[0]
      start_time = float(annotation.split()[1])
      end_time = float(annotation.split()[2])
      label = fl.split('_')[0]
      if label in classes:
        video_db[video].append([start_time, end_time, classes[label]])

for video in video_db:
  video_db[video].sort(key=lambda x: x[0])


roidb = [] # save the segments information
FPS = 25   # currently FPS=25
step = 128
path = './preprocess/'

def generate_roi(video, start, end, stride):
  tmp = {}
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['bg_name'] = path + 'test/frames/' + video
  tmp['fg_name'] = path + 'test/frames/' + video
  return tmp
  

for video in video_db:
  length = len(os.listdir('./frames/'+video))
  for win in [512]:
    stride = win / 512
    for start in xrange(0, length - win + 1, step):
      end = min(start + win, length)
      tmp = generate_roi(video, start, end, stride)
      roidb.append(tmp)
      if USE_FLIPPED:
         flipped_tmp = copy.deepcopy(tmp)
         flipped_tmp['flipped'] = True
         roidb.append(flipped_tmp)


print "Save dictionary"
cPickle.dump(roidb, open('test_data_512.pkl','w'), cPickle.HIGHEST_PROTOCOL)


