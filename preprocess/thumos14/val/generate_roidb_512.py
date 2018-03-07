
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

print "Get val dictionary"
META_FILE='${PATH}/data/val/validation_set_meta/validation_set.mat'
data = sio.loadmat(META_FILE)['validation_videos'][0]

ANNOTATION_DIR = '../../data/val/TH14_Temporal_annotations_validation/annotation/'
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


roidb = []
FPS = 25  # currently set FPS = 25
step = 128
path = './preprocess/'
stride=1
def generate_roi(rois, video, start, end):
  tmp = {}
  tmp['gt_classes'] = rois[:,2]
  tmp['max_classes'] = rois[:,2]
  tmp['max_overlaps'] = np.ones(len(rois))
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['wins'] = rois[:,:2] - start    
  tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0] + 1    
  tmp['bg_name'] = path + 'val/frames/' + video
  tmp['fg_name'] = path + 'val/frames/' + video
  return tmp
  

remove = 0
overall = 0
duration = []
for video in video_db:
  length = len(os.listdir('./frames/'+video))
  db = np.array(video_db[video])
  overall += len(db)
  if len(db) == 0: continue
  db[:,:2] = db[:,:2] * FPS                   
  debug = []
  for start in xrange(0, max(1, length - 512 + 1), step):   
    end = min(start + 512, length)                          
    rois = db[np.logical_and(db[:,0] >= start, db[:,1] < end)]  
    if len(rois) > 0:
      tmp = generate_roi(rois, video, start, end)
      roidb.append(tmp)
      duration = duration + [d for d in tmp['durations']]
      if USE_FLIPPED:
         flipped_tmp = copy.deepcopy(tmp)
         flipped_tmp['flipped'] = True
         roidb.append(flipped_tmp)
      for d in rois:
         debug.append(d)

  for end in xrange(length - 1, 512 - 1, -step):
    start = end - 512
    rois = db[np.logical_and(db[:,0] >= start, db[:,1] < end)]  
    if len(rois) > 0:
      tmp = generate_roi(rois, video, start, end)
      roidb.append(tmp)
      duration = duration + [d for d in tmp['durations']]
      if USE_FLIPPED:
         flipped_tmp = copy.deepcopy(tmp)
         flipped_tmp['flipped'] = True
         roidb.append(flipped_tmp)
      for d in rois:
         debug.append(d)

  debug_res = [list(x) for x in set(tuple(x) for x in debug)]
  if len(debug_res) < len(db):
    remove += len(db) - len(debug_res)

print remove, ' / ', overall

print "Save dictionary"
cPickle.dump(roidb, open('val_data_512.pkl','w'), cPickle.HIGHEST_PROTOCOL)


