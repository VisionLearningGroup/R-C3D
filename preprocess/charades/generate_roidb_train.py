# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

### please change the corresponding path prefix ${PATH}

import os
import csv
import cPickle
import subprocess
import numpy as np
import copy

#USE_FLIPPED = False
USE_FLIPPED = True
META_FILE = '../../charades/Charades/Charades_v1_%s.csv'
PATH = '${PATH}/preprocess/charades/frames/'

FPS = 25
LENGTH = 768
STEP = LENGTH / 4
WINS = [LENGTH * 5]

min_length = 0 # frame num (filter out one second)  
overlap_thresh = 0.80

def generate_roi(rois, vid, start, end, stride):
  tmp = {}
  tmp['wins'] = ( rois[:,:2] - start ) / stride
  tmp['gt_classes'] = rois[:,2]
  tmp['max_classes'] = rois[:,2]
  tmp['max_overlaps'] = np.ones(len(rois))
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0] + 1
  tmp['bg_name'] = PATH + vid
  tmp['fg_name'] = PATH + vid
  if not os.path.isfile(tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'):
    print tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'
    raise
  return tmp

def generate_roidb(split):
  with open(META_FILE%split, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = [row for row in reader]
  data = data[1:] # get rid of the first row

  video_list = set(os.listdir(PATH))

  roidb = []
  # duration = []
  # video_db = {}
  for i,row in enumerate(data):
    vid = row[0]
    if vid in video_list:
      if len(row[9]) != 0:
        acts = row[9].split(';')
        db = np.zeros((len(acts), 3))
        for i, item in enumerate(acts):
          db[i,0] = int(round(float(item.split()[1])*FPS))
          db[i,1] = int(round(float(item.split()[2])*FPS))
          db[i,2] = int(item.split()[0][1:])+1  # add 0: background

        folder = PATH + vid + '/'
        length = len(os.listdir(folder))

        for win in WINS:
          stride = win / LENGTH
          step = stride * STEP
          # Forward Direction
          for start in xrange(0, max(1, length - win + 1), step):
            end = min(start + win, length)
            assert end <= length
            rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] <= start))]

            # Remove duration less than min_length                                                   
            if len(rois) > 0:
              duration = rois[:,1] - rois[:,0]
              rois = rois[duration >= min_length]

            # Remove overlap less than overlap_thresh 
            if len(rois) > 0:
              overlap = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0 / (rois[:,1] - rois[:,0])
              assert min(overlap) >= 0
              assert max(overlap) <= 1
              rois = rois[overlap >= overlap_thresh]

            # Add data
            if len(rois) > 0:
              rois[:,0] = np.maximum(start, rois[:,0])
              rois[:,1] = np.minimum(end, rois[:,1])
              tmp = generate_roi(rois, vid, start, end, stride)
              roidb.append(tmp)
              if USE_FLIPPED:   
                flipped_tmp = copy.deepcopy(tmp)
                flipped_tmp['flipped'] = True
                roidb.append(flipped_tmp)

          # Backward Direction
          for end in xrange(length, win-1, - step):
            start = end - win
            assert start >= 0
            rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] <= start))]

            # Remove duration less than min_length                                                   
            if len(rois) > 0:
              duration = rois[:,1] - rois[:,0]
              rois = rois[duration >= min_length]

            # Remove overlap less than overlap_thresh 
            if len(rois) > 0:
              overlap = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0 / (rois[:,1] - rois[:,0])
              assert min(overlap) >= 0
              assert max(overlap) <= 1
              rois = rois[overlap >= overlap_thresh]

            # Add data
            if len(rois) > 0:
              rois[:,0] = np.maximum(start, rois[:,0])
              rois[:,1] = np.minimum(end, rois[:,1])
              tmp = generate_roi(rois, vid, start, end, stride)
              roidb.append(tmp)
              if USE_FLIPPED:   
                flipped_tmp = copy.deepcopy(tmp)
                flipped_tmp['flipped'] = True
                roidb.append(flipped_tmp)
  return roidb

train_roidb = generate_roidb('train')
print len(train_roidb)


print "Save dictionary"
cPickle.dump(train_roidb, open('train_data_5fps_flip_768.pkl','w'), cPickle.HIGHEST_PROTOCOL)




