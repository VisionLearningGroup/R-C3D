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

def generate_roi( vid, start, end, stride):
  tmp = {}
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
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
      folder = PATH + vid + '/'
      length = len(os.listdir(folder))

      for win in WINS:
        stride = win / LENGTH
        step = stride * STEP
        # Forward Direction
        for start in xrange(0, max(1, length - win + 1), step):
          end = min(start + win, length)
          assert end <= length

          # Add data
          tmp = generate_roi(vid, start, end, stride)
          roidb.append(tmp)
          if USE_FLIPPED:   
            flipped_tmp = copy.deepcopy(tmp)
            flipped_tmp['flipped'] = True
            roidb.append(flipped_tmp)

        # Backward Direction
        for end in xrange(length, win-1, - step):
          start = end - win
          assert start >= 0

          # Add data
          tmp = generate_roi(vid, start, end, stride)
          roidb.append(tmp)
          if USE_FLIPPED:   
            flipped_tmp = copy.deepcopy(tmp)
            flipped_tmp['flipped'] = True
            roidb.append(flipped_tmp)
  return roidb

test_roidb = generate_roidb('test')
print len(test_roidb)

print "Save dictionary"
cPickle.dump(test_roidb, open('test_data_5fps_768.pkl','w'), cPickle.HIGHEST_PROTOCOL)



