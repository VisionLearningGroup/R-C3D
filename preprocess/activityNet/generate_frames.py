# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
from util import *
import json

fps = 25

VIDEO_DIR = './videos/'
video_list = os.listdir(VIDEO_DIR)

META_FILE = './activity_net.v1-3.min.json'
meta_data = json.load(open(META_FILE))

mkdir('./frames')
def generate_frame(split):
  mkdir('./frames/%s' % split)
  for vid, vinfo in meta_data['database'].items():
    if vinfo['subset'] == split:
      vname = [s for s in video_list if vid in s]
      if len(vname) != 0 :
        filename = VIDEO_DIR + vname[0]
        duration = vinfo['duration']
        outpath = './frames/%s/%s/' % (split, vid)
        outfile = outpath + "image_%5d.jpg"
        rm(outpath)
        mkdir(outpath)
        ffmpeg(filename, outfile, fps)
        for framename in os.listdir(outpath):
          resize(outpath + framename)
        frame_size = len(os.listdir(outpath))
        # print filename, duration, fps, frame_size

generate_frame('training')
generate_frame('validation')
#generate_frame('testing')
