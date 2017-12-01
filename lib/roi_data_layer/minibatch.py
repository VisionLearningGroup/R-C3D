# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from tdcnn.config import cfg
from utils.blob import prep_im_for_blob, video_list_to_blob

DEBUG = False

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_videos = len(roidb)
    # Sample random scales to use for each video in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.LENGTH),
                                    size=num_videos)
    assert(cfg.TRAIN.BATCH_SIZE % num_videos == 0), \
        'num_videos ({}) must divide BATCH_SIZE ({})'. \
        format(num_videos, cfg.TRAIN.BATCH_SIZE)
    rois_per_video = cfg.TRAIN.BATCH_SIZE / num_videos
    fg_rois_per_video = np.round(cfg.TRAIN.FG_FRACTION * rois_per_video)

    # Get the input video blob, formatted for caffe
    video_blob = _get_video_blob(roidb, random_scale_inds)

    blobs = {'data': video_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(roidb) == 1, "Single batch only"
        # gt windows: (x1, x2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_windows = np.empty((len(gt_inds), 3), dtype=np.float32)
        gt_windows[:, 0:2] = roidb[0]['wins'][gt_inds, :]
        gt_windows[:, -1] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_windows'] = gt_windows
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 3), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 2 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        raise NotImplementedError

    return blobs

def _get_video_blob(roidb, scale_inds):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """
    processed_videos = []
    video_scales = []
    for i,item in enumerate(roidb):
      video_length = cfg.TRAIN.LENGTH[scale_inds[0]]  
      video = np.zeros((video_length, cfg.TRAIN.CROP_SIZE,
                        cfg.TRAIN.CROP_SIZE, 3))
      if cfg.INPUT == 'video':
        j = 0
        random_idx = [np.random.randint(cfg.TRAIN.FRAME_SIZE[1]-cfg.TRAIN.CROP_SIZE),
                      np.random.randint(cfg.TRAIN.FRAME_SIZE[0]-cfg.TRAIN.CROP_SIZE)]
        for video_info in item['frames']:
          prefix = item['fg_name'] if video_info[0] else item['bg_name']
          for idx in xrange(video_info[1], video_info[2], video_info[3]):
            frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))
            frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]),
                                     cfg.TRAIN.CROP_SIZE, random_idx)

            if item['flipped']:
                frame = frame[:, ::-1, :]

            video[j] = frame
            j = j + 1

        while ( j < video_length):
          video[j] = frame
          j = j + 1

      else:
        j = 0
        random_idx = [np.random.randint(cfg.TRAIN.FRAME_SIZE[1]-cfg.TRAIN.CROP_SIZE),
                      np.random.randint(cfg.TRAIN.FRAME_SIZE[0]-cfg.TRAIN.CROP_SIZE)]
        for video_info in item['frames']:
          prefix = item['fg_name'] if video_info[0] else item['bg_name']
          for idx in xrange(video_info[1], video_info[2]):
            frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))
            frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]),
                                     cfg.TRAIN.CROP_SIZE, random_idx)

            if item['flipped']:
                frame = frame[:, ::-1, :]

            if DEBUG:
              cv2.imshow('frame', frame/255.0)
              cv2.waitKey(0)
              cv2.destroyAllWindows()

            video[j] = frame
            j = j + 1

        while ( j <= video_length):
          video[j] = frame
          j = j + 1
      processed_videos.append(video)

    # Create a blob to hold the input images
    blob = video_list_to_blob(processed_videos)

    return blob
