# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

"""Test a R-C3D network."""

from tdcnn.config import cfg
from tdcnn.twin_transform import clip_wins, twin_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from tdcnn.nms_wrapper import nms
import cPickle
from utils.blob import video_list_to_blob, prep_im_for_blob
import os

def _get_video_blob(roidb):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """

    processed_videos = []

    item = roidb

    for key in item:
      print key, ": ", item[key]
    video_length = cfg.TEST.LENGTH[0]
    video = np.zeros((video_length, cfg.TEST.CROP_SIZE,
                      cfg.TEST.CROP_SIZE, 3))

    j = 0
    random_idx = [int(cfg.TEST.FRAME_SIZE[1]-cfg.TEST.CROP_SIZE) / 2,
                  int(cfg.TEST.FRAME_SIZE[0]-cfg.TEST.CROP_SIZE) / 2]

    if cfg.INPUT == 'video':
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

    else:
      for video_info in item['frames']:
        prefix = item['fg_name'] if video_info[0] else item['bg_name']
        for idx in xrange(video_info[1], video_info[2]):
          frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))
          frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TEST.FRAME_SIZE[::-1]),
                                   cfg.TEST.CROP_SIZE, random_idx)

          if item['flipped']:
              frame = frame[:, ::-1, :]

          video[j] = frame
          j = j + 1

    while ( j < video_length):
      video[j] = frame
      j = j + 1
    processed_videos.append(video)

    # Create a blob to hold the input images
    blob = video_list_to_blob(processed_videos)

    return blob

def _get_blobs(video, rois = None):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'] = video
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs


def video_detect(net, video, wins=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        wins (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        wins (ndarray): R x (4*K) array of predicted bounding wins
    """
    blobs = _get_blobs(video)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:   #no use
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        wins = wins[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if not cfg.TEST.HAS_RPN:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if not cfg.TEST.HAS_RPN:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert net.blobs['data'].shape[0] == 1, "Only single-image batch implemented"
        rois = net.blobs['rpn_rois'].data.copy()
        # unscale back to raw image space
        wins = rois[:, 1:3]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.TWIN_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['twin_pred']
        pred_wins = twin_transform_inv(wins, box_deltas)
        pred_wins = clip_wins(pred_wins, video.shape[2])
    else:
        # Simply repeat the wins, once for each class
        pred_wins = np.tile(wins, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of wins
        scores = scores[inv_index, :]
        pred_wins = pred_wins[inv_index, :]

    return scores, pred_wins

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        twin = dets[i, :2]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((twin[0], twin[1]),
                              twin[2] - twin[0],
                              twin[3] - twin[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_wins, thresh):
    """Apply non-maximum suppression to all predicted wins output by the
    test_net method.
    """
    num_classes = len(all_wins)
    num_images = len(all_wins[0])
    nms_wins = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_wins[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of wins
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_wins[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_wins

def test_net(net, roidb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_videos = len(roidb)
    # all detections are collected into:
    #    all_wins[cls][image] = N x 2 array of detections in
    #    (x1, x2, score)
    all_wins = [[[] for _ in xrange(num_videos)]
                 for _ in xrange(cfg.NUM_CLASSES)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(num_videos):
        # filter out any ground truth wins
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['wins'][roidb[i]['gt_classes'] == 0]

        video = _get_video_blob(roidb[i])
        _t['im_detect'].tic()
        scores, wins = video_detect(net, video, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, cfg.NUM_CLASSES):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_wins = wins[inds, j*2:(j+1)*2]
            cls_dets = np.hstack((cls_wins, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)  #0.3
            if len(keep) != 0:
              cls_dets = cls_dets[keep, :]
              print "activity: ", j
              print cls_dets
            all_wins[j][i] = cls_dets

        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_videos, _t['im_detect'].average_time,
                      _t['misc'].average_time)
