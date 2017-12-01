# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
import caffe
import yaml
from tdcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_twin import twin_overlaps
from tdcnn.twin_transform import twin_transform

DEBUG = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and time window regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        anchor_scales = layer_params.get('scales', (2,4,5,6,8,9,10,12,14,16))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 1::2] - self._anchors[:, 0::2],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 2))
            self._squared_sums = np.zeros((1, 2))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        length, height, width = bottom[0].data.shape[-3:]
        if DEBUG:
            print 'AnchorTargetLayer: length', length

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * length, height, width)
        # twin_targets
        top[1].reshape(1, A * 2, length, height, width)
        # twin_inside_weights
        top[2].reshape(1, A * 2, length, height, width)
        # twin_outside_weights
        top[3].reshape(1, A * 2, length, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted twin deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        length, height, width = bottom[0].data.shape[-3:]
        # GT boxes (x1, x2, label)
        gt_boxes = bottom[1].data

        if DEBUG:
            print ''
            print 'length, height, width: ({}, {}, {})'.format(length, height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from twin deltas and shifted anchors
        shifts = np.arange(0, length) * self._feat_stride
        # add A anchors (1, A, 2) to
        # cell K shifts (K, 1, 2) to get
        # shift anchors (K, A, 2)
        # reshape to (K*A, 2) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 2)) +
                       shifts.reshape((1, K, 1)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 2))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] < bottom[2].data.shape[2] + self._allowed_border)
        )[0]

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape
            print 'anchors', anchors

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = twin_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if DEBUG:
          print "max_overlaps", max_overlaps
          print "gt_max_overlaps", gt_max_overlaps

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
          # assign bg labels first so that positive labels can clobber them
          labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
          # assign bg labels last so that negative labels can clobber positives
          labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))

        twin_targets = np.zeros((len(inds_inside), 2), dtype=np.float32)
        twin_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        if DEBUG:
            print "twin_targets", twin_targets

        twin_inside_weights = np.zeros((len(inds_inside), 2), dtype=np.float32)
        twin_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_TWIN_INSIDE_WEIGHTS)

        twin_outside_weights = np.zeros((len(inds_inside), 2), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 2)) * 1.0 / num_examples
            negative_weights = np.ones((1, 2)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        twin_outside_weights[labels == 1, :] = positive_weights
        twin_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += twin_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (twin_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        twin_targets = _unmap(twin_targets, total_anchors, inds_inside, fill=0)
        twin_inside_weights = _unmap(twin_inside_weights, total_anchors, inds_inside, fill=0)
        twin_outside_weights = _unmap(twin_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)

        # labels
        labels = labels.reshape((1, length, height, width, A)).transpose(0, 4, 1, 2, 3)
        labels = labels.reshape((1, 1, A * length, height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # twin_targets
        twin_targets = twin_targets \
            .reshape((1, length, height, width, A * 2)).transpose(0, 4, 1, 2, 3)
        top[1].reshape(*twin_targets.shape)
        top[1].data[...] = twin_targets

        # twin_inside_weights
        twin_inside_weights = twin_inside_weights \
            .reshape((1, length, height, width, A * 2)).transpose(0, 4, 1, 2, 3)
        assert twin_inside_weights.shape[3] == height
        assert twin_inside_weights.shape[4] == width
        top[2].reshape(*twin_inside_weights.shape)
        top[2].data[...] = twin_inside_weights

        # twin_outside_weights
        twin_outside_weights = twin_outside_weights \
            .reshape((1, length, height, width, A * 2)).transpose(0, 4, 1, 2, 3)
        assert twin_outside_weights.shape[3] == height
        assert twin_outside_weights.shape[4] == width
        top[3].reshape(*twin_outside_weights.shape)
        top[3].data[...] = twin_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 2
    assert gt_rois.shape[1] == 3

    return twin_transform(ex_rois, gt_rois[:, :2]).astype(np.float32, copy=False)
