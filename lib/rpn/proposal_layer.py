# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from tdcnn.config import cfg
from generate_anchors import generate_anchors
from tdcnn.twin_transform import twin_transform_inv, clip_wins
from tdcnn.nms_wrapper import nms

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular wins (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', tuple(range(1,64)))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 3-tuple
        # (n, x1, x2) specifying an video batch index n and a
        # rectangle (x1, x2)
        top[0].reshape(1, 3)  

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor wins centered on cell i
        #   apply predicted twin deltas at cell i to each of the A anchors
        # clip predicted wins to video
        # remove predicted wins with length < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase)
        cfg_key = 'TRAIN' if self.phase == 0 else 'TEST'
        print cfg_key
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :, :]  
        twin_deltas = bottom[1].data

        # 1. Generate proposals from twin deltas and shifted anchors
        length, height, width = scores.shape[-3:]

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shifts = np.arange(0, length) * self._feat_stride

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 2) to
        # cell K shifts (K, 1, 2) to get
        # shift anchors (K, A, 2)
        # reshape to (K*A, 2) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 2)) + \
                  shifts.reshape((1, K, 1)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 2))

        # Transpose and reshape predicted twin transformations to get them
        # into the same order as the anchors:
        #
        # twin deltas will be (1, 2 * A, L, H, W) format
        # transpose to (1, L, H, W, 2 * A)
        # reshape to (1 * L * H * W * A, 2) where rows are ordered by (l, h, w, a)
        # in slowest to fastest order
        twin_deltas = twin_deltas.transpose((0, 2, 3, 4, 1)).reshape((-1, 2))

        # Same story for the scores:
        #
        # scores are (1, A, L, H, W) format
        # transpose to (1, L, H, W, A)
        # reshape to (1 * L, H * W * A, 1) where rows are ordered by (l, h, w, a)
        scores = scores.transpose((0, 2, 3, 4, 1)).reshape((-1, 1))

        # Convert anchors into proposals via twin transformations
        proposals = twin_transform_inv(anchors, twin_deltas)

        # 2. clip predicted wins to video
        proposals = clip_wins(proposals, length * self._feat_stride)

        # 3. remove predicted wins with either height or width < threshold
        # (NOTE: convert min_size to input video scale stored in im_info[2])
        keep = _filter_wins(proposals, min_size)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input video, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_wins(wins, min_size):
    """Remove all wins with any side smaller than min_size."""
    ls = wins[:, 1] - wins[:, 0] + 1
    keep = np.where(ls >= min_size)[0]
    return keep
