# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from tdcnn.config import cfg
from tdcnn.twin_transform import twin_transform
from utils.cython_twin import twin_overlaps

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']
        self._sample = layer_params.get('sample', 'Random')

        if DEBUG:
            self._fg_num = 0
            self._bg_num = 0
            self._count = 0

        # sampled rois (0, x1, x2)
        top[0].reshape(1, 3)
        # labels
        top[1].reshape(1, 1)
        # twin_targets
        top[2].reshape(1, self._num_classes * 2)
        # twin_inside_weights
        top[3].reshape(1, self._num_classes * 2)
        # twin_outside_weights
        top[4].reshape(1, self._num_classes * 2)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, x2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT wins (x1, x2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_wins = bottom[1].data
        # loss
        if self._sample == 'Hard':
          loss = bottom[2].data

        # Include ground-truth wins in the set of candidate rois
        zeros = np.zeros((gt_wins.shape[0], 1), dtype=gt_wins.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_wins[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'


        if self._sample == "All":
          labels, rois, twin_targets, twin_inside_weights = _sample_all_rois(
              all_rois, gt_wins, self._num_classes)
        elif self._sample == "Hard":
          labels, rois, twin_targets, twin_inside_weights = _sample_hard_rois(
              all_rois, gt_wins, loss, self._num_classes)
        else:
          # Sample rois with classification labels and bounding box regression
          # targets
          num_images = 1
          rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
          fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))
          labels, rois, twin_targets, twin_inside_weights = _sample_rois(
              all_rois, gt_wins, fg_rois_per_image,
              rois_per_image, self._num_classes)
        

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # twin_targets
        top[2].reshape(*twin_targets.shape)
        top[2].data[...] = twin_targets

        # twin_inside_weights
        top[3].reshape(*twin_inside_weights.shape)
        top[3].data[...] = twin_inside_weights

        # twin_outside_weights
        top[4].reshape(*twin_inside_weights.shape)
        top[4].data[...] = np.array(twin_inside_weights > 0).astype(np.float32)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_twin_regression_labels(twin_target_data, num_classes):
    """Bounding-box regression targets (twin_target_data) are stored in a
    compact form N x (class, tx, tl)

    This function expands those targets into the 4-of-2*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        twin_target (ndarray): N x 4K blob of regression targets
        twin_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = twin_target_data[:, 0]
    twin_targets = np.zeros((clss.size, 2 * num_classes), dtype=np.float32)
    twin_inside_weights = np.zeros(twin_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(2 * cls)
        end = start + 2
        twin_targets[ind, start:end] = twin_target_data[ind, 1:]
        twin_inside_weights[ind, start:end] = cfg.TRAIN.TWIN_INSIDE_WEIGHTS
    return twin_targets, twin_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 2
    assert gt_rois.shape[1] == 2

    targets = twin_transform(ex_rois, gt_rois)
    if cfg.TRAIN.TWIN_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.TWIN_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.TWIN_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_wins, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_wins)
    overlaps = twin_overlaps(
        np.ascontiguousarray(all_rois[:, 1:3], dtype=np.float),
        np.ascontiguousarray(gt_wins[:, :2], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_wins[gt_assignment, 2]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:  
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0   
    rois = all_rois[keep_inds]

    twin_target_data = _compute_targets(
        rois[:, 1:3], gt_wins[gt_assignment[keep_inds], :2], labels)

    twin_targets, twin_inside_weights = \
        _get_twin_regression_labels(twin_target_data, num_classes)

    return labels, rois, twin_targets, twin_inside_weights

def _sample_all_rois(all_rois, gt_wins, num_classes):
    """Generate all RoIs comprising foreground and background examples.
    """
    # overlaps: (rois x gt_wins)
    overlaps = twin_overlaps(
        np.ascontiguousarray(all_rois[:, 1:3], dtype=np.float),
        np.ascontiguousarray(gt_wins[:, :2], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_wins[gt_assignment, 2]

    labels = labels
    rois = all_rois

    twin_target_data = _compute_targets(
        rois[:, 1:3], gt_wins[gt_assignment, :2], labels)

    twin_targets, twin_inside_weights = \
        _get_twin_regression_labels(twin_target_data, num_classes)

    return labels, rois, twin_targets, twin_inside_weights

def _sample_hard_rois(all_rois, gt_wins, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a hard sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_wins)
    overlaps = twin_overlaps(
        np.ascontiguousarray(all_rois[:, 1:3], dtype=np.float),
        np.ascontiguousarray(gt_wins[:, :2], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_wins[gt_assignment, 2]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    twin_target_data = _compute_targets(
        rois[:, 1:3], gt_wins[gt_assignment[keep_inds], :2], labels)

    twin_targets, twin_inside_weights = \
        _get_twin_regression_labels(twin_target_data, num_classes)

    return labels, rois, twin_targets, twin_inside_weights

