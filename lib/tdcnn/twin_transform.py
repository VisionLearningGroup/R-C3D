# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import numpy as np

def twin_transform(ex_rois, gt_rois):
    ex_lengths = ex_rois[:, 1] - ex_rois[:, 0] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_lengths

    gt_lengths = gt_rois[:, 1] - gt_rois[:, 0] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_lengths

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_lengths
    targets_dl = np.log(gt_lengths / ex_lengths)

    targets = np.vstack(
        (targets_dx, targets_dl)).transpose()
    return targets

def twin_transform_inv(wins, deltas):
    if wins.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    wins = wins.astype(deltas.dtype, copy=False)

    lengths = wins[:, 1] - wins[:, 0] + 1.0
    ctr_x = wins[:, 0] + 0.5 * lengths

    dx = deltas[:, 0::2]
    dl = deltas[:, 1::2]

    pred_ctr_x = dx * lengths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_l = np.exp(dl) * lengths[:, np.newaxis]

    pred_wins = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_wins[:, 0::2] = pred_ctr_x - 0.5 * pred_l
    # x2
    pred_wins[:, 1::2] = pred_ctr_x + 0.5 * pred_l

    return pred_wins

def clip_wins(wins, video_length):
    """
    Clip wins to video boundaries.
    """

    # x1 >= 0
    wins[:, 0::2] = np.maximum(np.minimum(wins[:, 0::2], video_length - 1), 0)
    # x2 < im_shape[1]
    wins[:, 1::2] = np.maximum(np.minimum(wins[:, 1::2], video_length - 1), 0)
    return wins
