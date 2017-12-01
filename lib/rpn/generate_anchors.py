# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import numpy as np

def generate_anchors(base_size=8, scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect 
    scales wrt a reference (0, 7) window.
    """

    base_anchor = np.array([1, base_size]) - 1
    anchors = _scale_enum(base_anchor, scales)
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    l = anchor[1] - anchor[0] + 1
    x_ctr = anchor[0] + 0.5 * (l - 1)
    return l, x_ctr 

def _mkanchors(ls, x_ctr):
    """
    Given a vector of lengths (ls) around a center
    (x_ctr), output a set of anchors (windows).
    """

    ls = ls[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ls - 1),
                         x_ctr + 0.5 * (ls - 1)))
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    l, x_ctr = _whctrs(anchor)
    ls = l * scales
    anchors = _mkanchors(ls, x_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    from IPython import embed; embed()
