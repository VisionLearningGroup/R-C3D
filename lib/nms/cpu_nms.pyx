# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 2]

    cdef np.ndarray[np.float32_t, ndim=1] lens = (x2 - x1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, ix2, ilength
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, xx2
    cdef np.float32_t l
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ilength = lens[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(x1[i], x1[j])
            xx2 = min(x2[i], x2[j])
            inter = max(0.0, xx2 - xx1 + 1)
            ovr = inter / (ilength + lens[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep
