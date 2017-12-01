# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def twin_overlaps(
        np.ndarray[DTYPE_t, ndim=2] wins,
        np.ndarray[DTYPE_t, ndim=2] query_wins):
    """
    Parameters
    ----------
    wins: (N, 2) ndarray of float
    query_wins: (K, 2) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between wins and query_wins
    """
    cdef unsigned int N = wins.shape[0]
    cdef unsigned int K = query_wins.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t it, ut, win_len
    cdef unsigned int k, n
    for k in range(K):
        win_len = (query_wins[k, 1] - query_wins[k, 0] + 1)
        for n in range(N):
            it = (
                min(wins[n, 1], query_wins[k, 1]) -
                max(wins[n, 0], query_wins[k, 0]) + 1
            )
            if it > 0:
                ut = float((wins[n, 1] - wins[n, 0] + 1) + win_len - it)
                overlaps[n, k] = it / ut
    return overlaps
