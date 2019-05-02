# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from scipy.optimize import linear_sum_assignment

from lib.core.inference import get_final_preds


def calc_dists(preds, target):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)

    dists = np.sqrt(((preds.reshape(preds.shape[0], 1, preds.shape[1]) - \
        target.reshape(1, target.shape[0], target.shape[1])) ** 2) \
        .sum(axis=-1))

    return dists


def calc_tp_fp_fn(preds, target, hit_thr=2):

    dists = calc_dists(preds, target)

    row_ind, col_ind = linear_sum_assignment(dists)

    tp = np.sum(dists[row_ind, col_ind] <= hit_thr)
    fp = dists.shape[0] - tp
    fn = dists.shape[1] - tp

    return tp, fp, fn
