# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import cv2

import numpy as np

def get_final_preds(batch_heatmaps, score_thresh=0.5):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    assert batch_heatmaps.shape[1] == 1, 'batch_images must be single channel'

    batch_size = batch_heatmaps.shape[0]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]

    # local_max_heatmaps = np.zeros((batch_size, height, width, 1), dtype=float32)
    batch_preds = []

    for batch_idx in range(batch_size):
        heatmaps = batch_heatmaps[batch_idx]
        heatmaps = heatmaps.squeeze()

        heatmaps_padded = cv2.copyMakeBorder(heatmaps, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

        local_max = np.ones(heatmaps.shape, dtype=np.bool)

        for n_idx in range(9):
            if n_idx == 4:
                continue

            neighbors = heatmaps_padded[
                (n_idx % 3):(n_idx % 3 + height),
                (n_idx // 3):(n_idx // 3 + width)
            ]

            local_max = np.logical_and(
                local_max,
                neighbors <= heatmaps
            )

        # heatmaps[np.logical_not(local_max)] = 0.0

        # local_max_heatmaps[batch_idx] = heatmaps

        rows, cols = np.where(np.logical_and(local_max, heatmaps > score_thresh))

        scores = heatmaps[rows, cols]

        preds = np.vstack((cols, rows, scores)).transpose()

        batch_preds.append(preds)


    return batch_preds
