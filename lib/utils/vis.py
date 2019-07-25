# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Detection output visualization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os

# from lib.utils.colormap import colormap
# import lib.utils.env as envu
# import lib.utils.keypoints as keypoint_utils
#
# # Matplotlib requires certain adjustments in some environments
# # Must happen before importing matplotlib
# envu.set_up_matplotlib()
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
#
# plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator
#
#
# _GRAY = (218, 227, 218)
# _GREEN = (18, 127, 15)
# _WHITE = (255, 255, 255)

def vis_preds(img, preds, tp):
    # Draw the detections.
    img = img.copy()
    for idx in range(preds.shape[0]):
        pt = preds[idx, 0].astype(np.int32) * 2, preds[idx, 1].astype(np.int32) * 2

        if tp[idx]:
            cv2.circle(
                img, pt,
                radius=1, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(
                img, pt,
                radius=1, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    return img
