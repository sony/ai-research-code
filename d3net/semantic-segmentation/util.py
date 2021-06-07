# Copyright 2021 Sony Group Corporation.
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

'''
Image preprocessing code
'''

import cv2
import numpy as np
import nnabla as nn
from model import d3net_segmentation


def get_segmentation(img, hparams):
    '''
    Encode images with backbone and decode into a semantic segmentation map of the same size as input
    '''
    img_copy = img.copy()

    img = normalize(img, hparams['mean'], hparams['std'])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)

    # Convert numpy to NNabla Variable
    img = nn.Variable.from_numpy_array(img)

    # Get the semantic segmentation map
    seg_map = d3net_segmentation(img, hparams, test=True)

    segmented_img = visualize(img_copy, seg_map, hparams['palette'])
    return segmented_img


def normalize(img, mean, std, to_rgb=True):
    '''
    Normalize the image
    '''
    img = img.copy().astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.subtract(img, mean)
    img = cv2.multiply(img, stdinv)

    return img


def visualize(img_copy, seg_map, palette, opacity=0.5):
    '''
    Draw `result` over `img`
    '''
    palette = np.array(palette)
    assert palette.shape[0] == 19
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0

    seg_map.forward(clear_buffer=True)
    seg_map = seg_map.d.argmax(axis=1)
    seg_map = seg_map.squeeze()
    color_seg = np.zeros(
        (seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)

    for label, color in enumerate(palette):
        color_seg[seg_map == label, :] = color

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    segmented_img = img_copy * (1 - opacity) + color_seg * opacity
    return segmented_img.astype(np.uint8)
