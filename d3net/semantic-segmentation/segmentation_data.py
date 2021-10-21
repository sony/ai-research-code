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
CityScapes Segmentation data-iterator code.
'''

import os
import numpy as np
import cv2
import nnabla as nn
from nnabla.utils.data_iterator import data_iterator_simple
from nnabla.utils.image_utils import imread
import image_preprocess


class CityScapesDatasetPath(object):
    '''
    A Helper Class which resolves the path to images
    in CityScapes dataset.
    '''

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.val_file = os.path.join(self.data_dir, 'val.txt')

    def get_image_path(self, name, train):
        folder = 'train' if train else 'val'
        return os.path.join(self.data_dir, 'leftImg8bit', folder, name + '_leftImg8bit.png')

    def get_label_path(self, name, train):
        folder = 'train' if train else 'val'
        return os.path.join(self.data_dir, 'gtFine', folder, name + '_gtFine_labelTrainIds.png')

    def get_image_paths(self, train=True):
        file_name = self.train_file if train else self.val_file
        names = np.loadtxt(file_name, dtype=str)
        return [self.get_image_path(name, train) for name in names]

    def get_label_paths(self, train=True):
        file_name = self.train_file if train else self.val_file
        names = np.loadtxt(file_name, dtype=str)
        return [self.get_label_path(name, train) for name in names]


def palette_png_reader(fname):
    '''
    '''
    assert 'PilBackend' in nn.utils.image_utils.get_available_backends()
    if nn.utils.image_utils.get_backend() != 'PilBackend':
        nn.utils.image_utils.set_backend("PilBackEnd")
    return imread(fname, return_palette_indices=True)


def data_iterator_segmentation(batch_size, image_paths, label_paths, rng=None, train=True):
    '''
    Returns a data iterator object for semantic image segmentation dataset.

    Args:
        batch_size (int): Batch size
        image_paths (list of str): A list of image paths
        label_paths (list of str): A list of label image paths
        rng (None or numpy.random.RandomState):
            A random number generator used in shuffling dataset and data augmentation.
        train (bool): It performs random data augmentation as preprocessing if train is True.
        num_classs (int): Number of classes. Requierd if `label_mask_transformer` is not passed.
    '''
    assert len(image_paths) == len(label_paths)
    num_examples = len(image_paths)

    def image_label_load_func(i):
        '''
        Returns:
            image: c x h x w array
            label: c x h x w array
        '''
        img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        lab = palette_png_reader(label_paths[i])
        img, lab = image_preprocess.preprocess_image_and_label(
            img, lab, rng=rng)
        return img, lab

    return data_iterator_simple(image_label_load_func, num_examples, batch_size, shuffle=train, rng=rng)


def data_iterator_cityscapes(batch_size, data_dir, rng=None, train=True):
    '''
    Returns a data iterator object for CityScapes segmentation dataset.

    args:
        data_dir (str):
            A folder contains CityScapes dataset.

    See `data_iterator_segmentation` for other arguments.

    '''

    cityscapes = CityScapesDatasetPath(data_dir)
    image_paths = cityscapes.get_image_paths(train=train)
    label_paths = cityscapes.get_label_paths(train=train)

    return data_iterator_segmentation(batch_size, image_paths, label_paths, rng, train)
