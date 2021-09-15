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
Data Augmentation and Pre-Processing code
'''

import cv2
import numpy as np


def random_flip(image, label, prob=None, rng=None):
    '''
    augmentation - flip the input image and label/ground truth image horizontally
    '''
    if rng.rand() < prob:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    return image, label


def random_crop(image, label, crop_size, cat_max_ratio=0.75, ignore_index=255, rng=None):
    '''
    augmentation - random cropping
    '''
    def get_crop_bbox(crop_size, img):
        """Random cropping - bounding box."""
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = rng.randint(0, margin_h + 1)
        offset_w = rng.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        return img[crop_y1:crop_y2, crop_x1:crop_x2, ...]

    crop_bbox = get_crop_bbox(crop_size, image)

    if cat_max_ratio < 1.:
        # Repeat 10 times
        for _ in range(10):
            seg_temp = crop(label, crop_bbox)
            labels, cnt = np.unique(seg_temp, return_counts=True)
            cnt = cnt[labels != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(
                    cnt) < cat_max_ratio:
                break
            crop_bbox = get_crop_bbox(crop_size, image)

    # crop the image
    crop_img = crop(image, crop_bbox)

    # crop the label
    crop_lab = crop(label, crop_bbox)

    return crop_img, crop_lab


def random_scale(image, label, rng=None):
    '''
    augmentation - random upscale/downscale
    '''
    min_scale_factor = 0.5
    max_scale_factor = 2.
    scale = rng.random_sample() * (max_scale_factor - min_scale_factor) + min_scale_factor
    target_h = int(image.shape[0] * scale)
    target_w = 2 * target_h  # For keeping original aspect ratio of 2
    scaled_image = cv2.resize(
        image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    scaled_label = cv2.resize(
        label, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return scaled_image, scaled_label


def standardize(image, to_rgb=True):
    '''
    standardize the image
    '''
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    image = image.copy().astype(np.float32)

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))

    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.subtract(image, mean)
    image = cv2.multiply(image, stdinv)

    return image


def pad(image, label, desired_size=(512, 1024)):
    '''
    image padding to make image size/label size = desired size
    '''
    # mean pixel value
    color = [0, 0, 0]

    old_size = image.shape[:2]  # old_size is in (height, width) format

    new_size = tuple([x + max(ds - x, 0)
                      for x, ds in zip(old_size, desired_size)])
    pad_h = new_size[0] - old_size[0]
    pad_w = new_size[1] - old_size[1]

    # Pad image with mean pixel value
    args = (0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
    new_im = cv2.copyMakeBorder(
        image, *args, value=color)
    label = cv2.copyMakeBorder(label, *args, value=0)
    return new_im, label


def photometric_distortion(image, brightness_delta=32,
                           contrast_range=(0.5, 1.5),
                           saturation_range=(0.5, 1.5),
                           hue_delta=18, rng=None):
    '''
    Apply photometric distortion to image sequentially. Each transformation
    is applied with a probability of 0.5.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    '''

    def convert(image, alpha=1, beta=0):
        '''
        Multiple with alpha and add beat with clip.
        '''
        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

    def contrast(image):
        '''
        Contrast distortion.
        '''
        contrast_lower, contrast_upper = contrast_range
        if rng.randint(2):
            return convert(image, alpha=rng.uniform(contrast_lower, contrast_upper))
        return image

    # Brightness distortion
    if rng.randint(2):
        image = convert(
            image, beta=rng.uniform(-brightness_delta, brightness_delta))

    # mode == 0 --> do random contrast first
    # mode == 1 --> do random contrast last
    mode = rng.randint(2)
    if mode == 1:
        image = contrast(image)

    # Saturation distortion
    saturation_lower, saturation_upper = saturation_range
    if rng.randint(2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 1] = convert(image[:, :, 1],
                                 alpha=rng.uniform(saturation_lower, saturation_upper))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # Hue distortion
    if rng.randint(2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 0] = (image[:, :, 0].astype(
            int) + rng.randint(-hue_delta, hue_delta)) % 180
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    if mode == 0:
        image = contrast(image)

    return image


def create_mask(label):
    '''
    Create masks of 1, 0 for ignore labels (255)
    '''
    mask = (label != 255)
    mask = mask.astype(np.int32)
    label[label == 255] = 0
    label = label.astype(np.int32)
    return label, mask


def preprocess_image_and_label(image, label, rng=None, desired_size=(512, 1024)):
    '''
    Data Augmentation and Pre-Processing
    '''
    if rng is None:
        rng = np.random.RandomState()
    image, label = random_scale(image, label, rng=rng)
    image, label = random_crop(image, label, desired_size, rng=rng)
    image, label = random_flip(image, label, prob=0.5, rng=rng)
    image = photometric_distortion(image, rng=rng)
    image = standardize(image)
    image, label = pad(image, label, desired_size)

    # dimension adjustment and formatting
    if len(image.shape) < 3:
        image = np.expand_dims(image, -1)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    label = np.expand_dims(label, axis=0)
    label, mask = create_mask(label)

    return image, label, mask
