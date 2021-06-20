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
D3Net Semantic Segmentation Inferenece Code
'''

import argparse
import cv2
import yaml
import nnabla as nn
from nnabla.ext_utils import get_extension_context
from util import get_segmentation


def run_segmentation(args):
    '''
    Run D3Net Semantic Segmentation Inferenece
    '''
    # Set NNabla extention
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)

    # Load the model weights
    nn.load_parameters(args.model)

    # Load D3Net Hyper parameters (D3Net-L or D3Net-S)
    with open(args.config_file) as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)

    img = cv2.imread(args.test_image_file, cv2.IMREAD_COLOR)
    segmented_img = get_segmentation(img, hparams)

    cv2.imwrite('./result.jpg', segmented_img)


def get_args(description='D3Net Semantic Segmentation Inference'):
    '''
    Get command line arguments.
    Arguments set the default values of command line arguments.
    '''
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--test-image-file', '-i', type=str,
                        help='path to test image', required=True)
    parser.add_argument('--model', '-m', type=str,
                        default='./D3Net_L.h5', help='Path to model file.')
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension modules('cpu', 'cudnn')")
    parser.add_argument('--config-file', '-cfg', type=str, default='./configs/D3Net_L.yaml',
                        help="Configuration file('D3Net_L.yaml', 'D3Net_S.yaml')")
    return parser.parse_args()


if __name__ == '__main__':
    run_segmentation(get_args())
