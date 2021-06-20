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
D3net Architecture definition.
'''

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I


class D3NetBC():
    '''
    D3Net backbone.

    N.Takahashi, et al. "Densely connected multidilated convolutional networks for dense prediction tasks", CVPR2021
    arXiv: https://arxiv.org/abs/2011.11844
    '''

    def __init__(self,
                 hparams,
                 test=False,
                 recompute=False):
        self.hparams = hparams
        self.test = test
        self.recompute = recompute
        self.x_list = []

    def conv2d(self, conv_input, out_channels, kernel_size, stride, bias=True, name='', dilation=1, pad=0):
        '''
        Define 2D-Convolution Layer
        '''
        conv_out = PF.convolution(conv_input, out_channels, kernel=(kernel_size, kernel_size), stride=(
            stride, stride), with_bias=bias, dilation=(dilation, dilation), pad=(pad, pad), name=name)
        conv_out.apply(recompute=self.recompute)
        return conv_out

    def batch_norm(self, inp, name):
        '''
        Define BatchNormalization Layer
        '''
        out = PF.batch_normalization(inp, batch_stat=not self.test, name=name)
        return out

    def transition(self, inp, out_channels):
        '''
        Transitional Layer Between D3-Blocks
        '''
        out = F.relu(self.batch_norm(inp, name='bn'), inplace=True)
        out = self.conv2d(out, out_channels, kernel_size=1,
                          stride=1, bias=False, name='conv')
        out = F.average_pooling(out, kernel=(2, 2))
        return out

    def bn_conv_block(self, inp, growth_rate, kernel_size=3, dilation=1, pad=1, stride=1):
        '''
        Batch Normalization and Convolution Block
        '''
        with nn.parameter_scope('bottle_neck'):
            # Conv 3x3
            out = self.batch_norm(inp, name='norm1')
            out = F.relu(out, inplace=True)
            out = self.conv2d(out, growth_rate, kernel_size=kernel_size,
                              stride=stride, bias=True, name='conv1', dilation=dilation, pad=pad)
        return out

    def dilated_dense_block(self, inp, in_channels, growth_rate, num_layers, kernel_size=3, out_block=1, dilation=True, bc_ch=None):
        '''
        Dilated Dense Block
        '''
        with nn.parameter_scope('initial_layer'):
            if bc_ch is not None and bc_ch < in_channels:
                with nn.parameter_scope('bc_layer'):
                    out = self.bn_conv_block(
                        inp, bc_ch, dilation=1, kernel_size=1, pad=0)
            else:
                out = inp
            with nn.parameter_scope('init_layer'):
                out = self.bn_conv_block(
                    out, growth_rate*num_layers, dilation=1, kernel_size=kernel_size, pad=1)

        lst = []
        for i in range(num_layers):
            # Split Variable(h) and append them in lst.
            lst.append(out[:, i*growth_rate:(i+1)*growth_rate])

        def update(inp_, n):
            for j in range(num_layers-n-1):
                lst[j+1+n] += inp_[:, j*growth_rate:(j+1)*growth_rate]

        for i in range(num_layers-1):
            d = int(2**(i+1)) if dilation else 1
            with nn.parameter_scope('layers/layer%s' % (i+1)):
                update(self.bn_conv_block(
                    lst[i], growth_rate*(num_layers-i-1), dilation=d, kernel_size=kernel_size, pad=d), i)

        # concatenate the splitted and updated Variables from the lst
        out = F.concatenate(*lst, axis=1)
        return out[:, -out_block*growth_rate:]

    def dilated_dense_block_2(self, inp, in_channels, growth_rate, num_layers, kernel_size=3, out_block=1, dilation=True, block_comp=1, bc_ch=None):
        '''
        Dilated Dense Block-2
        '''
        with nn.parameter_scope('d2block'):
            out = self.dilated_dense_block(inp, in_channels, growth_rate, num_layers,
                                           kernel_size=kernel_size, out_block=out_block, dilation=dilation, bc_ch=bc_ch)

        if block_comp < 1:
            with nn.parameter_scope('block_dim_reduction'):
                out = self.batch_norm(out, name='norm1')
                out = F.relu(out, inplace=True)
                out = self.conv2d(out, int(growth_rate*out_block*block_comp),
                                  kernel_size=1, stride=1, bias=False, name='conv1')
        return out

    def d3_block(self, inp, i, kernel_size=3):
        '''
        Definition of D3 Block
        '''
        num_ch = inp.shape[1]
        growth_rate = self.hparams['dens_k'][i]
        num_layers = self.hparams['num_layers'][i]
        n_blocks = self.hparams['n_blocks'][i]
        out_block = self.hparams['dense_n_out_layer_block'][i]
        dilation = self.hparams['dilation'][i]
        block_comp = self.hparams['block_comp'][i]
        intermediate_out_ch = self.hparams['intermediate_out_ch'][i]

        out = inp
        block_out_size = int(growth_rate*out_block*block_comp)

        with nn.parameter_scope('dense%s' % (i+1)):
            with nn.parameter_scope('d2_block'):
                for j in range(n_blocks):
                    with nn.parameter_scope('block%s' % j):
                        block_out = self.dilated_dense_block_2(
                            out, num_ch, growth_rate, num_layers, kernel_size=kernel_size, out_block=out_block, dilation=dilation, block_comp=block_comp, bc_ch=growth_rate*4)
                    out = F.concatenate(out, block_out, axis=1)
                    num_ch += block_out_size

        if i == 3:
            intermediate_out = self.conv2d(
                out, intermediate_out_ch, kernel_size=1, stride=1, bias=True, name='final_out%s/%s' % (i+1, 0))
            self.x_list.append(F.relu(self.batch_norm(
                intermediate_out, name='final_out%s/%s' % (i+1, 1)), inplace=True))
            return None

        intermediate_out = self.conv2d(out, intermediate_out_ch, kernel_size=1,
                                       stride=1, bias=True, name='intermediate_out%s/%s' % (i+1, 0))
        self.x_list.append(F.relu(self.batch_norm(
            intermediate_out, name='intermediate_out%s/%s' % (i+1, 1)), inplace=True))

        n_feat = int(growth_rate * out_block * block_comp) * \
            n_blocks + inp.shape[1]
        trans_ch = n_feat // self.hparams['trans_comp_factor']

        with nn.parameter_scope('transition%s' % (i+1)):
            out = self.transition(out, trans_ch)
        return out

    def __call__(self, img):
        out = self.conv2d(img, self.hparams['num_init_features'],
                          kernel_size=3, stride=2, bias=False, name='conv1', pad=1)
        out = F.relu(self.batch_norm(out, name='bn1'), inplace=True)
        out = self.conv2d(out, self.hparams['num_init_features'],
                          kernel_size=3, stride=2, bias=False, name='conv2', pad=1)

        # scale 1
        out = self.d3_block(out, i=0)
        # scale 2
        out = self.d3_block(out, i=1)
        # scale 3
        out = self.d3_block(out, i=2)
        # scale 1
        out = self.d3_block(out, i=3)

        return self.x_list


class FCNHead():
    '''
    Fully Convolution Networks for Semantic Segmentation; decodes the extracted features into a semantic segmentation map
    This head is implementation of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    '''

    def __init__(self,
                 hparams,
                 output_size,
                 test=False,
                 recompute=False):
        self.hparams = hparams
        self.output_size = output_size
        self.test = test
        self.recompute = recompute

    def conv2d(self, conv_input, out_channels, kernel_size, stride, bias=True, name='', dilation=1, pad=0):
        '''
        Define simple 2D convolution
        '''
        b_init = I.ConstantInitializer(value=0)
        w_init = I.NormalInitializer(sigma=0.01)
        conv_out = PF.convolution(conv_input, out_channels, kernel=(kernel_size, kernel_size), stride=(stride, stride),
                                  with_bias=bias, dilation=(dilation, dilation), pad=(pad, pad), name=name, w_init=w_init, b_init=b_init)
        conv_out.apply(recompute=self.recompute)
        return conv_out

    def batch_norm(self, inp, name):
        '''
        Define BatchNormalization Layer with weight initialization
        '''
        param_dict = {'beta': I.ConstantInitializer(
            value=0), 'gamma': I.ConstantInitializer(value=1)}
        out = PF.batch_normalization(
            inp, batch_stat=not self.test, name=name, param_init=param_dict)
        return out

    def __call__(self, features):
        upsampled_inputs = [F.interpolate(x, output_size=features[0].shape[2:], mode='linear',
                                          align_corners=False, half_pixel=True) for x in features]
        inputs = F.concatenate(*upsampled_inputs, axis=1)
        out = self.conv2d(inputs, self.hparams['channels'], kernel_size=1,
                          stride=1, bias=False, name='convs/0/conv')
        out = F.relu(self.batch_norm(out, name='convs/0/bn'))
        out = self.conv2d(out, self.hparams['num_classes'], kernel_size=1,
                          stride=1, bias=True, name='conv_seg')
        out = F.interpolate(out, output_size=self.output_size, mode='linear',
                            align_corners=False, half_pixel=True)
        if self.test:
            return F.softmax(out, axis=1)
        return out


def d3net_segmentation(img, hparams, test=False, recompute=False):
    '''
    Get Semantic Segmentation map of img by applying D3Net backbone and Fully Convolutional Network
    '''
    # get the encoded features by D3Net Backbone
    with nn.parameter_scope('backbone'):
        d3net = D3NetBC(hparams, test=test, recompute=recompute)
        features = d3net(img)

    # decode the extracted features into a semantic segmentation map
    with nn.parameter_scope('decode_head'):
        fcn_head = FCNHead(
            hparams, output_size=img.shape[2:], test=test, recompute=recompute)
        seg_map = fcn_head(features)

    return seg_map
