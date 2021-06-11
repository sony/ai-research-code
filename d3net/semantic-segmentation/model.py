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


def conv2d(conv_input, out_channels, kernel_size, stride, bias=True, name='', dilation=1, pad=0):
    '''
    Define simple 2D convolution
    '''
    conv_out = PF.convolution(conv_input, out_channels, kernel=(kernel_size, kernel_size), stride=(stride, stride),
                              with_bias=bias, dilation=(dilation, dilation), pad=(pad, pad), name=name)
    return conv_out


def transition(inp, out_channels, test):
    '''
    Transitional layer between D3 blocks
    '''
    out = F.relu(PF.batch_normalization(inp, batch_stat=not test, name='bn'))
    out = conv2d(out, out_channels, kernel_size=1,
                 stride=1, bias=False, name='conv')
    out = F.average_pooling(out, (2, 2))
    return out


def bn_conv_block(inp, growth_rate, test, kernel_size=3, dilation=1, pad=1, stride=1):
    '''
    Batch normalization and convolution block
    '''
    with nn.parameter_scope('bottle_neck'):
        # Conv 3x3
        out = PF.batch_normalization(inp, batch_stat=not test, name='norm1')
        out = F.relu(out, inplace=True)
        out = conv2d(out, growth_rate, kernel_size=kernel_size, stride=stride,
                     bias=True, name='conv1', dilation=dilation, pad=pad)
    return out


def dilated_dense_block(inp, in_channels, growth_rate, num_layers, test, kernel_size=3,
                        out_block=1, dilation=True, bc_ch=None):
    '''
    Dilated dense block
    '''
    with nn.parameter_scope('initial_layer'):
        if bc_ch is not None and bc_ch < in_channels:
            with nn.parameter_scope('bc_layer'):
                out = bn_conv_block(inp, bc_ch, test, dilation=1, kernel_size=1, pad=0)
        else:
            out = inp
        with nn.parameter_scope('init_layer'):
            out = bn_conv_block(out, growth_rate*num_layers, test, dilation=1, kernel_size=kernel_size, pad=1)

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
            update(bn_conv_block(lst[i], growth_rate*(num_layers-i-1), test,
                                 dilation=d, kernel_size=kernel_size, pad=d), i)

    # concatenate the splitted and updated Variables from the lst
    out = F.concatenate(*lst, axis=1)
    return out[:, -out_block*growth_rate:]


def dilated_dense_block_2(inp, in_channels, growth_rate, num_layers, test, kernel_size=3,
                          out_block=1, dilation=True, block_comp=1, bc_ch=None):
    '''
    Dilated dense block 2
    '''
    with nn.parameter_scope('d2block'):
        out = dilated_dense_block(inp, in_channels, growth_rate, num_layers, test,
                                  kernel_size=kernel_size, out_block=out_block, dilation=dilation, bc_ch=bc_ch)

    if block_comp < 1:
        with nn.parameter_scope('block_dim_reduction'):
            out = PF.batch_normalization(
                out, batch_stat=not test, name='norm1')
            out = F.relu(out, inplace=True)
            out = conv2d(out, int(growth_rate*out_block*block_comp),
                         kernel_size=1, stride=1, bias=False, name='conv1')
    return out


def d3_block(inp, in_channels, growth_rate, num_layers, n_blocks, test, kernel_size=3,
             out_block=2, dilation=True, block_comp=1):
    '''
    Definition of D3 block
    '''
    out = inp
    num_ch = in_channels
    block_out_size = int(growth_rate*out_block*block_comp)
    with nn.parameter_scope('d2_block'):
        for i in range(n_blocks):
            with nn.parameter_scope('block%s' % i):
                block_out = dilated_dense_block_2(out, num_ch, growth_rate, num_layers, test, kernel_size=kernel_size,
                                                  out_block=out_block, dilation=dilation, block_comp=block_comp, bc_ch=growth_rate*4)
            out = F.concatenate(out, block_out, axis=1)
            num_ch += block_out_size
    return out


def d3net(x, hparams, test=False):
    '''
    D3 net architecture definition
    '''
    x = conv2d(x, hparams['num_init_features'], kernel_size=3,
               stride=2, bias=False, name='conv1', pad=1)
    x = F.relu(PF.batch_normalization(
        x, batch_stat=not test, name='bn1'), inplace=True)
    x = conv2d(x, hparams['num_init_features'], kernel_size=3,
               stride=2, bias=False, name='conv2', pad=1)

    x_list = []

    # scale 1
    with nn.parameter_scope('dense1'):
        x = d3_block(x, in_channels=hparams['num_init_features'], n_blocks=hparams['n_blocks'][0], test=test,
                     growth_rate=hparams['dens_k'][0], num_layers=hparams['num_layers'][0],
                     out_block=hparams['dense_n_out_layer_block'][0], dilation=hparams['dilation'][0],
                     block_comp=hparams['block_comp'][0])
    n_feat = int(hparams['dens_k'][0]*hparams['dense_n_out_layer_block'][0]
                 * hparams['block_comp'][0])*hparams['n_blocks'][0] + hparams['num_init_features']

    x1 = conv2d(x, hparams['intermediate_out_ch'][0], kernel_size=1,
                stride=1, bias=True, name='intermediate_out1/0')
    x_list.append(F.relu(PF.batch_normalization(
        x1, batch_stat=not test, name='intermediate_out1/1'), inplace=True))

    trans_ch = n_feat // hparams['trans_comp_factor']
    with nn.parameter_scope('transition1'):
        x = transition(x, trans_ch, test)

    # scale 2
    with nn.parameter_scope('dense2'):
        x = d3_block(x, in_channels=trans_ch, n_blocks=hparams['n_blocks'][1], test=test,
                     growth_rate=hparams['dens_k'][1], num_layers=hparams['num_layers'][1],
                     out_block=hparams['dense_n_out_layer_block'][1], dilation=hparams['dilation'][1],
                     block_comp=hparams['block_comp'][1])
    n_feat = int(hparams['dens_k'][1]*hparams['dense_n_out_layer_block'][1] *
                 hparams['block_comp'][1])*hparams['n_blocks'][1] + trans_ch

    x2 = conv2d(x, hparams['intermediate_out_ch'][1], kernel_size=1,
                stride=1, bias=True, name='intermediate_out2/0')
    x_list.append(F.relu(PF.batch_normalization(
        x2, batch_stat=not test, name='intermediate_out2/1'), inplace=True))

    trans_ch = n_feat // hparams['trans_comp_factor']
    with nn.parameter_scope('transition2'):
        x = transition(x, trans_ch, test)

    # scale 3
    with nn.parameter_scope('dense3'):
        x = d3_block(x, in_channels=trans_ch, n_blocks=hparams['n_blocks'][2], test=test,
                     growth_rate=hparams['dens_k'][2], num_layers=hparams['num_layers'][2],
                     out_block=hparams['dense_n_out_layer_block'][2], dilation=hparams['dilation'][2],
                     block_comp=hparams['block_comp'][2])

    n_feat = int(hparams['dens_k'][2]*hparams['dense_n_out_layer_block'][2] *
                 hparams['block_comp'][2])*hparams['n_blocks'][2] + trans_ch

    x3 = conv2d(x, hparams['intermediate_out_ch'][2], kernel_size=1,
                stride=1, bias=True, name='intermediate_out3/0')
    x_list.append(F.relu(PF.batch_normalization(
        x3, batch_stat=not test, name='intermediate_out3/1'), inplace=True))

    trans_ch = n_feat // hparams['trans_comp_factor']
    with nn.parameter_scope('transition3'):
        x = transition(x, trans_ch, test)

    # scale 1
    with nn.parameter_scope('dense4'):
        x = d3_block(x, in_channels=trans_ch, n_blocks=hparams['n_blocks'][3], test=test,
                     growth_rate=hparams['dens_k'][3], num_layers=hparams['num_layers'][3],
                     out_block=hparams['dense_n_out_layer_block'][3], dilation=hparams['dilation'][3],
                     block_comp=hparams['block_comp'][3])

    n_feat = int(hparams['dens_k'][3]*hparams['dense_n_out_layer_block'][3] *
                 hparams['block_comp'][3])*hparams['n_blocks'][3] + trans_ch

    x4 = conv2d(x, hparams['intermediate_out_ch'][3],
                kernel_size=1, stride=1, bias=True, name='final_out4/0')
    x_list.append(F.relu(PF.batch_normalization(
        x4, batch_stat=not test, name='final_out4/1'), inplace=True))

    return x_list


def fcn_head(features, hparams, output_size, test=False):
    '''
    Fully Convolution Networks for Semantic Segmentation; decodes the extracted features into a semantic segmentation map
    '''
    upsampled_inputs = [F.interpolate(x, output_size=features[0].shape[2:], mode='linear',
                                      align_corners=hparams['align_corners'], half_pixel=True) for x in features]
    inputs = F.concatenate(*upsampled_inputs, axis=1)
    out = conv2d(inputs, hparams['channels'], kernel_size=1,
                 stride=1, bias=False, name='convs/0/conv')
    out = F.relu(PF.batch_normalization(
        out, batch_stat=not test, name='convs/0/bn'))
    out = conv2d(out, hparams['num_classes'], kernel_size=1,
                 stride=1, bias=True, name='conv_seg')
    out = F.interpolate(out, output_size=output_size, mode='linear',
                        align_corners=hparams['align_corners'], half_pixel=True)
    out = F.softmax(out, axis=1)
    return out


def d3net_segmentation(img, hparams, test=False):
    '''
    Get Semantic Segmentation map of img by applying D3Net and Fully Convolution Networks
    '''
    # get the encoded features by D3Net Backbone
    with nn.parameter_scope('backbone'):
        features = d3net(img, hparams, test=test)

    # decode the extracted features into a semantic segmentation map
    with nn.parameter_scope('decode_head'):
        seg_map = fcn_head(features, hparams,
                           output_size=img.shape[2:], test=test)
    return seg_map
