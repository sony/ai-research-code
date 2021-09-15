# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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
MSE and SDR Combination loss definition for MSS
'''

import nnabla.functions as F


def mse(x, y):
    # l2 distance and reduce mean
    se = F.squared_error(x, y)
    return F.mean(se)


def unsqueeze(x):
    # unsqueeze at first axis
    return F.reshape(x, (1,) + x.shape)


def mse_loss(mix_spec, msk_hat, gt_spec):
    # MSE-Combination Loss
    # mix_spec     -> (Fbin, BatchSize(16), 2(channels), Frames)
    # msk_hat      -> (Fbin, BatchSize(16), 8(4source x 2channels), Frames)
    #                       channel-dim -> [bass1, bass2, drums1, drums2, ...]
    # gt_spec      -> (Fbin, BatchSize(16), 8(4source x 2channels), Frames)

    assert msk_hat.shape == gt_spec.shape

    m_1 = msk_hat[Ellipsis, 0:2, :]  # bass
    m_2 = msk_hat[Ellipsis, 2:4, :]  # drums
    m_3 = msk_hat[Ellipsis, 4:6, :]  # vocals
    m_4 = msk_hat[Ellipsis, 6:8, :]  # other
    gt_1 = gt_spec[Ellipsis, 0:2, :]  # bass
    gt_2 = gt_spec[Ellipsis, 2:4, :]  # drums
    gt_3 = gt_spec[Ellipsis, 4:6, :]  # vocals
    gt_4 = gt_spec[Ellipsis, 6:8, :]  # other

    # 4C1 Combination Losses
    loss_mse_1 = mse(m_1*mix_spec, gt_1)
    loss_mse_2 = mse(m_2*mix_spec, gt_2)
    loss_mse_3 = mse(m_3*mix_spec, gt_3)
    loss_mse_4 = mse(m_4*mix_spec, gt_4)

    # 4C2 Combination Losses
    loss_mse_5 = mse((m_1+m_2)*mix_spec, (gt_1+gt_2))
    loss_mse_6 = mse((m_1+m_3)*mix_spec, (gt_1+gt_3))
    loss_mse_7 = mse((m_1+m_4)*mix_spec, (gt_1+gt_4))
    loss_mse_8 = mse((m_2+m_3)*mix_spec, (gt_2+gt_3))
    loss_mse_9 = mse((m_2+m_4)*mix_spec, (gt_2+gt_4))
    loss_mse_10 = mse((m_3+m_4)*mix_spec, (gt_3+gt_4))

    # 4C3 Combination Losses
    loss_mse_11 = mse((m_1+m_2+m_3)*mix_spec, (gt_1+gt_2+gt_3))
    loss_mse_12 = mse((m_1+m_2+m_4)*mix_spec, (gt_1+gt_2+gt_4))
    loss_mse_13 = mse((m_1+m_3+m_4)*mix_spec, (gt_1+gt_3+gt_4))
    loss_mse_14 = mse((m_2+m_3+m_4)*mix_spec, (gt_2+gt_3+gt_4))

    # All 14 Combination Losses (4C1 + 4C2 + 4C3)
    loss_mse = (loss_mse_1 + loss_mse_2 + loss_mse_3 + loss_mse_4 + loss_mse_5 + loss_mse_6 + loss_mse_7 +
                loss_mse_8 + loss_mse_9 + loss_mse_10 + loss_mse_11 + loss_mse_12 + loss_mse_13 + loss_mse_14) / 14.0

    return loss_mse


def sdr_loss(mix, pred, gt_time):
    # SDR-Combination Loss
    # mix     -> (BatchSize(16), 2(1 source x 2 channels), TimeLen)  -> (B, C, T)
    # pred    -> (4(sources), Bsize, 2(channels), Len)               -> (S, B, C, T)
    # gt_time -> (BatchSize(16), 8(4 source x 2 channels), TimeLen)  -> (B, S*C, T)
    #                       channel-dim -> [bass1, bass2, drums1, drums2, ...]

    _, batch_size, n_channels, length = pred.shape

    # Fix Length
    mix = mix[Ellipsis, :length]
    gt_time = gt_time[Ellipsis, :length]

    # Fix Shape
    mix = unsqueeze(mix)  # [1, B, C, T]
    gt_time = unsqueeze(gt_time)  # [1, B, S*C, T]
    data_t = mix  # [1, B, C, T]

    for i in range(4):
        data_t = F.concatenate(data_t, gt_time[Ellipsis, 2*i:2*i+2, :], axis=0)

    data_t = F.reshape(data_t, (-1, length))  # [5*B*C, T]
    pred = F.reshape(pred, (batch_size*n_channels *
                            pred.shape[0], pred.shape[-1]))  # [B*C*S, T]

    # Combination List (4C2 + 4C3)
    combi_list = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
                  (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
    for combi in combi_list:
        if len(combi) == 2:
            tmp_data = data_t[batch_size*n_channels*combi[0]:batch_size*n_channels*(
                combi[0]+1), Ellipsis] + data_t[batch_size*n_channels*combi[1]:batch_size*n_channels*(combi[1]+1), Ellipsis]

            tmp_pred = pred[batch_size*n_channels*(combi[0]-1):batch_size*n_channels*combi[0], Ellipsis] + \
                pred[batch_size*n_channels*(
                                                   combi[1]-1):batch_size*n_channels*combi[1], Ellipsis]
        else:
            tmp_data = data_t[batch_size*n_channels*combi[0]:batch_size*n_channels*(combi[0]+1), Ellipsis] + data_t[batch_size*n_channels*combi[1]:batch_size*n_channels*(
                combi[1]+1), Ellipsis] + data_t[batch_size*n_channels*combi[2]:batch_size*n_channels*(combi[2]+1), Ellipsis]

            tmp_pred = pred[batch_size*n_channels*(combi[0]-1):batch_size*n_channels*combi[0], Ellipsis] + pred[batch_size*n_channels*(
                combi[1]-1):batch_size*n_channels*combi[1], Ellipsis] + pred[batch_size*n_channels*(combi[2]-1):batch_size*n_channels*combi[2], Ellipsis]

        data_t = F.concatenate(data_t, tmp_data, axis=0)
        pred = F.concatenate(pred, tmp_pred, axis=0)

    # All 14 Combinations (4C1 + 4C2 + 4C3)
    mix_t = F.tile(data_t[:batch_size*n_channels, Ellipsis], (14, 1))
    data_t = data_t[batch_size*n_channels:, Ellipsis]

    # SDR Loss Calculation
    loss_sdr = sdr_loss_core(pred, data_t, mix_t, weighted=True)

    return 1.0 + loss_sdr


def sdr_loss_core(inp, gt, mix, weighted=True):
    assert inp.shape == gt.shape  # (Batch, Len)
    assert mix.shape == gt.shape   # (Batch, Len)

    inp = inp[:, 200:-200]
    gt = gt[:, 200:-200]
    mix = mix[:, 200:-200]

    ns = mix - gt
    ns_hat = mix - inp

    if weighted:
        alpha = F.sum((gt*gt), 1, keepdims=True) / (F.sum((gt*gt), 1,
                                                          keepdims=True) + F.sum((ns*ns), 1, keepdims=True) + 1e-10)
    else:
        alpha = 0.5

    # Target
    num_cln = F.sum((inp*gt), 1, keepdims=True)
    denom_cln = ((1e-10 + F.sum((inp*inp), 1, keepdims=True))
                 ** 0.5) * ((1e-10 + F.sum((gt*gt), 1, keepdims=True)) ** 0.5)
    sdr_cln = num_cln / (denom_cln + 1e-10)

    # Noise
    num_noise = F.sum((ns*ns_hat), 1, keepdims=True)
    denom_noise = ((1e-10 + F.sum((ns_hat*ns_hat), 1, keepdims=True))
                   ** 0.5) * ((1e-10 + F.sum((ns*ns), 1, keepdims=True)) ** 0.5)
    sdr_noise = num_noise / (denom_noise + 1e-10)

    return F.mean(-alpha*sdr_cln - (1. - alpha)*sdr_noise)
