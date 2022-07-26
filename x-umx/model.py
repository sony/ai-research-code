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
X-UMX/UMX Architecture definition for MSS.
'''

import collections
import nnabla as nn
from nnabla.parameter import get_parameter_or_create
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
from loss import mse_loss, sdr_loss


def get_stft(x, n_fft=4096, n_hop=1024, center=True):
    '''
    Multichannel STFT

    Input: (nb_samples, nb_channels, nb_timesteps)
    Output: (nb_samples, nb_channels, nb_bins, nb_frames), 
            (nb_samples, nb_channels, nb_bins, nb_frames)
    '''
    nb_samples, nb_channels, _ = x.shape
    x = F.reshape(x, (nb_samples*nb_channels, -1))

    real, imag = F.stft(
        x, n_fft, n_hop, n_fft,
        window_type='hanning',
        center=center,
        pad_mode='reflect'
    )

    real = F.reshape(real,
                     (nb_samples, nb_channels, n_fft // 2 + 1, -1)
                     )

    imag = F.reshape(imag,
                     (nb_samples, nb_channels, n_fft // 2 + 1, -1)
                     )

    return real, imag


def get_spectogram(real, imag, power=1, mono=True):
    '''
    Input:  (nb_samples, nb_channels, nb_bins, nb_frames), 
            (nb_samples, nb_channels, nb_bins, nb_frames)
    Output: (nb_frames, nb_samples, nb_channels, nb_bins)
    '''
    spec = ((real ** 2) + (imag ** 2)) ** (power / 2.0)

    if mono:
        spec = F.mean(spec, axis=1, keepdims=True)

    return F.transpose(spec, ((3, 0, 1, 2)))


class BaseClass():
    def __init__(
        self,
        test=False,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        nb_layers=3,
        nb_of_directions=2,
        input_mean=None,
        input_scale=None,
        max_bin=None,
    ):
        self.test = test
        self.nb_output_bins = n_fft // 2 + 1
        self.input_is_spectrogram = input_is_spectrogram
        self.hidden_size = hidden_size
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.nb_of_directions = nb_of_directions
        self.input_mean = input_mean
        self.input_scale = input_scale

        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        if input_mean is None:
            self.input_mean = np.zeros((self.nb_bins,))
        if input_scale is None:
            self.input_scale = np.ones((self.nb_bins,))

    def create_input_mean_parameters(self, name):
        param = get_parameter_or_create(name, shape=(
            self.nb_bins,), initializer=-self.input_mean[:self.nb_bins])
        return F.reshape(param, (1, 1, 1, self.nb_bins))

    def create_input_scale_parameters(self, name):
        param = get_parameter_or_create(name, shape=(
            self.nb_bins,), initializer=1.0/self.input_scale[:self.nb_bins])
        return F.reshape(param, (1, 1, 1, self.nb_bins))

    def create_output_parameters(self, name):
        param = get_parameter_or_create(name, shape=(
            self.nb_output_bins,), initializer=np.ones((self.nb_output_bins,)))
        return F.reshape(param, (1, 1, 1, self.nb_output_bins))

    def lstm(self, lstm_in, nb_samples, scope_name):
        '''
        Apply 3-layered LSTM
        '''
        h = F.constant(shape=(self.nb_layers, self.nb_of_directions,
                              nb_samples, self.hidden_size // 2))
        c = F.constant(shape=(self.nb_layers, self.nb_of_directions,
                              nb_samples, self.hidden_size // 2))
        lstm_out, _, _ = PF.lstm(lstm_in, h, c, num_layers=self.nb_layers,
                                 bidirectional=True, training=not self.test, dropout=0.4, name=scope_name)
        return lstm_out

    def fc_bn(self, fc_bn_in, out_channels, scope_name, activation=None):
        '''
        Apply dense and batch norm layer
        '''
        fc_out = PF.affine(fc_bn_in, out_channels, base_axis=2,
                           with_bias=False, name=scope_name)
        fc_bn_out = PF.batch_normalization(
            fc_out, axes=[2], batch_stat=not self.test, name=scope_name)
        if activation == 'tanh':
            fc_bn_out = F.tanh(fc_bn_out)
        elif activation == 'relu':
            fc_bn_out = F.relu(fc_bn_out)
        return fc_bn_out


class OpenUnmix_CrossNet(BaseClass):
    def __init__(
        self,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        is_predict=False
    ):
        super(OpenUnmix_CrossNet, self).__init__(
            input_mean=input_mean, input_scale=input_scale, max_bin=max_bin)

        self.is_predict = is_predict

        # Initilize input mean for all the 4 categories of track
        self.input_mean_bass = self.create_input_mean_parameters(
            'input_mean_bass')
        self.input_mean_drums = self.create_input_mean_parameters(
            'input_mean_drums')
        self.input_mean_vocals = self.create_input_mean_parameters(
            'input_mean_vocals')
        self.input_mean_other = self.create_input_mean_parameters(
            'input_mean_other')

        # Initilize input scale for all the 4 categories of track
        self.input_scale_bass = self.create_input_scale_parameters(
            'input_scale_bass')
        self.input_scale_drums = self.create_input_scale_parameters(
            'input_scale_drums')
        self.input_scale_vocals = self.create_input_scale_parameters(
            'input_scale_vocals')
        self.input_scale_other = self.create_input_scale_parameters(
            'input_scale_other')

        # Initilize output scale for all the 4 categories of track
        self.output_scale_bass = self.create_output_parameters(
            'output_scale_bass')
        self.output_scale_drums = self.create_output_parameters(
            'output_scale_drums')
        self.output_scale_vocals = self.create_output_parameters(
            'output_scale_vocals')
        self.output_scale_other = self.create_output_parameters(
            'output_scale_other')

        # Initilize output mean for all the 4 categories of track
        self.output_mean_bass = self.create_output_parameters(
            'output_mean_bass')
        self.output_mean_drums = self.create_output_parameters(
            'output_mean_drums')
        self.output_mean_vocals = self.create_output_parameters(
            'output_mean_vocals')
        self.output_mean_other = self.create_output_parameters(
            'output_mean_other')

    def __call__(self, x, test=False):
        '''
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Outputs: Input Power/Mag Spectrogram, Output Power/Mag Spectrogram and Predictd sources
        '''
        self.test = test
        fft_real, fft_imag = get_stft(x, n_fft=self.n_fft, n_hop=self.n_hop)
        x_theta = F.atan2(fft_imag, fft_real)
        x = get_spectogram(fft_real, fft_imag, mono=(self.nb_channels == 1))

        nb_frames, nb_samples, nb_channels, nb_bins = x.shape

        mix_spec = F.identity(x)
        x = x[..., :self.nb_bins]

        # clone
        x_bass = F.identity(x)
        x_drums = F.identity(x)
        x_vocals = F.identity(x)
        x_other = F.identity(x)

        # shift and scale input to mean=0 std=1 (across all bins)
        x_bass += self.input_mean_bass
        x_drums += self.input_mean_drums
        x_vocals += self.input_mean_vocals
        x_other += self.input_mean_other

        x_bass *= self.input_scale_bass
        x_drums *= self.input_scale_drums
        x_vocals *= self.input_scale_vocals
        x_other *= self.input_scale_other

        # encode and normalize every instance in a batch
        x_bass = self.fc_bn(x_bass, self.hidden_size,
                            "fc1_bass", activation='tanh')
        x_drums = self.fc_bn(x_drums, self.hidden_size,
                             "fc1_drums", activation='tanh')
        x_vocals = self.fc_bn(x_vocals, self.hidden_size,
                              "fc1_vocals", activation='tanh')
        x_other = self.fc_bn(x_other, self.hidden_size,
                             "fc1_other", activation='tanh')

        # Average the sources
        cross_1 = (x_bass + x_drums + x_vocals + x_other) / 4.0

        # apply 3-layers of stacked LSTM
        lstm_out_bass = self.lstm(cross_1, nb_samples, "lstm_bass")
        lstm_out_drums = self.lstm(cross_1, nb_samples, "lstm_drums")
        lstm_out_vocals = self.lstm(cross_1, nb_samples, "lstm_vocals")
        lstm_out_other = self.lstm(cross_1, nb_samples, "lstm_other")

        # lstm skip connection
        x_bass = F.concatenate(x_bass, lstm_out_bass)
        x_drums = F.concatenate(x_drums, lstm_out_drums)
        x_vocals = F.concatenate(x_vocals, lstm_out_vocals)
        x_other = F.concatenate(x_other, lstm_out_other)

        cross_2 = (x_bass + x_drums + x_vocals + x_other) / 4.0

        # first dense stage + batch norm
        x_bass = self.fc_bn(cross_2, self.hidden_size,
                            "fc2_bass", activation='relu')
        x_drums = self.fc_bn(cross_2, self.hidden_size,
                             "fc2_drums", activation='relu')
        x_vocals = self.fc_bn(cross_2, self.hidden_size,
                              "fc2_vocals", activation='relu')
        x_other = self.fc_bn(cross_2, self.hidden_size,
                             "fc2_other", activation='relu')

        # second dense stage + batch norm
        x_bass = self.fc_bn(x_bass, nb_channels*nb_bins, "fc3_bass")
        x_drums = self.fc_bn(x_drums, nb_channels*nb_bins, "fc3_drums")
        x_vocals = self.fc_bn(x_vocals, nb_channels*nb_bins, "fc3_vocals")
        x_other = self.fc_bn(x_other, nb_channels*nb_bins, "fc3_other")

        # reshape back to original dim
        x_bass = F.reshape(x_bass, (nb_frames, nb_samples,
                                    nb_channels, self.nb_output_bins))
        x_drums = F.reshape(x_drums, (nb_frames, nb_samples,
                                      nb_channels, self.nb_output_bins))
        x_vocals = F.reshape(
            x_vocals, (nb_frames, nb_samples, nb_channels, self.nb_output_bins))
        x_other = F.reshape(x_other, (nb_frames, nb_samples,
                                      nb_channels, self.nb_output_bins))

        # apply output scale and shift
        x_bass *= self.output_scale_bass
        x_drums *= self.output_scale_drums
        x_vocals *= self.output_scale_vocals
        x_other *= self.output_scale_other

        x_bass += self.output_mean_bass
        x_drums += self.output_mean_drums
        x_vocals += self.output_mean_vocals
        x_other += self.output_mean_other

        # since our output is non-negative, we can apply RELU
        mask_bass = F.relu(x_bass)
        mask_drums = F.relu(x_drums)
        mask_vocals = F.relu(x_vocals)
        mask_other = F.relu(x_other)

        # (Frames, Bsize, Channels, Fbins)
        x_bass = mask_bass * mix_spec
        x_drums = mask_drums * mix_spec
        x_vocals = mask_vocals * mix_spec
        x_other = mask_other * mix_spec

        if not self.is_predict:
            tmp = F.stack(*[x_bass, x_drums, x_vocals, x_other], axis=0)
            # (4(sources), Frames, Bsize(16), 2(channels), Fbins) ==> (4, Bsize, Channels, Fbins, Frames)
            tmp = F.transpose(tmp, (0, 2, 3, 4, 1))
            pred_r, pred_i = [], []
            for i in range(tmp.shape[0]):
                pred_r.append(tmp[i] * F.cos(x_theta))
                pred_i.append(tmp[i] * F.sin(x_theta))
            pred_r = F.stack(*pred_r, axis=0)
            pred_i = F.stack(*pred_i, axis=0)
            pred_r = F.reshape(
                pred_r, (4*nb_samples*nb_channels, self.nb_output_bins, nb_frames))
            pred_i = F.reshape(
                pred_i, (4*nb_samples*nb_channels, self.nb_output_bins, nb_frames))
            pred = F.istft(pred_r, pred_i, self.n_fft, self.n_hop, self.n_fft,
                           window_type='hanning', center=True, pad_mode='constant')
            pred = F.reshape(pred, (4, nb_samples, nb_channels, -1))
        else:
            pred = None

        return mix_spec, F.concatenate(mask_bass, mask_drums, mask_vocals, mask_other, axis=2), pred


class OpenUnmix(BaseClass):
    def __init__(
        self,
        input_mean=None,
        input_scale=None,
        max_bin=None,
    ):
        super(OpenUnmix, self).__init__(input_mean=input_mean,
                                        input_scale=input_scale, max_bin=max_bin)

        self.input_mean = self.create_input_mean_parameters('input_mean')
        self.input_scale = self.create_input_scale_parameters('input_scale')
        self.output_scale = self.create_output_parameters('output_scale')
        self.output_mean = self.create_output_parameters('output_mean')

    def __call__(self, x, test=False):
        '''
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Outputs: Output Power/Mag Spectrogram
        '''
        self.test = test
        if not self.input_is_spectrogram:
            x = get_spectogram(*get_stft(x, n_fft=self.n_fft, n_hop=self.n_hop,
                                         center=self.test), mono=(self.nb_channels == 1))
        nb_frames, nb_samples, nb_channels, nb_bins = x.shape
        mix_spec = F.identity(x)

        # crop
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # encode and normalize every instance in a batch
        x = self.fc_bn(x, self.hidden_size, "fc1", activation='tanh')

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x, nb_samples, "lstm")

        # lstm skip connection
        x = F.concatenate(x, lstm_out)

        # first dense stage + batch norm
        x = self.fc_bn(x, self.hidden_size, "fc2", activation='relu')

        # second dense stage + batch norm
        x = self.fc_bn(x, nb_channels*nb_bins, "fc3")

        # reshape back to original dim
        x = F.reshape(x, (nb_frames, nb_samples,
                          nb_channels, self.nb_output_bins))

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        return F.relu(x) * mix_spec


def get_model(args, input_mean, input_scale, max_bin=None):
    '''
    Create computation graph and variables for X-UMX/UMX.
    '''
    # target channels (2 for UMX and 8 for X-UMX (2 * 4-target sources
    target_channels = args.nb_channels if args.umx_train else 4 * args.nb_channels

    # Create input variables.
    mixture_audio = nn.Variable(
        (args.batch_size, args.nb_channels, args.sample_rate * args.seq_dur))
    target_audio = nn.Variable(
        (args.batch_size, target_channels, args.sample_rate * args.seq_dur))
    vmixture_audio = nn.Variable(
        (1, args.nb_channels, args.sample_rate * args.valid_dur))
    vtarget_audio = nn.Variable(
        (1, target_channels, args.sample_rate * args.valid_dur))

    if args.umx_train:
        # create training graph for UMX
        unmix = OpenUnmix(input_mean=input_mean,
                          input_scale=input_scale, max_bin=max_bin)
        pred_spec = unmix(mixture_audio)
        target_spec = get_spectogram(*get_stft(target_audio, n_fft=unmix.n_fft,
                                               n_hop=unmix.n_hop, center=False), mono=(unmix.nb_channels == 1))
        loss = F.mean(F.squared_error(pred_spec, target_spec))

        # create validation graph for UMX
        vpred_spec = unmix(vmixture_audio, test=True)
        vtarget_spec = get_spectogram(
            *get_stft(vtarget_audio, n_fft=unmix.n_fft, n_hop=unmix.n_hop), mono=(unmix.nb_channels == 1))
        vloss = F.mean(F.squared_error(vpred_spec, vtarget_spec))
    else:
        # create training graph for X-UMX
        unmix = OpenUnmix_CrossNet(
            input_mean=input_mean, input_scale=input_scale, max_bin=max_bin)
        mix_spec, m_hat, pred = unmix(mixture_audio)
        target_spec = get_spectogram(
            *get_stft(target_audio, n_fft=unmix.n_fft, n_hop=unmix.n_hop), mono=(unmix.nb_channels == 1))
        loss_f = mse_loss(mix_spec, m_hat, target_spec)
        loss_t = sdr_loss(mixture_audio, pred, target_audio)
        loss = args.mcoef * loss_t + loss_f

        # create validation graph for X-UMX
        vmix_spec, vm_hat, vpred = unmix(vmixture_audio, test=True)
        vtarget_spec = get_spectogram(
            *get_stft(vtarget_audio, n_fft=unmix.n_fft, n_hop=unmix.n_hop), mono=(unmix.nb_channels == 1))
        vloss_f = mse_loss(vmix_spec, vm_hat, vtarget_spec)
        vloss_t = sdr_loss(vmixture_audio, vpred, vtarget_audio)
        vloss = args.mcoef * vloss_t + vloss_f

    loss.persistent = True
    vloss.persistent = True

    Network = collections.namedtuple(
        'Network', 'loss, vloss, mixture_audio, target_audio, vmixture_audio, vtarget_audio')
    return Network(
        loss=loss,
        vloss=vloss,
        mixture_audio=mixture_audio,
        target_audio=target_audio,
        vmixture_audio=vmixture_audio,
        vtarget_audio=vtarget_audio
    )
