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

import nnabla as nn
from nnabla.parameter import get_parameter_or_create
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np


def STFT(x, n_fft=4096, n_hop=1024, center=True):
    """Multichannel STFT

    Input: (nb_samples, nb_channels, nb_timesteps)
    Output: (nb_samples, nb_channels, nb_bins, nb_frames), 
            (nb_samples, nb_channels, nb_bins, nb_frames)
    """
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


def istft(y_r, y_i, window_size, stride, fft_size, window_type='hanning', center=True):
    '''Workaround wrapper of ISTFT for fixing a bug in nnabla<=1.15.0
    '''
    from utils import get_nnabla_version_integer
    if get_nnabla_version_integer() > 11500:
        return F.istft(**locals())
    import numpy as np
    from nnabla.parameter import get_parameter, get_parameter_or_create
    conv_cos = get_parameter('conv_cos')
    conv_sin = get_parameter('conv_sin')

    if conv_cos is None or conv_sin is None:
        if window_type == 'hanning':
            window_func = np.hanning(window_size + 1)[:-1]
        elif window_type == 'hamming':
            window_func = np.hamming(window_size + 1)[:-1]
        elif window_type == 'rectangular' or window_type is None:
            window_func = np.ones(window_size)
        else:
            raise ValueError("Unknown window type {}.".format(window_type))

        # pad window if `fft_size > window_size`
        if fft_size > window_size:
            diff = fft_size - window_size
            window_func = np.pad(
                window_func, (diff//2, diff - diff//2), mode='constant')
        elif fft_size < window_size:
            raise ValueError(
                "FFT size has to be as least as large as window size.")

        # compute inverse STFT filter coefficients
        if fft_size % stride != 0:
            raise ValueError("FFT size needs to be a multiple of stride.")

        inv_window_func = np.zeros_like(window_func)
        for s in range(0, fft_size, stride):
            inv_window_func += np.roll(np.square(window_func), s)

        mat_cos = np.zeros((fft_size//2 + 1, 1, fft_size))
        mat_sin = np.zeros((fft_size//2 + 1, 1, fft_size))

        for w in range(fft_size//2+1):
            alpha = 1.0 if w == 0 or w == fft_size//2 else 2.0
            alpha /= fft_size
            for t in range(fft_size):
                mat_cos[w, 0, t] = alpha * \
                    np.cos(2. * np.pi * w * t / fft_size)
                mat_sin[w, 0, t] = alpha * \
                    np.sin(2. * np.pi * w * t / fft_size)
        mat_cos = mat_cos * window_func / inv_window_func
        mat_sin = mat_sin * window_func / inv_window_func

        conv_cos = get_parameter_or_create(
            'conv_cos', initializer=mat_cos, need_grad=False)
        conv_sin = get_parameter_or_create(
            'conv_sin', initializer=mat_sin, need_grad=False)

    # compute inverse STFT
    x_cos = F.deconvolution(y_r, conv_cos, stride=(stride,))
    x_sin = F.deconvolution(y_i, conv_sin, stride=(stride,))

    x = F.reshape(x_cos - x_sin, (x_cos.shape[0], x_cos.shape[2]))

    if center:
        x = x[:, fft_size//2:-fft_size//2]

    return x


def Spectrogram(real, imag, power=1, mono=True):
    """
    Input:  (nb_samples, nb_channels, nb_bins, nb_frames), 
            (nb_samples, nb_channels, nb_bins, nb_frames)
    Output: (nb_frames, nb_samples, nb_channels, nb_bins)
    """
    spec = ((real ** 2) + (imag ** 2)) ** (power / 2.0)

    if mono:
        spec = F.mean(spec, axis=1, keepdims=True)

    return F.transpose(spec, ((3, 0, 1, 2)))


class OpenUnmix_CrossNet():
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        hidden_size=512,
        nb_channels=2,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Outputs: Input Power/Mag Spectrogram, Output Power/Mag Spectrogram and Predictd sources
        """
        super(OpenUnmix_CrossNet, self).__init__()

        self.is_predict = False
        self.nb_output_bins = n_fft // 2 + 1
        self.hidden_size = hidden_size
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.power = power
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.unidirectional = unidirectional

        if unidirectional:
            self.nb_of_directions = 1
        else:
            self.nb_of_directions = 2

        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        if input_mean is None:
            input_mean = np.zeros((self.nb_bins,))
        if input_scale is None:
            input_scale = np.ones((self.nb_bins))

        self.input_mean_bass = get_parameter_or_create(
            'input_mean_bass', (self.nb_bins,), initializer=-input_mean[:self.nb_bins])
        self.input_mean_drums = get_parameter_or_create(
            'input_mean_drums', (self.nb_bins,), initializer=-input_mean[:self.nb_bins])
        self.input_mean_vocals = get_parameter_or_create(
            'input_mean_vocals', (self.nb_bins,), initializer=-input_mean[:self.nb_bins])
        self.input_mean_other = get_parameter_or_create(
            'input_mean_other', (self.nb_bins,), initializer=-input_mean[:self.nb_bins])

        self.input_scale_bass = get_parameter_or_create(
            'input_scale_bass', (self.nb_bins,), initializer=1.0/input_scale[:self.nb_bins])
        self.input_scale_drums = get_parameter_or_create(
            'input_scale_drums', (self.nb_bins,), initializer=1.0/input_scale[:self.nb_bins])
        self.input_scale_vocals = get_parameter_or_create(
            'input_scale_vocals', (self.nb_bins,), initializer=1.0/input_scale[:self.nb_bins])
        self.input_scale_other = get_parameter_or_create(
            'input_scale_other', (self.nb_bins,), initializer=1.0/input_scale[:self.nb_bins])

        self.output_scale_bass = get_parameter_or_create(
            'output_scale_bass', (self.nb_output_bins,), initializer=np.ones((self.nb_output_bins,)))
        self.output_scale_drums = get_parameter_or_create(
            'output_scale_drums', (self.nb_output_bins,), initializer=np.ones((self.nb_output_bins,)))
        self.output_scale_vocals = get_parameter_or_create(
            'output_scale_vocals', (self.nb_output_bins,), initializer=np.ones((self.nb_output_bins,)))
        self.output_scale_other = get_parameter_or_create(
            'output_scale_other', (self.nb_output_bins,), initializer=np.ones((self.nb_output_bins,)))

        self.output_mean_bass = get_parameter_or_create(
            'output_mean_bass', (self.nb_output_bins,), initializer=np.ones((self.nb_output_bins,)))
        self.output_mean_drums = get_parameter_or_create(
            'output_mean_drums', (self.nb_output_bins,), initializer=np.ones((self.nb_output_bins,)))
        self.output_mean_vocals = get_parameter_or_create(
            'output_mean_vocals', (self.nb_output_bins,), initializer=np.ones((self.nb_output_bins,)))
        self.output_mean_other = get_parameter_or_create(
            'output_mean_other', (self.nb_output_bins,), initializer=np.ones((self.nb_output_bins,)))

    def lstm(self, lstm_in, nb_samples, scope_name, test):
        """
        Apply 3-layered LSTM
        """

        if self.unidirectional:
            lstm_hidden_size = self.hidden_size
        else:
            lstm_hidden_size = self.hidden_size // 2
        h = nn.Variable((self.nb_layers, self.nb_of_directions,
                         nb_samples, lstm_hidden_size), need_grad=False)
        c = nn.Variable((self.nb_layers, self.nb_of_directions,
                         nb_samples, lstm_hidden_size), need_grad=False)
        h.data.zero()
        c.data.zero()
        lstm_out, _, _ = PF.lstm(lstm_in, h, c, num_layers=self.nb_layers,
                                 bidirectional=not self.unidirectional, training=not test, dropout=0.4, name=scope_name)
        return lstm_out

    def fc_bn(self, fc_bn_in, out_channels, scope_name, test, activation=None):
        """
        Apply dense and batch norm layer
        """
        fc_out = PF.affine(fc_bn_in, out_channels, base_axis=2,
                           with_bias=False, name=scope_name)
        fc_bn_out = PF.batch_normalization(
            fc_out, axes=[2], batch_stat=not test, name=scope_name)
        if activation == 'tanh':
            fc_bn_out = F.tanh(fc_bn_out)
        elif activation == 'relu':
            fc_bn_out = F.relu(fc_bn_out)
        return fc_bn_out

    def __call__(self, x, test=False):

        fft_real, fft_imag = STFT(x, n_fft=self.n_fft, n_hop=self.n_hop)
        x_theta = F.atan2(fft_imag, fft_real)

        x = Spectrogram(fft_real, fft_imag, power=self.power,
                        mono=(self.nb_channels == 1))

        nb_frames, nb_samples, nb_channels, nb_bins = x.shape

        mix_spec = F.identity(x)
        x = x[..., :self.nb_bins]

        # clone
        x_bass = F.identity(x)
        x_drums = F.identity(x)
        x_vocals = F.identity(x)
        x_other = F.identity(x)

        # shift and scale input to mean=0 std=1 (across all bins)
        x_bass += F.reshape(self.input_mean_bass,
                            shape=(1, 1, 1, self.nb_bins), inplace=False)
        x_drums += F.reshape(self.input_mean_drums,
                             shape=(1, 1, 1, self.nb_bins), inplace=False)
        x_vocals += F.reshape(self.input_mean_vocals,
                              shape=(1, 1, 1, self.nb_bins), inplace=False)
        x_other += F.reshape(self.input_mean_other,
                             shape=(1, 1, 1, self.nb_bins), inplace=False)

        x_bass *= F.reshape(self.input_scale_bass,
                            shape=(1, 1, 1, self.nb_bins), inplace=False)
        x_drums *= F.reshape(self.input_scale_drums,
                             shape=(1, 1, 1, self.nb_bins), inplace=False)
        x_vocals *= F.reshape(self.input_scale_vocals,
                              shape=(1, 1, 1, self.nb_bins), inplace=False)
        x_other *= F.reshape(self.input_scale_other,
                             shape=(1, 1, 1, self.nb_bins), inplace=False)

        # encode and normalize every instance in a batch
        x_bass = self.fc_bn(x_bass, self.hidden_size,
                            "fc1_bass", test, activation='tanh')
        x_drums = self.fc_bn(x_drums, self.hidden_size,
                             "fc1_drums", test, activation='tanh')
        x_vocals = self.fc_bn(x_vocals, self.hidden_size,
                              "fc1_vocals", test, activation='tanh')
        x_other = self.fc_bn(x_other, self.hidden_size,
                             "fc1_other", test, activation='tanh')

        # Average the sources
        cross_1 = (x_bass + x_drums + x_vocals + x_other) / 4.0

        # apply 3-layers of stacked LSTM
        lstm_out_bass = self.lstm(cross_1, nb_samples, "lstm_bass", test)
        lstm_out_drums = self.lstm(cross_1, nb_samples, "lstm_drums", test)
        lstm_out_vocals = self.lstm(cross_1, nb_samples, "lstm_vocals", test)
        lstm_out_other = self.lstm(cross_1, nb_samples, "lstm_other", test)

        # lstm skip connection
        x_bass = F.concatenate(x_bass, lstm_out_bass)
        x_drums = F.concatenate(x_drums, lstm_out_drums)
        x_vocals = F.concatenate(x_vocals, lstm_out_vocals)
        x_other = F.concatenate(x_other, lstm_out_other)

        cross_2 = (x_bass + x_drums + x_vocals + x_other) / 4.0

        # first dense stage + batch norm
        x_bass = self.fc_bn(cross_2, self.hidden_size,
                            "fc2_bass", test, activation='relu')
        x_drums = self.fc_bn(cross_2, self.hidden_size,
                             "fc2_drums", test, activation='relu')
        x_vocals = self.fc_bn(cross_2, self.hidden_size,
                              "fc2_vocals", test, activation='relu')
        x_other = self.fc_bn(cross_2, self.hidden_size,
                             "fc2_other", test, activation='relu')

        # second dense stage + batch norm
        x_bass = self.fc_bn(x_bass, nb_channels*nb_bins, "fc3_bass", test)
        x_drums = self.fc_bn(x_drums, nb_channels*nb_bins, "fc3_drums", test)
        x_vocals = self.fc_bn(x_vocals, nb_channels *
                              nb_bins, "fc3_vocals", test)
        x_other = self.fc_bn(x_other, nb_channels*nb_bins, "fc3_other", test)

        # reshape back to original dim
        x_bass = F.reshape(x_bass,
                           (nb_frames, nb_samples, nb_channels, self.nb_output_bins))
        x_drums = F.reshape(x_drums,
                            (nb_frames, nb_samples, nb_channels, self.nb_output_bins))
        x_vocals = F.reshape(x_vocals,
                             (nb_frames, nb_samples, nb_channels, self.nb_output_bins))
        x_other = F.reshape(x_other,
                            (nb_frames, nb_samples, nb_channels, self.nb_output_bins))

        # apply output scaling
        x_bass *= F.reshape(self.output_scale_bass,
                            shape=(1, 1, 1, self.nb_output_bins), inplace=False)
        x_drums *= F.reshape(self.output_scale_drums,
                             shape=(1, 1, 1, self.nb_output_bins), inplace=False)
        x_vocals *= F.reshape(self.output_scale_vocals,
                              shape=(1, 1, 1, self.nb_output_bins), inplace=False)
        x_other *= F.reshape(self.output_scale_other,
                             shape=(1, 1, 1, self.nb_output_bins), inplace=False)

        x_bass += F.reshape(self.output_mean_bass,
                            shape=(1, 1, 1, self.nb_output_bins), inplace=False)
        x_drums += F.reshape(self.output_mean_drums,
                             shape=(1, 1, 1, self.nb_output_bins), inplace=False)
        x_vocals += F.reshape(self.output_mean_vocals,
                              shape=(1, 1, 1, self.nb_output_bins), inplace=False)
        x_other += F.reshape(self.output_mean_other,
                             shape=(1, 1, 1, self.nb_output_bins), inplace=False)

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
                pred_r, (4*nb_samples*nb_channels, 2049, nb_frames))
            pred_i = F.reshape(
                pred_i, (4*nb_samples*nb_channels, 2049, nb_frames))
            pred = istft(pred_r, pred_i, self.n_fft, self.n_hop,
                           self.n_fft, window_type='hanning', center=True)
            pred = F.reshape(pred, (4, nb_samples, nb_channels, -1))

        else:
            pred = None

        return mix_spec, F.concatenate(mask_bass, mask_drums, mask_vocals, mask_other, axis=2), pred
