# Copyright 2021 Sony Group Corporation
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
import nnabla.functions as F
import numpy as np

from utils.audio import log_mel_spectrogram, random_split

from .module import Module
from .ops import DownBlock, UpBlock, res_block, wn_conv


class Encoder(Module):
    r"""Implementation of the content encoder.

    Args:
        hp (HParams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp
        for i in range(len(hp.ratios)):
            setattr(self, f"block_{i}", DownBlock(hp))

    def call(self, x):
        hp = self.hp

        with nn.parameter_scope("first_layer"):
            x = F.pad(x, (0, 0, 3, 3), 'reflect')
            x = wn_conv(x, hp.ngf, (7,))

        for i, r in enumerate(reversed(hp.ratios)):
            x = getattr(self, f"block_{i}")(x, r, 2**(i + 1))

        with nn.parameter_scope("last_layer"):
            x = F.gelu(x)
            x = F.pad(x, (0, 0, 3, 3), 'reflect')
            x = wn_conv(x, x.shape[1], (7,))

        with nn.parameter_scope("content"):
            x = F.gelu(x)
            x = F.pad(x, (0, 0, 3, 3), 'reflect')
            x = wn_conv(x, hp.bottleneck_dim, (7,), with_bias=False)
            x = x / F.sum(x**2 + 1e-12, axis=1, keepdims=True)**0.5

        return x


class Decoder(Module):
    r"""Implementation of the generator.

    Args:
        hp (HParams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp
        for i in range(len(hp.ratios)):
            setattr(self, f"block_{i}", UpBlock(hp))

    def call(self, x, spk_emb):
        hp = self.hp
        self.hop_length = np.prod(hp.ratios)
        mult = int(2 ** len(hp.ratios))

        with nn.parameter_scope("upsample"):
            x = F.pad(x, (0, 0, 3, 3), 'reflect')
            x = wn_conv(x, mult * hp.ngf, (7,))

        with nn.parameter_scope("first_layer"):
            x = F.gelu(x)
            x = F.pad(x, (0, 0, 3, 3), 'reflect')
            x = wn_conv(x, x.shape[1], (7,))

        for i, r in enumerate(hp.ratios):
            x = getattr(self, f"block_{i}")(x, spk_emb, r, mult // (2**i))

        with nn.parameter_scope("waveform"):
            x = F.gelu(x)
            x = F.pad(x, (0, 0, 3, 3), 'reflect')
            x = wn_conv(x, 1, (7,))
            x = F.tanh(x)

        return x


class Speaker(Module):
    r"""Implementation of the speaker encoder.

    Args:
        hp (HParams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp
        self.rng = np.random.RandomState(hp.seed)

    def call(self, x):
        hp = self.hp
        dim = hp.n_speaker_embedding
        kernel, pad = (3,), (1,)

        with nn.parameter_scope('melspectrogram'):
            out = log_mel_spectrogram(x, hp.sr, 1024)
            if self.training:
                out = random_split(
                    out, axis=2, rng=self.rng,
                    lo=hp.split_low,
                    hi=hp.split_hight,
                )

        with nn.parameter_scope('init_conv'):
            out = wn_conv(out, 32, kernel=kernel, pad=pad)

        for i in range(hp.n_spk_layers):
            dim_out = min(out.shape[1] * 2, 512)
            out = res_block(
                out, dim_out,
                kernel=kernel, pad=pad,
                scope=f'downsample_{i}',
                training=self.training
            )

        with nn.parameter_scope('last_layer'):
            out = F.average_pooling(out, kernel=(1, out.shape[-1]))
            out = F.leaky_relu(out, 0.2)

        with nn.parameter_scope('mean'):
            mu = wn_conv(out, dim, kernel=(1,))

        with nn.parameter_scope('logvar'):
            logvar = wn_conv(out, dim, kernel=(1,))

        return mu, logvar


class NVCNet(Module):
    def __init__(self, hp):
        self.hp = hp
        self.encoder = Encoder(hp)
        self.decoder = Decoder(hp)
        self.speaker = Speaker(hp)

    def call(self, x, y):
        r"""Convert an audio for a given reference.

        Args:
            x (nn.Variable): Input audio of shape (B, 1, L).
            y_tar (nn.Variable): Target class of shape (B, 1).

        Returns:
            nn.Variable: Converted audio.
        """
        style = self.embed(y)[0]
        content = self.encode(x)
        out = self.decode(content, style)
        return out

    def encode(self, x):
        r"""Encode an input audio.

        Args:
            x (nn.Variable): Input audio of shape (B, 1, L).

        Returns:
            nn.Variable: Content info.
        """
        with nn.parameter_scope('', self.parameter_scope):
            with nn.parameter_scope('encoder'):
                return self.encoder(x)

    def decode(self, content, spk_emb):
        r"""Generate an audio from content and speaker info.

        Args:
            content (nn.Variable): Content info.
            y_tar (nn.Variable): Target class of shape (B, 1).

        Returns:
            nn.Variable: Generated audio.
        """
        with nn.parameter_scope('', self.parameter_scope):
            with nn.parameter_scope('decoder'):
                x = self.decoder(content, spk_emb)
        return x

    def embed(self, x):
        r"""Returns an embedding for a given audio reference.

        Args:
            x (nn.Variable): Input audio of shape (B, 1, L).

        Returns:
            nn.Variable: Embedding of the audio of shape (B, D, 1).
            nn.Variable: Mean of the output distribution of shape (B, D, 1).
            nn.Variable: Log variance of the output distribution of shape
                (B, D, 1).
        """
        with nn.parameter_scope('', self.parameter_scope):
            with nn.parameter_scope('embedding'):
                mu, logvar = self.speaker(x)
                spk_emb = self.sample(mu, logvar)
        return spk_emb, mu, logvar

    def kl_loss(self, mu, logvar):
        r"""Returns the Kullback-Leibler divergence loss with a standard Gaussian.

        Args:
            mu (nn.Variable): Mean of the distribution of shape (B, D, 1).
            logvar (nn.Variable): Log variance of the distribution of
                shape (B, D, 1).

        Returns:
            nn.Variable: Kullback-Leibler divergence loss.
        """
        return 0.5 * F.mean(F.sum(F.exp(logvar) + mu**2 - 1. - logvar, axis=1))

    def sample(self, mu, logvar):
        r"""Samples from a Gaussian distribution.

        Args:
            mu (nn.Variable): Mean of the distribution of shape (B, D, 1).
            logvar (nn.Variable): Log variance of the distribution of
                shape (B, D, 1).

        Returns:
            nn.Variable: A sample.
        """
        if self.training:
            eps = F.randn(shape=mu.shape)
            return mu + F.exp(0.5 * logvar) * eps
        return mu


class NLayerDiscriminator(Module):
    r"""A single discriminator.

    Args:
        hp (HParams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp

    def call(self, x, y):
        hp = self.hp
        results = []
        with nn.parameter_scope('layer_0'):
            x = F.pad(x, (0, 0, 7, 7), 'reflect')
            x = wn_conv(x, hp.ndf, (15,))
            x = F.leaky_relu(x, 0.2, inplace=True)
            results.append(x)

        nf = hp.ndf
        stride = hp.downsamp_factor

        for i in range(1, hp.n_layers_D + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)
            with nn.parameter_scope(f'layer_{i}'):
                x = wn_conv(
                    x, nf, (stride * 10 + 1,),
                    stride=(stride,),
                    pad=(stride * 5,),
                    group=nf_prev // 4,
                )
                x = F.leaky_relu(x, 0.2, inplace=True)
                results.append(x)

        with nn.parameter_scope(f'layer_{hp.n_layers_D + 1}'):
            nf = min(nf * 2, 1024)
            x = wn_conv(x, nf, kernel=(5,), pad=(2,))
            x = F.leaky_relu(x, 0.2, inplace=True)
            results.append(x)

        with nn.parameter_scope(f'layer_{hp.n_layers_D + 2}'):
            x = wn_conv(x, hp.n_speakers, kernel=(3,), pad=(1,))
            if y is not None:
                idx = F.stack(
                    F.arange(0, hp.batch_size),
                    y.reshape((hp.batch_size,))
                )
                x = F.gather_nd(x, idx)
            results.append(x)

        return results


class Discriminator(Module):
    r"""Implementation of the multi-scale discriminator.

    Args:
        hp (HParams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp
        for i in range(hp.num_D):
            setattr(self, f'dis_{i}', NLayerDiscriminator(hp))

    def call(self, x, y):
        hp = self.hp
        results = []
        for i in range(hp.num_D):
            results.append(getattr(self, f'dis_{i}')(x, y))
            x = F.average_pooling(
                x, (1, 4),
                stride=(1, 2),
                pad=(0, 1),
                including_pad=False
            )
        return results

    def spectral_loss(self, x, target):
        r"""Returns the multi-scale spectral loss.

        Args:
            x (nn.Variable): Input variable.
            target (nn.Variable): Target variable.

        Returns:
            nn.Variable: Multi-scale spectral loss.
        """
        loss = []
        for window_size in self.hp.window_sizes:
            sx = log_mel_spectrogram(x, self.hp.sr, window_size)
            st = log_mel_spectrogram(target, self.hp.sr, window_size)
            st.need_grad = False  # avoid grads flowing though targets
            loss.append(F.mean(F.squared_error(sx, st)))
        return sum(loss)

    def preservation_loss(self, x, target):
        r"""Returns content preservation loss.

            Args:
                x (nn.Variable): Input variable.
                target (nn.Variable): Target variable.

            Returns:
                nn.Variable: Output loss.
            """
        loss = F.mean(F.squared_error(x, target))
        return loss

    def perceptual_loss(self, x, target):
        r"""Returns perceptual loss."""
        loss = []
        out_x, out_t = self(x, None), self(target, None)
        for (a, t) in zip(out_x, out_t):
            for la, lt in zip(a[:-1], t[:-1]):
                lt.need_grad = False  # avoid grads flowing though targets
                loss.append(F.mean(F.absolute_error(la, lt)))
        return sum(loss) / self.hp.num_D

    def adversarial_loss(self, results, v):
        r"""Returns the adversarial loss.

        Args:
            results (list): Output from discriminator.
            v (int, optional): Target value. Real=1.0, fake=0.0.

        Returns:
            nn.Variable: Output variable.
        """
        loss = []
        for out in results:
            t = F.constant(v, shape=out[-1].shape)
            r = F.sigmoid_cross_entropy(out[-1], t)
            loss.append(F.mean(r))
        return sum(loss)
