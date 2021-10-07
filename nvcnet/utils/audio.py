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

import librosa as lr
import nnabla as nn
import nnabla.functions as F
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from nnabla.random import prng
from scipy.ndimage import interpolation

from .misc import RandomSplit


def stft(x, window_size, stride, fft_size,
         window_type='hanning', center=True, pad_mode='reflect'):
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
            window_func, (diff // 2, diff - diff // 2), mode='constant')
    elif fft_size < window_size:
        raise ValueError(
            "FFT size has to be as least as large as window size.")

    # compute STFT filter coefficients
    mat_r = np.zeros((fft_size // 2 + 1, 1, fft_size))
    mat_i = np.zeros((fft_size // 2 + 1, 1, fft_size))

    for w in range(fft_size // 2 + 1):
        for t in range(fft_size):
            mat_r[w, 0, t] = np.cos(2. * np.pi * w * t / fft_size)
            mat_i[w, 0, t] = -np.sin(2. * np.pi * w * t / fft_size)

    conv_r = nn.Variable.from_numpy_array(mat_r * window_func)
    conv_i = nn.Variable.from_numpy_array(mat_i * window_func)

    if center:
        # pad at begin/end (per default this is a reflection padding)
        x = F.pad(x, (fft_size // 2, fft_size // 2), mode=pad_mode)

    # compute STFT
    y_r = F.convolution(x, conv_r, stride=(stride,))
    y_i = F.convolution(x, conv_i, stride=(stride,))

    return y_r, y_i


def spectrogram(wave, window_size):
    """Computes the spectrogram from the waveform.

    Args:
        wave (nn.Variable): Input waveform of shape (B, 1, L).
        window_size (int): Window size.

    Returns:
        nn.Variable: The square spectrogram.
    """
    re, im = stft(wave, window_size=window_size,
                  stride=window_size // 4, fft_size=window_size)
    return F.pow_scalar(re**2 + im**2, 0.5)


def log_spectrogram(wave, window_size):
    r"""Return log spectrogram.

    Args:
        wave (nn.Variable): Input waveform of shape (B, 1, L).
        window_size (int): Window size.

    Returns:
        nn.Variable: Log spectrogram.
    """
    linear = spectrogram(wave, window_size)
    return F.log(linear * 1e4 + 1.0)


def log_mel_spectrogram(wave, sr, window_size, n_mels=80):
    """Return log mel-spectrogram.

    Args:
        wave (nn.Variable): Input waveform of shape (B, 1, L).
        sr (int): Sampling rate.
        window_size (int): Window size.
        n_mels (int): Number of mel banks.
        jitter (bool): Whether to apply random crop. Defaults to False.
        max_jitter_steps (int): Maximum number of jitter steps if jitter is
            set to `True`.

    Returns:
        nn.Variable: Log mel-spectrogram.
    """
    linear = spectrogram(wave, window_size)
    mel_basis = librosa_mel_fn(
        sr, window_size, n_mels=n_mels,
        fmin=80.0, fmax=7600.0
    )
    basis = nn.Variable.from_numpy_array(mel_basis[None, ...])
    mels = F.batch_matmul(basis, linear)
    return F.log(mels * 1e4 + 1.0)


def stretch_audio(x, rate, window_size=512):
    """Stretch the audio speech using spectrogram.

    Args:
        x (numpy.ndarray): Input waveform.
        rate (float): Rate of stretching.
        window_size (int, optional): Window size for stft. Defaults to 512.

    Returns:
        numpy.ndarray: The stretched audio.
    """
    c = lr.stft(
        x, n_fft=window_size, hop_length=window_size // 4,
        win_length=window_size)
    re = interpolation.zoom(c.real, zoom=(1, rate))
    im = interpolation.zoom(c.imag, zoom=(1, rate))
    w = lr.istft(re + im * 1j, hop_length=window_size //
                 4, win_length=window_size)
    return w


def random_flip(x):
    r"""Random flipping sign of a Variable.

    Args:
        x (nn.Variable): Input Variable.
    """
    shape = (x.shape[0], 1, 1)
    scale = 2 * F.randint(0, 2, shape=shape) - 1
    return x * scale


def random_scaling(x, lo, hi):
    r"""Random scaling a Variable.

    Args:
        x (nn.Variable): Input Variable.
        lo (int): Low value.
        hi (int): High value.

    Returns:
        nn.Variable: Output Variable.
    """
    shape = (x.shape[0], 1, 1)
    scale = F.rand(lo, hi, shape=shape)
    return x * scale


def random_jitter(wave, max_jitter_steps):
    r"""Temporal jitter."""
    shape = wave.shape
    wave = F.pad(wave, (0, 0, max_jitter_steps, max_jitter_steps))
    wave = F.random_crop(wave, shape=shape)
    return wave


def random_split(x, lo, hi, axis=1, rng=None, ctx=None):
    r"""Returns a tensor by random splitting it into smaller parts, then
    concatenate together.

    Args:
        x (nn.Variable): Input Variable.
        lo (int): Low value.
        hi (int): High value.
        axis (int, optional): Axis to perform random split.
            Defaults to 1.
        rng ([type], optional): RandomState. Defaults to None.
        ctx ([type], optional): Context. Defaults to None.

    Returns:
        nn.Variable: Output Variable.
    """
    rng = rng or prng
    func = RandomSplit(lo, hi, axis, rng, ctx)
    return func(x)
