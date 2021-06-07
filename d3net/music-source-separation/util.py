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

import numpy as np
import nnabla as nn
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.utils import mediainfo
from model import d3_net


def generate_data(file_name, fft_size, hop_size, n_channels):
    '''
    Function to load a song specified by `file_name`
    Inputs:
      `file_name`: file name of the input audio
      `fft_size`: FFT window sizes
      `hop_size`: FFT hop size
      `n_channels`: No of channels in the input audio
    Outputs:
      `sample_rate`  :  sample_rate of the input audio
      `data`   :  STFT of the input audio with size `NumFrames x NumChannels x NumBins` (complex-valued) 
    '''
    # get audio data from the path - all formats recognized by FFMPEG are recognized
    audio_with_meta = AudioSegment.from_file(file_name)
    sample_rate = int(mediainfo(file_name)['sample_rate'])
    channel_sounds = audio_with_meta.split_to_mono()
    samples = [s.get_array_of_samples() for idx, s in enumerate(channel_sounds) if idx < 2]
    fp_arr = np.array(samples).T.astype(np.float32)
    audio = fp_arr / np.iinfo(samples[0].typecode).max

    # loop over all channels and compute sequence
    for i in range(audio.shape[1]):
        stft = librosa.stft(audio[:, i].flatten(),
                            n_fft=fft_size, hop_length=hop_size).transpose()
        if i == 0:
            data = np.ndarray(shape=(stft.shape[0], n_channels, fft_size // 2 + 1),
                              dtype=np.complex64)
        data[:, i, :] = stft

    if n_channels == 2 and audio.shape[1] == 1:
        data[:, 1] = data[:, 0]

    return sample_rate, data


def stft2time_domain(stft, hop_size, stft_center=True):
    '''
    Function to obtain time domain audio data from STFT
    Inputs:
      `stft`: `NumFrames x NumChannels x NumBins`
      `hop_size`: FFT hop size
      `stft_center`: No of channels in the input audio
    Outputs:
      `audio`  :  Time domain audio data 
    '''
    if stft.shape[1] == 1:
        audio = librosa.istft(stft[:, 0, :].transpose(
        ), hop_length=hop_size, center=stft_center)
    elif stft.shape[1] >= 2:
        for i in range(stft.shape[1]):
            if i == 0:
                audio = np.expand_dims(librosa.istft(
                    stft[:, i, :].transpose(), hop_length=hop_size, center=stft_center), axis=1)
            else:
                audio = np.concatenate((audio, np.expand_dims(librosa.istft(
                    stft[:, i, :].transpose(), hop_length=hop_size, center=stft_center), axis=1)), axis=1)

    return audio


def save_stft_wav(stft, hop_size, sample_rate, outfile_name, stft_center=True, samplewidth=2):
    '''
    Helper function to save wav file from STFT
    '''
    audio = stft2time_domain(stft, hop_size, stft_center)
    save_timedomain_signal_wav(audio, sample_rate, outfile_name, samplewidth)
    return audio


def save_timedomain_signal_wav(audio, sample_rate, outfile_name, samplewidth=2):
    '''
    Helper function to save wav file from audio data
    '''
    if samplewidth == 2:
        sf.write(outfile_name, audio, sample_rate, 'PCM_16')
    elif samplewidth == 3:
        sf.write(outfile_name, audio, sample_rate, 'PCM_24')
    else:
        sf.write(outfile_name, audio, sample_rate, 'PCM_32')


def model_separate(inp_mag, hparams, ch_flip_average=False):
    '''
    Helper function to run separation
    '''
    _out_model = calc_output_overlap_add(
        inp_mag, hparams, ch_flip_average=ch_flip_average)
    return np.ascontiguousarray(_out_model)


def calc_output_overlap_add(inp, hparams, out_ch=None, ch_flip_average=False):
    '''
    Clop both sides of outputs of the network and overlap add by shifting
    Inputs:
      `inp` :   input mixture x
      `hparams` : Hyper parameters
      `out_ch`: output channels
      `ch_flip_average`: channel flip flag
    Outputs:
      `output' : output STFT
    '''
    patch_length = hparams['test_patch_len']
    shift = hparams['out_shift']
    scale = patch_length // shift  # this MUST be Natural number.
    zmod_length = patch_length - (inp.shape[0] % patch_length)
    patches = np.ceil((inp.shape[0] + zmod_length) // shift).astype(np.int32)
    output_length = patches * shift + 2 * shift * (scale - 1)

    zpad = np.zeros([shift * (scale - 1), inp.shape[1],
                     inp.shape[2]], dtype=np.float32)
    zpad2 = np.zeros([shift * scale, inp.shape[1],
                      inp.shape[2]], dtype=np.float32)
    zpadmod = np.zeros(
        [zmod_length, inp.shape[1], inp.shape[2]], dtype=np.float32)

    inp_padded = np.concatenate((zpad, inp, zpadmod, zpad2), axis=0)
    if out_ch is None:
        out_ch = inp.shape[1]
    output = np.zeros([output_length, out_ch, inp.shape[2]], dtype=np.float32)

    for patch_id in range(patches):
        inp_ = inp_padded[patch_id * shift: patch_id *
                          shift + patch_length, :, :]
        inp_ = np.expand_dims(inp_, axis=0)
        if ch_flip_average:
            inp_ = np.concatenate([inp_, inp_[:, :, ::-1, :]], axis=0)

        out_ = d3_net(nn.Variable.from_numpy_array(inp_), hparams, test=True)
        out_.forward(clear_buffer=True)
        out_ = out_.data.data

        if ch_flip_average:
            out_ = (out_[0] + out_[1, :, ::-1])*0.5
        else:
            out_ = out_[0]
        output[patch_id * shift: (patch_id + scale)
               * shift, :, :] += out_[:shift * scale, :, :]

    output = output[shift * (scale - 1):shift *
                    (scale - 1) + inp.shape[0]] / scale
    return output
