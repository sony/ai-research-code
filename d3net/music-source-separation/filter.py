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


def estimate_spatial_covariance(stft, eps):
    '''
    Function that estimates the PSD(Power Spectral Density) and spatial covariance matrix.
    Inputs:
      `stft`: frames x channels x freq_bins
      'eps' : small constant (to avoid numerical problems)
    Outputs:
      `psd`   :  frames x freq_bins
      `s_covariance`   : channels x channels x freq_bins
    '''

    # estimate PSD (we add a small `eps` to avoid numerical problems)
    psd = np.mean(np.square(np.abs(stft)), axis=1)
    psd += psd.max() * eps

    # estimate spatial covariance matrix
    s_covariance = np.sum(np.expand_dims(stft, axis=2) *
                          np.conj(np.expand_dims(stft, axis=1)), axis=0)
    s_covariance /= np.sum(psd, axis=0) + np.finfo(np.float64).eps
    return psd, s_covariance


def apply_mwf(stfts, stft_mixture=None, num_mwf_iterations=1):
    '''
    Apply Multichannel Wiener Filter
    Inputs:
      `stft`: Sources STFTs
      'stft_mixture' : Mixture STFT
    Outputs:
      `stfts` :  Sources STFTs with Multichannel Wiener Filter applied
    '''

    # define small constant (to avoid numerical problems)
    eps = 1e-10
    psds = {}
    s_covariances = {}
    instruments = list(stfts.keys())

    # compute inverse covariance matrix of mix
    stft = stfts[instruments[0]]
    s_covariances['mix'] = np.zeros(
        (stft.shape[0], stft.shape[1], stft.shape[1], stft.shape[2]), dtype=np.complex128)

    for idx in instruments:
        # estimate PSD and spatial covariance matrix
        psds[idx], s_covariances[idx] = estimate_spatial_covariance(
            stfts[idx], eps)
        s_covariances['mix'] += psds[idx][:, np.newaxis,
                                          np.newaxis, :] * s_covariances[idx][np.newaxis, :, :, :]

    inv_mix_cs = np.zeros_like(s_covariances['mix'])

    # 0. compute determinant for each s_covariances['mix']
    det = s_covariances['mix'][:, 0, 0, :] * s_covariances['mix'][:, 1, 1,
                                                                  :] - s_covariances['mix'][:, 1, 0, :] * s_covariances['mix'][:, 0, 1, :]

    # 1. compute trace of each s_covariances['mix']^T * s_covariances['mix'] (needed for pseudo-inverse)
    trace = np.sum(np.square(np.abs(s_covariances['mix'])), axis=(
        1, 2)) + np.finfo(np.float64).eps

    # 1. handle case of invertible 2x2 matrix
    idx_inv1, idx_inv2 = np.nonzero(np.abs(det) >= eps)
    inv_mix_cs[idx_inv1, 0, 0,
               idx_inv2] = s_covariances['mix'][idx_inv1, 1, 1, idx_inv2]
    inv_mix_cs[idx_inv1, 1, 1,
               idx_inv2] = s_covariances['mix'][idx_inv1, 0, 0, idx_inv2]
    inv_mix_cs[idx_inv1, 0, 1, idx_inv2] = - \
        s_covariances['mix'][idx_inv1, 0, 1, idx_inv2]
    inv_mix_cs[idx_inv1, 1, 0, idx_inv2] = - \
        s_covariances['mix'][idx_inv1, 1, 0, idx_inv2]
    inv_mix_cs[idx_inv1, :, :, idx_inv2] /= det[idx_inv1,
                                                np.newaxis, np.newaxis, idx_inv2]

    # 2. handle case of rank-1 matrix
    idx_non_inv1, idx_non_inv2 = np.nonzero(np.abs(det) < eps)
    inv_mix_cs[idx_non_inv1, 0, 0, idx_non_inv2] = np.conj(
        s_covariances['mix'][idx_non_inv1, 0, 0, idx_non_inv2])
    inv_mix_cs[idx_non_inv1, 1, 1, idx_non_inv2] = np.conj(
        s_covariances['mix'][idx_non_inv1, 1, 1, idx_non_inv2])
    inv_mix_cs[idx_non_inv1, 0, 1, idx_non_inv2] = np.conj(
        s_covariances['mix'][idx_non_inv1, 1, 0, idx_non_inv2])
    inv_mix_cs[idx_non_inv1, 1, 0, idx_non_inv2] = np.conj(
        s_covariances['mix'][idx_non_inv1, 0, 1, idx_non_inv2])
    inv_mix_cs[idx_non_inv1, :, :, idx_non_inv2] /= trace[idx_non_inv1,
                                                          np.newaxis, np.newaxis, idx_non_inv2]

    # compute new STFTs
    for idx in instruments:
        stfts[idx] = psds[idx][:, np.newaxis, :] * \
            np.sum(s_covariances[idx][np.newaxis, :, :, :] * np.sum(inv_mix_cs * stft_mixture[:, np.newaxis, :, :],
                                                                    axis=2)[:, np.newaxis, :, :], axis=2)
    return stfts
